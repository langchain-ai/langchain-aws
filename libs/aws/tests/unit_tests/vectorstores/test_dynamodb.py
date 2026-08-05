"""Unit tests for DynamoDBVectorStore.

Fully mocked boto3 client; no live AWS calls. Covers:
- provisioning (table create, index retrofit, create_table_if_not_exist=False)
- add_texts (embedding, batching, unprocessed-item retry, validation)
- similarity search (TopK bounds, distance passthrough, document mapping)
- relevance score conversion per distance function
- delete / get_by_ids
"""

from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from langchain_aws.vectorstores.dynamodb import DynamoDBVectorStore
from langchain_aws.vectorstores.dynamodb.base import _MAX_TOP_K


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings satisfying the Embeddings interface."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7 + i) for i in range(self.dim)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7 + i) for i in range(self.dim)]


@pytest.fixture
def mock_client() -> Mock:
    client = Mock()
    client.get_waiter = Mock(return_value=Mock())
    client.batch_write_item.return_value = {}
    client.describe_table.return_value = {
        "Table": {
            "VectorIndexes": [
                {"IndexName": "documents-vector-index", "IndexStatus": "ACTIVE"}
            ]
        }
    }
    return client


def _make_store(mock_client: Mock, **kwargs: Any) -> DynamoDBVectorStore:
    return DynamoDBVectorStore(
        table_name="docs",
        embedding=FakeEmbeddings(),
        client=mock_client,
        **kwargs,
    )


class TestProvisioning:
    def test_creates_table_when_absent(self, mock_client: Mock) -> None:
        from botocore.exceptions import ClientError

        mock_client.describe_table.side_effect = [
            ClientError(
                {"Error": {"Code": "ResourceNotFoundException"}}, "DescribeTable"
            ),
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": "documents-vector-index",
                            "IndexStatus": "ACTIVE",
                        }
                    ]
                }
            },
        ]
        store = _make_store(mock_client)
        store.add_texts(["hello"])
        params = mock_client.create_table.call_args.kwargs
        assert params["TableName"] == "docs"
        spec = params["VectorIndexes"][0]
        assert spec["Dimensions"] == 4
        assert spec["DistanceFunction"] == "COSINE"
        # Document store searches globally: no SearchSchema partition pinning
        assert "SearchSchema" not in spec

    def test_retrofits_index_on_existing_table(self, mock_client: Mock) -> None:
        mock_client.describe_table.side_effect = [
            {"Table": {"VectorIndexes": []}},
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": "documents-vector-index",
                            "IndexStatus": "ACTIVE",
                        }
                    ]
                }
            },
        ]
        store = _make_store(mock_client)
        store.add_texts(["hello"])
        params = mock_client.update_table.call_args.kwargs
        assert "Create" in params["VectorIndexUpdates"][0]

    def test_no_autocreate_raises(self, mock_client: Mock) -> None:
        from botocore.exceptions import ClientError

        mock_client.describe_table.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "DescribeTable"
        )
        store = _make_store(mock_client, create_table_if_not_exist=False)
        with pytest.raises(ValueError, match="does not exist"):
            store.add_texts(["hello"])


class TestAddTexts:
    def test_writes_items_with_embedding(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        ids = store.add_texts(["hello", "world"], metadatas=[{"a": 1}, {"a": 2}])
        assert len(ids) == 2
        batch = mock_client.batch_write_item.call_args.kwargs["RequestItems"]["docs"]
        assert len(batch) == 2
        item = batch[0]["PutRequest"]["Item"]
        assert item["page_content"]["S"] == "hello"
        assert len(item["embedding"]["L"]) == 4

    def test_mismatched_metadata_raises(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="metadatas"):
            store.add_texts(["a", "b"], metadatas=[{}])

    def test_mismatched_ids_raises(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="ids"):
            store.add_texts(["a", "b"], ids=["only-one"])

    def test_unprocessed_items_are_retried(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        first = {
            "UnprocessedItems": {"docs": [{"PutRequest": {"Item": {"id": {"S": "x"}}}}]}
        }
        mock_client.batch_write_item.side_effect = [first, {}]
        store.add_texts(["hello"])
        assert mock_client.batch_write_item.call_count == 2

    def test_empty_texts_noop(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        assert store.add_texts([]) == []
        mock_client.batch_write_item.assert_not_called()


class TestSearch:
    def _search_response(self) -> dict:
        return {
            "SearchResults": [
                {
                    "Item": {
                        "id": {"S": "doc1"},
                        "page_content": {"S": "hello"},
                        "metadata": {"M": {"a": {"N": "1"}}},
                    },
                    "Score": 0.25,
                }
            ]
        }

    def test_similarity_search_maps_documents(self, mock_client: Mock) -> None:
        mock_client.search_vectors.return_value = self._search_response()
        store = _make_store(mock_client)
        docs = store.similarity_search("greeting", k=1)
        assert docs[0].id == "doc1"
        assert docs[0].page_content == "hello"
        assert docs[0].metadata == {"a": 1}
        kwargs = mock_client.search_vectors.call_args.kwargs
        assert kwargs["TopK"] == 1
        assert kwargs["IndexName"] == "documents-vector-index"

    def test_with_score_returns_raw_distance(self, mock_client: Mock) -> None:
        mock_client.search_vectors.return_value = self._search_response()
        store = _make_store(mock_client)
        [(_, score)] = store.similarity_search_with_score("greeting", k=1)
        assert score == 0.25

    def test_k_bounds_enforced(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="TopK"):
            store.similarity_search("q", k=_MAX_TOP_K + 1)
        with pytest.raises(ValueError, match="TopK"):
            store.similarity_search("q", k=0)


class TestRelevanceScores:
    def test_cosine(self, mock_client: Mock) -> None:
        fn = _make_store(mock_client)._select_relevance_score_fn()
        assert fn(0.0) == 1.0  # identical
        assert fn(2.0) == 0.0  # opposite

    def test_euclidean(self, mock_client: Mock) -> None:
        store = _make_store(mock_client, distance_function="EUCLIDEAN")
        fn = store._select_relevance_score_fn()
        assert fn(0.0) == 1.0
        assert fn(1.0) == 0.5

    def test_dot_product_preserves_ranking(self, mock_client: Mock) -> None:
        # DOT_PRODUCT Score is a similarity (higher == more similar); the
        # relevance conversion must preserve that ordering and stay in (0, 1).
        store = _make_store(mock_client, distance_function="DOT_PRODUCT")
        fn = store._select_relevance_score_fn()
        assert fn(3.0) > fn(-1.0) > fn(-5.0)
        assert 0.0 < fn(-5.0) and fn(3.0) < 1.0

    def test_override(self, mock_client: Mock) -> None:
        store = _make_store(mock_client, relevance_score_fn=lambda d: 42.0)
        assert store._select_relevance_score_fn()(0.1) == 42.0


class TestDeleteAndGet:
    def test_delete_requires_ids(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="ids is required"):
            store.delete()

    def test_delete_batches(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        assert store.delete(ids=[f"id{i}" for i in range(30)]) is True
        assert mock_client.batch_write_item.call_count == 2  # 25 + 5

    def test_get_by_ids_skips_missing(self, mock_client: Mock) -> None:
        mock_client.batch_get_item.return_value = {
            "Responses": {
                "docs": [
                    {
                        "id": {"S": "a"},
                        "page_content": {"S": "text-a"},
                        "metadata": {"M": {}},
                    }
                ]
            }
        }
        store = _make_store(mock_client)
        docs = store.get_by_ids(["a", "missing"])
        assert len(docs) == 1
        assert docs[0].id == "a"


class TestFromTexts:
    def test_from_texts_provisions_and_adds(self, mock_client: Mock) -> None:
        store = DynamoDBVectorStore.from_texts(
            ["hello", "world"],
            embedding=FakeEmbeddings(),
            table_name="docs",
            client=mock_client,
        )
        assert isinstance(store, DynamoDBVectorStore)
        assert mock_client.batch_write_item.called


class TestReviewRegressions:
    """Regressions pinned from the pre-PR critical review."""

    def test_float_metadata_roundtrips(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        store.add_texts(["hello"], metadatas=[{"score": 0.5, "n": 2, "tags": [1.5]}])
        item = mock_client.batch_write_item.call_args.kwargs["RequestItems"]["docs"][0][
            "PutRequest"
        ]["Item"]
        # Serialized without TypeError; floats became DynamoDB numbers
        meta = item["metadata"]["M"]
        assert meta["score"]["N"] == "0.5"
        assert meta["n"]["N"] == "2"

    def test_metadata_decimals_normalized_on_read(self, mock_client: Mock) -> None:
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {
                    "Item": {
                        "id": {"S": "d1"},
                        "page_content": {"S": "x"},
                        "metadata": {"M": {"score": {"N": "0.5"}, "n": {"N": "2"}}},
                    },
                    "Score": 0.1,
                }
            ]
        }
        store = _make_store(mock_client)
        [doc] = store.similarity_search("q", k=1)
        assert doc.metadata == {"score": 0.5, "n": 2}
        assert isinstance(doc.metadata["score"], float)
        assert isinstance(doc.metadata["n"], int)

    def test_duplicate_ids_rejected_on_add(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="unique"):
            store.add_texts(["a", "b"], ids=["dup", "dup"])

    def test_get_by_ids_tolerates_duplicates(self, mock_client: Mock) -> None:
        mock_client.batch_get_item.return_value = {
            "Responses": {
                "docs": [
                    {
                        "id": {"S": "a"},
                        "page_content": {"S": "t"},
                        "metadata": {"M": {}},
                    }
                ]
            }
        }
        store = _make_store(mock_client)
        docs = store.get_by_ids(["a", "a"])
        assert len(docs) == 1
        keys = mock_client.batch_get_item.call_args.kwargs["RequestItems"]["docs"][
            "Keys"
        ]
        assert keys == [{"id": {"S": "a"}}]

    def test_filter_kwarg_rejected_loudly(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(ValueError, match="filter"):
            store.similarity_search("q", k=1, filter={"genre": "scifi"})
        with pytest.raises(ValueError, match="filter"):
            store.similarity_search_with_score("q", k=1, filter={"a": 1})

    def test_mmr_raises_not_implemented(self, mock_client: Mock) -> None:
        store = _make_store(mock_client)
        with pytest.raises(NotImplementedError):
            store.max_marginal_relevance_search("q", k=1)

    def test_wait_respects_backfilling(self, mock_client: Mock) -> None:
        mock_client.describe_table.side_effect = [
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": "documents-vector-index",
                            "IndexStatus": "ACTIVE",
                            "Backfilling": True,
                        }
                    ]
                }
            },
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": "documents-vector-index",
                            "IndexStatus": "ACTIVE",
                            "Backfilling": False,
                        }
                    ]
                }
            },
        ]
        store = _make_store(mock_client)
        store._wait_for_vector_index(timeout=60)
        assert mock_client.describe_table.call_count == 2

    def test_dims_mismatch_against_existing_index_raises(
        self, mock_client: Mock
    ) -> None:
        mock_client.describe_table.return_value = {
            "Table": {
                "VectorIndexes": [
                    {
                        "IndexName": "documents-vector-index",
                        "IndexStatus": "ACTIVE",
                        "Dimensions": 8,
                    }
                ]
            }
        }
        store = _make_store(mock_client)  # FakeEmbeddings emits 4 dims
        with pytest.raises(ValueError, match="Dimensions"):
            store.add_texts(["hello"])

    def test_by_vector_direct(self, mock_client: Mock) -> None:
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {
                    "Item": {
                        "id": {"S": "d1"},
                        "page_content": {"S": "x"},
                        "metadata": {"M": {}},
                    },
                    "Score": 0.3,
                }
            ]
        }
        store = _make_store(mock_client)
        [(doc, score)] = store.similarity_search_with_score_by_vector(
            [0.0, 1.0, 0.0, 0.1], k=1
        )
        assert doc.id == "d1" and score == 0.3

    def test_add_documents_and_from_documents_path(self, mock_client: Mock) -> None:
        from langchain_core.documents import Document

        store = _make_store(mock_client)
        ids = store.add_documents(
            [Document(page_content="hello", id="doc-1", metadata={"a": 1})]
        )
        assert ids == ["doc-1"]
