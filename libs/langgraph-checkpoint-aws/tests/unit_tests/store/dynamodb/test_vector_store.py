"""Unit tests for DynamoDBStore vector-search support.

Covers the behaviour added on top of the base key-value store:
- distance -> relevance score conversion (Vector Index returns a distance)
- timestamp parsing tolerance on the search projection
- atomic put via update_item (no read-before-write / TOCTOU)
- embedding dimension validation
- on-demand throughput cap only applied when explicitly requested
- filtered vector search over-fetches so it doesn't under-return
"""

import os
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from langgraph_checkpoint_aws.store import DynamoDBStore
from langgraph_checkpoint_aws.store.dynamodb.exceptions import (
    TableCreationError,
    ValidationError,
)
from langgraph_checkpoint_aws.store.dynamodb.vector import (
    _FILTER_OVERFETCH,
    _VECTOR_INDEX_NAME,
)


class FakeEmbeddings:
    """Minimal embeddings provider satisfying the Embeddings protocol."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim


def _dynamo_item(pk: str, sk: str, text: str) -> dict:
    return {
        "PK": {"S": pk},
        "SK": {"S": sk},
        "value": {"M": {"text": {"S": text}}},
        "created_at": {"S": "2026-07-17T00:00:00+00:00"},
        "updated_at": {"S": "2026-07-17T00:00:00+00:00"},
    }


@pytest.fixture
def mock_client() -> Mock:
    client = Mock()
    client.get_waiter = Mock(return_value=Mock())
    return client


def _make_store(mock_client, *, index=None, **kwargs) -> DynamoDBStore:
    with patch(
        "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
        return_value=mock_client,
    ):
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            return DynamoDBStore(table_name="t", index=index, **kwargs)


def _store_with_index(mock_client, *, dim=4, distance="COSINE", embed_dim=None):
    embed = FakeEmbeddings(embed_dim or dim)
    return _make_store(
        mock_client,
        index={"embed": embed, "dims": dim, "distance_function": distance},
    )


class TestDistanceToScore:
    """Vector Index returns a distance (lower==closer); LangGraph wants a
    score (higher==more relevant)."""

    def test_cosine(self, mock_client):
        store = _store_with_index(mock_client, distance="COSINE")
        assert store._distance_to_score(0.0) == pytest.approx(1.0)  # exact match
        assert store._distance_to_score(1.0) == pytest.approx(0.0)  # orthogonal
        assert store._distance_to_score(0.25) == pytest.approx(0.75)

    def test_euclidean_monotonic_decreasing(self, mock_client):
        store = _store_with_index(mock_client, distance="EUCLIDEAN")
        assert store._distance_to_score(0.0) == pytest.approx(1.0)
        assert store._distance_to_score(1.0) == pytest.approx(0.5)
        assert store._distance_to_score(3.0) == pytest.approx(0.25)

    def test_dot_product_preserves_ranking(self, mock_client):
        store = _store_with_index(mock_client, distance="DOT_PRODUCT")
        # For DOT_PRODUCT the service Score is already higher == more
        # similar (and may be negative); it must pass through unchanged.
        assert store._distance_to_score(2.0) > store._distance_to_score(1.0)
        assert store._distance_to_score(-1.0) == -1.0

    def test_none_distance(self, mock_client):
        store = _store_with_index(mock_client)
        assert store._distance_to_score(None) == 0.0


class TestParseTimestamp:
    def test_empty_falls_back_to_epoch(self, mock_client):
        store = _make_store(mock_client)
        assert store._parse_ts("").timestamp() == 0
        assert store._parse_ts(None).timestamp() == 0

    def test_valid_iso(self, mock_client):
        store = _make_store(mock_client)
        assert store._parse_ts("2026-07-17T00:00:00+00:00").year == 2026


class TestAtomicPut:
    """put must be a single atomic update_item, not read-before-write."""

    def test_put_uses_update_item_with_if_not_exists(self, mock_client):
        store = _make_store(mock_client)
        store.put(("users", "alice"), "prefs", {"theme": "dark"})

        mock_client.update_item.assert_called_once()
        # No read-before-write: get_item must not be used to preserve created_at
        mock_client.get_item.assert_not_called()
        mock_client.put_item.assert_not_called()

        kwargs = mock_client.update_item.call_args.kwargs
        expr = kwargs["UpdateExpression"]
        assert "if_not_exists(created_at" in expr
        assert kwargs["Key"]["PK"]["S"] == "users:alice"
        assert kwargs["Key"]["SK"]["S"] == "prefs"

    def test_put_embeds_and_validates_dims(self, mock_client):
        store = _store_with_index(mock_client, dim=4)
        store.put(("u", "a"), "m1", {"text": "hello"})
        kwargs = mock_client.update_item.call_args.kwargs
        # embedding stored as list of N under the vector attribute alias
        assert ":embedding" in kwargs["ExpressionAttributeValues"]
        emb = kwargs["ExpressionAttributeValues"][":embedding"]["L"]
        assert len(emb) == 4

    def test_put_dim_mismatch_raises(self, mock_client):
        # index configured for 4 dims but model emits 3
        store = _store_with_index(mock_client, dim=4, embed_dim=3)
        with pytest.raises(ValidationError, match="dimension mismatch"):
            store.put(("u", "a"), "m1", {"text": "hello"})

    def test_put_without_ttl_removes_stale_expiry(self, mock_client):
        """A re-put with no TTL must clear expires_at left by a prior put.

        put_item (full replace) cleared it implicitly; the update_item port
        must REMOVE it or the new value inherits the old expiry and vanishes.
        """
        store = _make_store(mock_client)
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        expr = mock_client.update_item.call_args.kwargs["UpdateExpression"]
        assert "REMOVE" in expr
        assert "expires_at" in expr.split("REMOVE", 1)[1]

    def test_put_with_ttl_sets_expiry_and_does_not_remove_it(self, mock_client):
        store = _make_store(
            mock_client, ttl={"default_ttl": 60, "refresh_on_read": False}
        )
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        expr = mock_client.update_item.call_args.kwargs["UpdateExpression"]
        assert "expires_at = :expires_at" in expr
        remove_clause = expr.split("REMOVE", 1)[1] if "REMOVE" in expr else ""
        assert "expires_at" not in remove_clause

    def test_put_without_embeddable_text_removes_stale_embedding(self, mock_client):
        """Re-putting a value with nothing to embed must drop the old vector.

        Otherwise semantic search keeps matching the item by its previous
        content.
        """
        store = _make_store(
            mock_client,
            index={"embed": FakeEmbeddings(4), "dims": 4, "fields": ["text"]},
        )
        # fields=["text"]; new value has no "text" field so nothing to embed
        store.put(("u", "a"), "m1", {"other": 123})
        kwargs = mock_client.update_item.call_args.kwargs
        expr = kwargs["UpdateExpression"]
        assert ":embedding" not in kwargs["ExpressionAttributeValues"]
        assert "REMOVE" in expr
        assert "#embedding" in expr.split("REMOVE", 1)[1]


class TestCreateTableThroughput:
    def test_no_cap_by_default(self, mock_client):
        store = _make_store(mock_client)
        store._create_table()
        params = mock_client.create_table.call_args.kwargs
        assert params["BillingMode"] == "PAY_PER_REQUEST"
        assert "OnDemandThroughput" not in params

    def test_cap_applied_when_requested(self, mock_client):
        store = _make_store(mock_client, max_read_capacity_units=100)
        store._create_table()
        params = mock_client.create_table.call_args.kwargs
        assert params["OnDemandThroughput"]["MaxReadRequestUnits"] == 100

    def test_vector_index_included_when_configured(self, mock_client):
        store = _store_with_index(mock_client, dim=8)
        # _create_table waits for the index to become ACTIVE via describe_table
        mock_client.describe_table.return_value = {
            "Table": {
                "VectorIndexes": [
                    {"IndexName": "memory-vector-index", "IndexStatus": "ACTIVE"}
                ]
            }
        }
        store._create_table()
        params = mock_client.create_table.call_args.kwargs
        vi = params["VectorIndexes"][0]
        assert vi["Dimensions"] == 8
        assert vi["DistanceFunction"] == "COSINE"
        assert vi["VectorAttribute"]["AttributeName"] == "embedding"


class TestVectorSearch:
    def _resp(self):
        return {
            "SearchResults": [
                {"Item": _dynamo_item("u:a", "m1", "x"), "Score": 0.0},
                {"Item": _dynamo_item("u:a", "m2", "y"), "Score": 1.0},
            ]
        }

    def test_scores_converted_and_ordered(self, mock_client):
        store = _store_with_index(mock_client, distance="COSINE")
        mock_client.search_vectors.return_value = self._resp()
        results = store.search(("u", "a"), query="hi", limit=5)
        # distance 0.0 -> score 1.0 ; distance 1.0 -> score 0.0
        assert [round(r.score, 3) for r in results] == [1.0, 0.0]

    def test_topk_equals_limit_plus_offset_without_filter(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.search_vectors.return_value = {"SearchResults": []}
        store.search(("u", "a"), query="hi", limit=5, offset=2)
        assert mock_client.search_vectors.call_args.kwargs["TopK"] == 7

    def test_overfetch_when_filtered(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.search_vectors.return_value = {"SearchResults": []}
        store.search(("u", "a"), query="hi", filter={"k": "v"}, limit=3)
        assert (
            mock_client.search_vectors.call_args.kwargs["TopK"] == 3 * _FILTER_OVERFETCH
        )

    def test_error_is_raised_not_swallowed(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.search_vectors.side_effect = RuntimeError("throttled")
        with pytest.raises(RuntimeError, match="throttled"):
            store.search(("u", "a"), query="hi", limit=5)


class TestTopKBounds:
    """SearchVectors has no pagination and the service caps TopK at 100
    (live error: "must be between 1 and 100 inclusive")."""

    def test_limit_plus_offset_over_cap_raises(self, mock_client):
        store = _store_with_index(mock_client)
        with pytest.raises(ValueError, match="top 100 matches"):
            store.search(("ns",), query="q", limit=80, offset=40)
        mock_client.search_vectors.assert_not_called()

    def test_filtered_overfetch_is_clamped_to_cap(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.search_vectors.return_value = {"SearchResults": []}
        store.search(("ns",), query="q", limit=20, filter={"a": 1})
        # 20 * _FILTER_OVERFETCH would be 200; must be clamped to service max
        assert mock_client.search_vectors.call_args.kwargs["TopK"] == 100

    def test_at_cap_boundary_allowed(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.search_vectors.return_value = {"SearchResults": []}
        store.search(("ns",), query="q", limit=60, offset=40)
        assert mock_client.search_vectors.call_args.kwargs["TopK"] == 100


def _active(name=_VECTOR_INDEX_NAME, status="ACTIVE"):
    return {"Table": {"VectorIndexes": [{"IndexName": name, "IndexStatus": status}]}}


class TestCreateVectorIndex:
    """_create_vector_index mutates infrastructure; pin its request shape
    and failure modes."""

    def test_update_table_payload_shape(self, mock_client):
        store = _store_with_index(mock_client, dim=8, distance="EUCLIDEAN")
        mock_client.describe_table.return_value = _active()
        store._create_vector_index()
        kwargs = mock_client.update_table.call_args.kwargs
        assert kwargs["TableName"] == "t"
        create = kwargs["VectorIndexUpdates"][0]["Create"]
        assert create["IndexName"] == _VECTOR_INDEX_NAME
        assert create["VectorAttribute"] == {"AttributeName": "embedding"}
        assert create["SearchSchema"] == [
            {"AttributeName": "PK", "SearchSchemaElementType": "HASH"}
        ]
        assert create["Projection"] == {"ProjectionType": "ALL"}
        assert create["Dimensions"] == 8
        assert create["DistanceFunction"] == "EUCLIDEAN"

    def test_client_error_is_reraised(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.update_table.side_effect = ClientError(
            {"Error": {"Code": "ResourceInUseException", "Message": "busy"}},
            "UpdateTable",
        )
        with pytest.raises(ClientError, match="ResourceInUseException"):
            store._create_vector_index()


class TestWaitForVectorIndex:
    def test_returns_once_active(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.describe_table.side_effect = [
            _active(status="CREATING"),
            _active(status="ACTIVE"),
        ]
        with patch("time.sleep") as slept:
            store._wait_for_vector_index()
        assert mock_client.describe_table.call_count == 2
        slept.assert_called_once()

    def test_polls_while_index_absent_then_active(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.describe_table.side_effect = [
            {"Table": {}},  # index not visible yet
            _active(name="other-index"),  # unrelated index only
            _active(status="ACTIVE"),
        ]
        with patch("time.sleep"):
            store._wait_for_vector_index()
        assert mock_client.describe_table.call_count == 3

    def test_timeout_raises_table_creation_error(self, mock_client):
        store = _store_with_index(mock_client)
        mock_client.describe_table.return_value = _active(status="CREATING")
        clock = iter(range(0, 10_000, 300))
        with patch("time.sleep"), patch("time.time", side_effect=lambda: next(clock)):
            with pytest.raises(TableCreationError, match="did not become ACTIVE"):
                store._wait_for_vector_index(timeout=600)


class TestReviewRegressions:
    """Regressions pinned from the pre-PR three-agent critical review."""

    def test_waiter_respects_backfilling(self, mock_client):
        store = _store_with_index(mock_client, dim=4)
        mock_client.describe_table.side_effect = [
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": _VECTOR_INDEX_NAME,
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
                            "IndexName": _VECTOR_INDEX_NAME,
                            "IndexStatus": "ACTIVE",
                            "Backfilling": False,
                        }
                    ]
                }
            },
        ]
        with patch("time.sleep"):
            store._wait_for_vector_index(timeout=60)
        assert mock_client.describe_table.call_count == 2

    def test_waiter_tolerates_missing_status(self, mock_client):
        store = _store_with_index(mock_client, dim=4)
        mock_client.describe_table.side_effect = [
            {"Table": {"VectorIndexes": [{"IndexName": _VECTOR_INDEX_NAME}]}},
            {
                "Table": {
                    "VectorIndexes": [
                        {
                            "IndexName": _VECTOR_INDEX_NAME,
                            "IndexStatus": "ACTIVE",
                        }
                    ]
                }
            },
        ]
        with patch("time.sleep"):
            store._wait_for_vector_index(timeout=60)
        assert mock_client.describe_table.call_count == 2

    def test_nonfinite_embedding_raises_on_put(self, mock_client):
        class NanEmbeddings:
            def embed_documents(self, texts):
                return [[float("nan"), 0.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [float("nan"), 0.0, 0.0, 0.0]

        store = _make_store(mock_client, index={"embed": NanEmbeddings(), "dims": 4})
        with pytest.raises(ValidationError, match="non-finite"):
            store.put(("u", "a"), "m1", {"text": "hello"})

    def test_field_mode_embeds_json_not_repr(self, mock_client):
        emb = FakeEmbeddings(4)
        seen = []
        orig = emb.embed_documents

        def capture(texts):
            seen.extend(texts)
            return orig(texts)

        emb.embed_documents = capture
        store = _make_store(
            mock_client, index={"embed": emb, "dims": 4, "fields": ["data"]}
        )
        store.put(("u", "a"), "m1", {"data": {"a": 1}})
        assert seen == ['{"a": 1}']  # JSON, not Python repr "{'a': 1}"

    def test_fallback_query_fetches_limit_plus_offset(self, mock_client):
        from langgraph.store.base import SearchOp

        mock_client.query.return_value = {"Items": []}
        store = _make_store(mock_client)  # no vector index
        store._batch_search_op(
            SearchOp(namespace_prefix=("u",), filter=None, limit=10, offset=5)
        )
        assert mock_client.query.call_args.kwargs["Limit"] == 15

    def test_query_without_index_warns(self, mock_client, caplog):
        import logging as _logging

        from langgraph.store.base import SearchOp

        mock_client.query.return_value = {"Items": []}
        store = _make_store(mock_client)
        with caplog.at_level(_logging.WARNING):
            store._batch_search_op(
                SearchOp(
                    namespace_prefix=("u",),
                    filter=None,
                    limit=5,
                    offset=0,
                    query="find me",
                )
            )
        assert any("no vector index" in r.message for r in caplog.records)

    def test_vector_search_refreshes_ttl(self, mock_client):
        store = _make_store(
            mock_client,
            index={"embed": FakeEmbeddings(4), "dims": 4},
            ttl={"default_ttl": 60, "refresh_on_read": True},
        )
        mock_client.search_vectors.return_value = {
            "SearchResults": [
                {
                    "Item": {
                        "PK": {"S": "u:a"},
                        "SK": {"S": "m1"},
                        "value": {"M": {}},
                        "created_at": {"S": "2026-01-01T00:00:00+00:00"},
                        "updated_at": {"S": "2026-01-01T00:00:00+00:00"},
                    },
                    "Score": 0.1,
                }
            ]
        }
        from langgraph.store.base import SearchOp

        with patch.object(store, "_refresh_ttl") as refresh:
            store._batch_search_op(
                SearchOp(
                    namespace_prefix=("u", "a"),
                    filter=None,
                    limit=5,
                    offset=0,
                    query="hello",
                    refresh_ttl=True,
                )
            )
            refresh.assert_called_once_with("u:a", "m1")
