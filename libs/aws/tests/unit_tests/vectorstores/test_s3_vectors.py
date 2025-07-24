import math
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_aws.retrievers.s3_vectors import AmazonS3VectorsRetriever
from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_embedding():
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    def embed_documents(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

    embedding.embed_documents.side_effect = embed_documents
    return embedding


@pytest.fixture
def vector_store(mock_embedding, mock_client):
    return AmazonS3Vectors(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding=mock_embedding,
        client=mock_client,
    )


def test_add_texts(vector_store):
    vector_store.client.get_index.return_value = {
        "vectorBucketName": vector_store.vector_bucket_name,
        "indexName": vector_store.index_name,
    }
    texts = ["text1", "text2", "text3"]
    metadatas = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]
    ids = ["id1", "id2", "id3"]
    result_ids = vector_store.add_texts(texts, metadatas, ids=ids)
    vector_store.embeddings.embed_documents.assert_called_once_with(texts)
    vector_store.client.get_index.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
    )
    vector_store.client.create_index.assert_not_called()
    vector_store.client.put_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        vectors=[
            {
                "key": "id1",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {"_page_content": "text1", "key1": "value1"},
            },
            {
                "key": "id2",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {"_page_content": "text2", "key2": "value2"},
            },
            {
                "key": "id3",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {"_page_content": "text3", "key3": "value3"},
            },
        ],
    )
    assert result_ids == ids


def test_add_texts_with_create_index(vector_store):
    vector_store.client.get_index.side_effect = ClientError(
        {"Error": {"Code": "NotFoundException"}}, ""
    )
    texts = ["text1", "text2"]
    result_ids = vector_store.add_texts(texts)
    vector_store.client.create_index.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        dataType="float32",
        dimension=4,
        distanceMetric="cosine",
    )
    assert len(result_ids) == len(texts)


def test_add_texts_with_create_index_failed(vector_store):
    with pytest.raises(ClientError):
        vector_store.client.get_index.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException"}}, ""
        )
        texts = ["text1", "text2"]
        vector_store.add_texts(texts)


def test_add_texts_with_create_index_and_non_filterable_metadata_keys(vector_store):
    vector_store.non_filterable_metadata_keys = ["non_filterable_key"]
    vector_store.client.get_index.side_effect = ClientError(
        {"Error": {"Code": "NotFoundException"}}, ""
    )
    texts = ["text1", "text2"]
    result_ids = vector_store.add_texts(texts)
    vector_store.client.create_index.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        dataType="float32",
        dimension=4,
        distanceMetric="cosine",
        metadataConfiguration={
            "nonFilterableMetadataKeys": ["non_filterable_key"],
        },
    )
    assert len(result_ids) == len(texts)


def test_add_texts_invalid_metadatas_length(vector_store):
    texts = ["text1", "text2"]
    metadatas = [{"meta": "a"}]
    with pytest.raises(ValueError):
        vector_store.add_texts(texts, metadatas=metadatas)


def test_add_texts_invalid_metadatas_type(vector_store):
    texts = ["text1"]
    metadatas = ["not a dict"]
    with pytest.raises(ValueError):
        vector_store.add_texts(texts, metadatas=metadatas)


def test_add_texts_invalid_ids_length(vector_store):
    texts = ["text1", "text2"]
    ids = ["id1"]
    with pytest.raises(ValueError):
        vector_store.add_texts(texts, ids=ids)


def test_add_texts_without_page_content_metadata_key(vector_store):
    vector_store.page_content_metadata_key = None
    vector_store.client.get_index.side_effect = ClientError(
        {"Error": {"Code": "NotFoundException"}}, ""
    )

    vector_store.add_texts(["text1", "text2"], ids=["id1", "id2"])
    vector_store.client.put_vectors.assert_called_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        vectors=[
            {
                "key": "id1",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {},
            },
            {
                "key": "id2",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {},
            },
        ],
    )


def test_add_texts_without_page_content_metadata_key_with_metadata(vector_store):
    vector_store.page_content_metadata_key = None
    vector_store.client.get_index.side_effect = ClientError(
        {"Error": {"Code": "NotFoundException"}}, ""
    )

    vector_store.add_texts(
        ["text1", "text2"],
        metadatas=[{"key1": "value1"}, {"key2": "value2"}],
        ids=["id1", "id2"],
    )
    vector_store.client.put_vectors.assert_called_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        vectors=[
            {
                "key": "id1",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {"key1": "value1"},
            },
            {
                "key": "id2",
                "data": {"float32": [0.1, 0.2, 0.3, 0.4]},
                "metadata": {"key2": "value2"},
            },
        ],
    )


def test_delete_all(vector_store):
    result = vector_store.delete()
    assert result is True
    vector_store.client.delete_index.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
    )
    vector_store.client.delete_vectors.assert_not_called()


def test_delete_by_ids(vector_store):
    result = vector_store.delete(["id1", "id2", "id3"])
    assert result is True
    vector_store.client.delete_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        keys=["id1", "id2", "id3"],
    )
    vector_store.client.delete_index.assert_not_called()


def test_delete_by_ids_with_batch(vector_store):
    result = vector_store.delete(["id1", "id2", "id3", "id4", "id5"], batch_size=2)
    assert result is True
    assert (
        vector_store.client.delete_vectors.call_args.kwargs["vectorBucketName"]
        == vector_store.vector_bucket_name
    )
    assert (
        vector_store.client.delete_vectors.call_args.kwargs["indexName"]
        == vector_store.index_name
    )
    assert vector_store.client.delete_vectors.call_count == 3
    assert vector_store.client.delete_vectors.call_args_list[0].kwargs["keys"] == [
        "id1",
        "id2",
    ]
    assert vector_store.client.delete_vectors.call_args_list[1].kwargs["keys"] == [
        "id3",
        "id4",
    ]
    assert vector_store.client.delete_vectors.call_args_list[2].kwargs["keys"] == [
        "id5"
    ]
    vector_store.client.delete_index.assert_not_called()


def test_get_by_ids(vector_store):
    vector_store.client.get_vectors.return_value = {
        "vectors": [
            {
                "key": "id2",
                "metadata": {"_page_content": "text2", "key2": "value2"},
            },
            {
                "key": "id1",
                "metadata": {"_page_content": "text1", "key1": "value1"},
            },
        ]
    }

    docs = vector_store.get_by_ids(
        [
            "id1",
            "id2",
            "id1",
        ]
    )
    assert docs == [
        Document("text1", id="id1", metadata={"key1": "value1"}),
        Document("text2", id="id2", metadata={"key2": "value2"}),
        Document("text1", id="id1", metadata={"key1": "value1"}),
    ]
    # Ensure different instances
    assert id(docs[0].metadata) != id(docs[2].metadata)
    vector_store.client.get_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        keys=["id1", "id2", "id1"],
        returnData=False,
        returnMetadata=True,
    )


def test_get_by_ids_invalid(vector_store):
    vector_store.client.get_vectors.return_value = {
        "vectors": [
            {
                "key": "id1",
                "metadata": {"_page_content": "text1", "key1": "value1"},
            },
            {
                "key": "id2",
                "metadata": {"_page_content": "text2", "key2": "value2"},
            },
        ]
    }
    with pytest.raises(ValueError):
        vector_store.get_by_ids(["id42"])


def test_get_by_ids_empty(vector_store):
    docs = vector_store.get_by_ids([])
    vector_store.client.get_vectors.asssert_not_called()
    assert docs == []


def test_get_by_ids_with_batch(vector_store):
    vector_store.client.get_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "text"}},
            {"key": "id2", "metadata": {"_page_content": "text"}},
        ]
    }
    docs = vector_store.get_by_ids(
        ["id1", "id2", "id1", "id2", "id1", "id2"], batch_size=2
    )
    assert len(docs) == 6
    assert (
        vector_store.client.get_vectors.call_args.kwargs["vectorBucketName"]
        == vector_store.vector_bucket_name
    )
    assert (
        vector_store.client.get_vectors.call_args.kwargs["indexName"]
        == vector_store.index_name
    )
    assert vector_store.client.get_vectors.call_args.kwargs["returnData"] is False
    assert vector_store.client.get_vectors.call_count == 3
    assert vector_store.client.get_vectors.call_args_list[0].kwargs["keys"] == [
        "id1",
        "id2",
    ]
    assert vector_store.client.get_vectors.call_args_list[1].kwargs["keys"] == [
        "id1",
        "id2",
    ]
    assert vector_store.client.get_vectors.call_args_list[2].kwargs["keys"] == [
        "id1",
        "id2",
    ]


def test_similarity_search(vector_store):
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1"},
            {"key": "id2"},
        ]
    }
    result = vector_store.similarity_search(
        "query text", k=2, filter={"genre": {"$eq": "family"}}
    )
    vector_store.embeddings.embed_query.assert_called_once_with("query text")
    vector_store.client.query_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        topK=2,
        queryVector={"float32": [0.1, 0.2, 0.3, 0.4]},
        filter={"genre": {"$eq": "family"}},
        returnMetadata=True,
        returnDistance=False,
    )
    assert result == [Document("", id="id1"), Document("", id="id2")]


def test_similarity_search_with_score(vector_store):
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "text1"}, "distance": 0.1},
            {"key": "id2", "metadata": {"_page_content": "text2"}, "distance": 0.2},
        ]
    }
    results = vector_store.similarity_search_with_score(
        "query text", k=30, filter={"genre": {"$eq": "family"}}
    )
    vector_store.embeddings.embed_query.assert_called_once_with("query text")
    vector_store.client.query_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        topK=30,
        queryVector={"float32": [0.1, 0.2, 0.3, 0.4]},
        filter={"genre": {"$eq": "family"}},
        returnMetadata=True,
        returnDistance=True,
    )
    assert results == [
        (Document("text1", id="id1"), 0.1),
        (Document("text2", id="id2"), 0.2),
    ]


def test_similarity_search_by_vector(vector_store):
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "text1"}},
            {"key": "id2", "metadata": {"_page_content": "text2"}},
        ]
    }
    embedding = [0.1, 0.2, 0.3, 0.4]
    results = vector_store.similarity_search_by_vector(
        embedding, k=2, filter={"genre": {"$eq": "family"}}
    )
    vector_store.client.query_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        topK=2,
        queryVector={"float32": embedding},
        filter={"genre": {"$eq": "family"}},
        returnMetadata=True,
        returnDistance=False,
    )
    assert results == [Document("text1", id="id1"), Document("text2", id="id2")]


def test_as_retriever_returns_retriever(vector_store):
    retriever = vector_store.as_retriever()
    assert isinstance(retriever, AmazonS3VectorsRetriever)


def test_from_texts():
    texts = []
    vector_store = AmazonS3Vectors.from_texts(
        texts,
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding=mock_embedding,
        client=mock_client,
    )
    assert isinstance(vector_store, AmazonS3Vectors)


@patch("langchain_aws.vectorstores.s3_vectors.base.create_aws_client")
def test_from_texts_without_client(mock_create_aws_client):
    mock_create_aws_client.return_value = MagicMock()

    texts = []
    AmazonS3Vectors.from_texts(
        texts,
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding=mock_embedding,
        region_name="us-west-2",
        credentials_profile_name="test-profile",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
        endpoint_url="https://example.com",
        config={"key": "value"},
    )
    mock_create_aws_client.assert_called_once_with(
        "s3vectors",
        region_name="us-west-2",
        credentials_profile_name="test-profile",
        aws_access_key_id=SecretStr("test-access-key"),
        aws_secret_access_key=SecretStr("test-secret-key"),
        aws_session_token=SecretStr("test-session-token"),
        endpoint_url="https://example.com",
        config={"key": "value"},
    )


def test_similarity_search_with_relevance_scores(vector_store):
    # cosine distance to similarity score [0, 1]
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "similar"}, "distance": 0.0},
            {
                "key": "id2",
                "metadata": {"_page_content": "not similar"},
                "distance": 1.0,
            },
        ]
    }

    results = vector_store.similarity_search_with_relevance_scores("query text")

    assert results == [
        (Document("similar", id="id1"), 1.0),
        (Document("not similar", id="id2"), 0.0),
    ]


def test_similarity_search_with_relevance_scores_euclidean(vector_store):
    # euclidean distance to similarity score [0, 1]
    vector_store.distance_metric = "euclidean"
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "similar"}, "distance": 0.0},
            {
                "key": "id2",
                "metadata": {"_page_content": "not similar"},
                "distance": math.sqrt(4096),
            },
        ]
    }

    results = vector_store.similarity_search_with_relevance_scores("query text")

    assert results == [
        (Document("similar", id="id1"), 1.0),
        (Document("not similar", id="id2"), 0.0),
    ]


def test_similarity_search_with_relevance_scores_custom(vector_store):
    vector_store.relevance_score_fn = lambda distance: 1.0 - distance / 2.0
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "similar"}, "distance": 0.0},
            {
                "key": "id2",
                "metadata": {"_page_content": "not similar"},
                "distance": 1.0,
            },
        ]
    }
    results = vector_store.similarity_search_with_relevance_scores("query text")
    assert results == [
        (Document("similar", id="id1"), 1.0),
        (Document("not similar", id="id2"), 0.5),
    ]


def test_similarity_search_with_relevance_scores_invalid(vector_store):
    vector_store.distance_metric = "unknown_metric"
    with pytest.raises(ValueError):
        vector_store.similarity_search_with_relevance_scores("query text")
