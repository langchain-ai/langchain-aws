# mypy: disable-error-code="no-untyped-def"
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_aws.vectorstores.s3_vectors.base import AmazonS3Vectors


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_embedding() -> MagicMock:
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    def embed_documents(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

    embedding.embed_documents.side_effect = embed_documents
    return embedding


@pytest.fixture
def vector_store(mock_embedding, mock_client) -> AmazonS3Vectors:
    return AmazonS3Vectors(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding=mock_embedding,
        client=mock_client,
    )


def test_invoke(vector_store) -> None:
    vector_store.client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "text1", "key1": "value1"}},
            {"key": "id2", "metadata": {"_page_content": "text2", "key2": "value2"}},
        ]
    }
    retriever = vector_store.as_retriever()
    results = retriever.invoke("query text")
    vector_store.embeddings.embed_query.assert_called_once_with("query text")
    vector_store.client.query_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        topK=4,
        filter=None,
        queryVector={"float32": [0.1, 0.2, 0.3, 0.4]},
        returnMetadata=True,
        returnDistance=False,
    )
    assert results == [
        Document("text1", id="id1", metadata={"key1": "value1"}),
        Document("text2", id="id2", metadata={"key2": "value2"}),
    ]


def test_invoke_with_query_embedding(mock_client) -> None:
    """Test that a separate query embedding can be used for queries."""
    # Create separate embeddings for documents and queries
    doc_embedding = MagicMock()
    doc_embedding.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    query_embedding = MagicMock()
    query_embedding.embed_query.return_value = [0.5, 0.6, 0.7, 0.8]

    # Create vector store with both embeddings
    vector_store = AmazonS3Vectors(
        vector_bucket_name="test-bucket",
        index_name="test-index",
        embedding=doc_embedding,
        query_embedding=query_embedding,
        client=mock_client,
    )

    mock_client.query_vectors.return_value = {
        "vectors": [
            {"key": "id1", "metadata": {"_page_content": "text1", "key1": "value1"}},
        ]
    }

    # Create retriever and invoke
    retriever = vector_store.as_retriever()
    results = retriever.invoke("query text")

    # Verify the query embedding was used, not the document embedding
    query_embedding.embed_query.assert_called_once_with("query text")
    doc_embedding.embed_query.assert_not_called()

    # Verify the query was made with the query embedding's vector
    mock_client.query_vectors.assert_called_once_with(
        vectorBucketName=vector_store.vector_bucket_name,
        indexName=vector_store.index_name,
        topK=4,
        filter=None,
        queryVector={"float32": [0.5, 0.6, 0.7, 0.8]},
        returnMetadata=True,
        returnDistance=False,
    )
    assert results == [
        Document("text1", id="id1", metadata={"key1": "value1"}),
    ]
