from typing import Any
from unittest.mock import Mock

import pytest

from langchain_aws import AmazonKendraRetriever
from langchain_aws.retrievers.kendra import RetrieveResultItem


@pytest.fixture
def mock_client() -> Mock:
    mock_client = Mock()
    return mock_client


@pytest.fixture
def retriever(mock_client: Any) -> AmazonKendraRetriever:
    return AmazonKendraRetriever(
        index_id="test_index_id", client=mock_client, top_k=3, min_score_confidence=0.6
    )


def test_get_relevant_documents(retriever, mock_client) -> None:  # type: ignore[no-untyped-def]
    # Mock data for Kendra response
    mock_retrieve_result = {
        "QueryId": "test_query_id",
        "ResultItems": [
            RetrieveResultItem(
                Id="doc1",
                DocumentId="doc1",
                DocumentURI="https://example.com/doc1",
                DocumentTitle="Document 1",
                Content="This is the content of Document 1.",
                ScoreAttributes={"ScoreConfidence": "HIGH"},
            ),
            RetrieveResultItem(
                Id="doc2",
                DocumentId="doc2",
                DocumentURI="https://example.com/doc2",
                DocumentTitle="Document 2",
                Content="This is the content of Document 2.",
                ScoreAttributes={"ScoreConfidence": "MEDIUM"},
            ),
            RetrieveResultItem(
                Id="doc3",
                DocumentId="doc3",
                DocumentURI="https://example.com/doc3",
                DocumentTitle="Document 3",
                Content="This is the content of Document 3.",
                ScoreAttributes={"ScoreConfidence": "HIGH"},
            ),
        ],
    }

    mock_client.retrieve.return_value = mock_retrieve_result

    query = "test query"

    docs = retriever.invoke(query)

    # Only documents with confidence score of HIGH are returned
    assert len(docs) == 2
    assert docs[0].page_content == (
        "Document Title: Document 1\nDocument Excerpt: \n"
        "This is the content of Document 1.\n"
    )
    assert docs[1].page_content == (
        "Document Title: Document 3\nDocument Excerpt: \n"
        "This is the content of Document 3.\n"
    )

    # Assert that the mock methods were called with the expected arguments
    mock_client.retrieve.assert_called_with(
        IndexId="test_index_id", QueryText="test query", PageSize=3
    )
    mock_client.query.assert_not_called()
