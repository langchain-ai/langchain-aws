from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from langchain_aws import AmazonKnowledgeBasesRetriever


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def retriever(mock_client: Mock) -> AmazonKnowledgeBasesRetriever:
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test-knowledge-base",
        client=mock_client,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},  # type: ignore[arg-type]
        min_score_confidence=0.0,
    )


def test_get_relevant_documents(retriever, mock_client) -> None:  # type: ignore[no-untyped-def]
    response = {
        "retrievalResults": [
            {
                "content": {"text": "This is the first result."},
                "location": "location1",
                "score": 0.9,
            },
            {
                "content": {"text": "This is the second result."},
                "location": "location2",
                "score": 0.8,
            },
            {"content": {"text": "This is the third result."}, "location": "location3"},
            {
                "content": {"text": "This is the fourth result."},
                "metadata": {"key1": "value1", "key2": "value2"},
            },
        ]
    }
    mock_client.retrieve.return_value = response

    query = "test query"

    expected_documents = [
        Document(
            page_content="This is the first result.",
            metadata={"location": "location1", "score": 0.9, "type": "TEXT"},
        ),
        Document(
            page_content="This is the second result.",
            metadata={"location": "location2", "score": 0.8, "type": "TEXT"},
        ),
        Document(
            page_content="This is the third result.",
            metadata={"location": "location3", "score": 0.0, "type": "TEXT"},
        ),
        Document(
            page_content="This is the fourth result.",
            metadata={
                "type": "TEXT",
                "score": 0.0,
                "source_metadata": {
                    "key1": "value1",
                    "key2": "value2",
                },
            },
        ),
    ]

    documents = retriever.invoke(query)

    assert documents == expected_documents

    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="test-knowledge-base",
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )


def test_get_relevant_documents_with_score(retriever, mock_client) -> None:  # type: ignore[no-untyped-def]
    response = {
        "retrievalResults": [
            {
                "content": {"text": "This is the first result."},
                "location": "location1",
                "score": 0.9,
            },
            {
                "content": {"text": "This is the second result."},
                "location": "location2",
                "score": 0.8,
            },
            {"content": {"text": "This is the third result."}, "location": "location3"},
            {
                "content": {"text": "This is the fourth result."},
                "metadata": {"key1": "value1", "key2": "value2"},
            },
        ]
    }
    mock_client.retrieve.return_value = response

    query = "test query"

    expected_documents = [
        Document(
            page_content="This is the first result.",
            metadata={"location": "location1", "score": 0.9, "type": "TEXT"},
        ),
        Document(
            page_content="This is the second result.",
            metadata={"location": "location2", "score": 0.8, "type": "TEXT"},
        ),
    ]

    retriever.min_score_confidence = 0.80
    documents = retriever.invoke(query)

    assert documents == expected_documents
