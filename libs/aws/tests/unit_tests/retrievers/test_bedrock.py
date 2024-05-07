from unittest.mock import MagicMock

import pytest
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_retriever_config():
    return {"vectorSearchConfiguration": {"numberOfResults": 4}}


@pytest.fixture
def amazon_retriever(mock_client, mock_retriever_config):
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test_kb_id",
        retrieval_config=mock_retriever_config,
        client=mock_client,
    )


def test_retriever_invoke(amazon_retriever, mock_client):
    query = "test query"
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {"content": {"text": "result1"}, "metadata": {"key": "value1"}},
            {
                "content": {"text": "result2"},
                "metadata": {"key": "value2"},
                "score": 1,
                "location": "testLocation",
            },
            {"content": {"text": "result3"}},
        ]
    }
    documents = amazon_retriever.invoke(query, run_manager=None)

    assert len(documents) == 3
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result1"
    assert documents[0].metadata == {"score": 0, "source_metadata": {"key": "value1"}}
    assert documents[1].page_content == "result2"
    assert documents[1].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
    }
    assert documents[2].page_content == "result3"
    assert documents[2].metadata == {"score": 0}
