import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_aws import BedrockRerank


# Mock setup
@pytest.fixture
def mock_bedrock_client():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "body": MagicMock(
            read=MagicMock(
                return_value=json.dumps(
                    {
                        "results": [
                            {"index": 0, "relevance_score": 0.95},
                            {"index": 1, "relevance_score": 0.90},
                        ]
                    }
                )
            )
        )
    }
    return mock_client


@pytest.fixture
def bedrock_rerank(mock_bedrock_client):
    return BedrockRerank(client=mock_bedrock_client)


# Test initialize_client
def test_initialize_client_with_profile():
    bedrock_rerank = BedrockRerank(aws_profile="default")
    bedrock_rerank.initialize_client()
    assert bedrock_rerank.client is not None


def test_initialize_client_without_profile():
    bedrock_rerank = BedrockRerank()
    bedrock_rerank.initialize_client()
    assert bedrock_rerank.client is not None


# Test rerank method
def test_rerank_success(bedrock_rerank):
    documents = ["doc1", "doc2", "doc3"]
    query = "Test query"
    results = bedrock_rerank.rerank(documents, query)
    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["relevance_score"] == 0.95


def test_rerank_empty_documents(bedrock_rerank):
    results = bedrock_rerank.rerank([], "query")
    assert results == []


# Test compress_documents method
def test_compress_documents(bedrock_rerank):
    documents = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]
    query = "Test query"
    compressed = bedrock_rerank.compress_documents(documents, query)
    assert len(compressed) == 2
    assert compressed[0].metadata["relevance_score"] == 0.95
    assert compressed[1].metadata["relevance_score"] == 0.90
