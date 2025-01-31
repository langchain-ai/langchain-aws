from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from langchain_aws import BedrockRerank


@pytest.fixture
def reranker():
    reranker = BedrockRerank()
    reranker.client = MagicMock()
    return reranker


@patch("langchain_aws.rerank.rerank.boto3.Session")
def test_initialize_client(mock_boto_session, reranker):
    session_instance = MagicMock()
    mock_boto_session.return_value = session_instance
    session_instance.client.return_value = MagicMock()
    reranker.initialize_client()
    assert reranker.client is not None


@patch("langchain_aws.rerank.rerank.BedrockRerank._get_model_arn")
def test_rerank(mock_get_model_arn, reranker):
    mock_get_model_arn.return_value = "arn:aws:bedrock:model/amazon.rerank-v1:0"
    reranker.client.rerank.return_value = {
        "results": [
            {"index": 0, "relevanceScore": 0.9},
            {"index": 1, "relevanceScore": 0.8},
        ]
    }

    documents = [Document("Doc 1"), Document("Doc 2")]
    query = "Example Query"
    results = reranker.rerank(documents, query)

    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["relevance_score"] == 0.9
    assert results[1]["index"] == 1
    assert results[1]["relevance_score"] == 0.8


@patch("langchain_aws.rerank.rerank.BedrockRerank.rerank")
def test_compress_documents(mock_rerank, reranker):
    mock_rerank.return_value = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.85},
    ]
    documents = [Document("Content 1"), Document("Content 2")]
    query = "Relevant query"
    compressed_docs = reranker.compress_documents(documents, query)

    assert len(compressed_docs) == 2
    assert compressed_docs[0].metadata["relevance_score"] == 0.95
    assert compressed_docs[1].metadata["relevance_score"] == 0.85


@patch("langchain_aws.rerank.rerank.BedrockRerank._get_model_arn")
def test_get_model_arn(mock_get_model_arn, reranker):
    mock_get_model_arn.return_value = "arn:aws:bedrock:model/amazon.rerank-v1:0"
    model_arn = reranker._get_model_arn()
    assert model_arn == "arn:aws:bedrock:model/amazon.rerank-v1:0"
