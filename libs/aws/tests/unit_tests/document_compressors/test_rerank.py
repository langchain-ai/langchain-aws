from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_aws.document_compressors.rerank import BedrockRerank


@pytest.fixture
def reranker() -> BedrockRerank:
    reranker = BedrockRerank(
        model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        region_name="us-east-1",
        )
    reranker.client = MagicMock()
    return reranker

@patch("boto3.Session")
def test_initialize_client(mock_boto_session: MagicMock, reranker: BedrockRerank) -> None:
    session_instance = MagicMock()
    mock_boto_session.return_value = session_instance
    session_instance.client.return_value = MagicMock()
    assert reranker.client is not None

@patch("langchain_aws.document_compressors.rerank.BedrockRerank.rerank")
def test_rerank(mock_rerank: MagicMock, reranker: BedrockRerank) -> None:
    mock_rerank.return_value = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.8},
    ]
    
    documents = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]
    query = "Example Query"
    results = reranker.rerank(documents, query)

    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["relevance_score"] == 0.9
    assert results[1]["index"] == 1
    assert results[1]["relevance_score"] == 0.8

@patch("langchain_aws.document_compressors.rerank.BedrockRerank.rerank")
def test_compress_documents(mock_rerank: MagicMock, reranker: BedrockRerank) -> None:
    mock_rerank.return_value = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.85},
    ]
    
    documents = [Document(page_content="Content 1"), Document(page_content="Content 2")]
    query = "Relevant query"
    compressed_docs = reranker.compress_documents(documents, query)

    assert len(compressed_docs) == 2
    assert compressed_docs[0].metadata["relevance_score"] == 0.95
    assert compressed_docs[1].metadata["relevance_score"] == 0.85
