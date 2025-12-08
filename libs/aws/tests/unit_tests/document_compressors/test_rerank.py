from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_aws.document_compressors.rerank import BedrockRerank


@pytest.fixture
@patch("langchain_aws.utils.create_aws_client")
def reranker(mock_create_client: MagicMock) -> BedrockRerank:
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    reranker = BedrockRerank(
        model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        region_name="us-east-1",
    )
    return reranker


@patch("boto3.Session")
def test_initialize_client(
    mock_boto_session: MagicMock, reranker: BedrockRerank
) -> None:
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

    documents = [
        Document(page_content="Content 1", id="doc1"),
        Document(page_content="Content 2", id="doc2"),
    ]
    query = "Relevant query"
    compressed_docs = reranker.compress_documents(documents, query)

    assert compressed_docs[0].id == "doc1"
    assert compressed_docs[0].page_content == "Content 1"
    assert compressed_docs[1].id == "doc2"
    assert compressed_docs[1].page_content == "Content 2"

    assert len(compressed_docs) == 2
    assert compressed_docs[0].metadata["relevance_score"] == 0.95
    assert compressed_docs[1].metadata["relevance_score"] == 0.85


@patch("langchain_aws.utils.create_aws_client")
def test_rerank_clamps_top_n_to_document_count(
    mock_create_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    captured_request: dict = {}

    def mock_rerank(**kwargs: dict) -> dict:
        captured_request.update(kwargs)
        num_results = kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"][
            "numberOfResults"
        ]
        return {
            "results": [
                {"index": i, "relevanceScore": 0.9 - (i * 0.1)}
                for i in range(num_results)
            ]
        }

    mock_client.rerank = mock_rerank

    reranker = BedrockRerank(
        model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        region_name="us-east-1",
        top_n=10,
        client=mock_client,
    )

    documents = [
        Document(page_content="Doc 1"),
        Document(page_content="Doc 2"),
        Document(page_content="Doc 3"),
    ]

    results = reranker.rerank(documents, query="test query")

    actual_num_results = captured_request["rerankingConfiguration"][
        "bedrockRerankingConfiguration"
    ]["numberOfResults"]
    assert actual_num_results == 3
    assert len(results) == 3


@patch("langchain_aws.utils.create_aws_client")
def test_rerank_top_n_override_also_clamped(
    mock_create_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    captured_request: dict = {}

    def mock_rerank(**kwargs: dict) -> dict:
        captured_request.update(kwargs)
        num_results = kwargs["rerankingConfiguration"]["bedrockRerankingConfiguration"][
            "numberOfResults"
        ]
        return {
            "results": [
                {"index": i, "relevanceScore": 0.9 - (i * 0.1)}
                for i in range(num_results)
            ]
        }

    mock_client.rerank = mock_rerank

    reranker = BedrockRerank(
        model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        region_name="us-east-1",
        top_n=2,
        client=mock_client,
    )

    documents = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]

    results = reranker.rerank(documents, query="test query", top_n=100)

    actual_num_results = captured_request["rerankingConfiguration"][
        "bedrockRerankingConfiguration"
    ]["numberOfResults"]
    assert actual_num_results == 2
    assert len(results) == 2
