"""Integration tests for Bedrock KB retriever (managed + agentic).

These tests hit the live Bedrock API and require:
- AWS credentials configured
- A managed knowledge base with ID set in KNOWLEDGE_BASE_ID env var
- boto3 >= 1.43.2

Run with: pytest tests/integration_tests/retrievers/test_managed_kb.py -m "not compile"
"""

import os

import pytest

from langchain_aws.retrievers.bedrock import (
    AmazonKnowledgeBasesRetriever,
    agentic_retrieve,
)

KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
REGION = os.environ.get("AWS_REGION", "us-west-2")


@pytest.mark.skipif(
    not KNOWLEDGE_BASE_ID,
    reason="KNOWLEDGE_BASE_ID env var not set",
)
class TestManagedKBRetriever:
    """Integration tests against a live managed knowledge base."""

    def test_managed_search_returns_documents(self) -> None:
        """Verify managedSearchConfiguration returns documents from the API."""
        retriever = AmazonKnowledgeBasesRetriever(
            min_score_confidence=None,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={"managedSearchConfiguration": {"numberOfResults": 3}},
            region_name=REGION,
        )

        docs = retriever.invoke("What actions are supported?")

        assert len(docs) > 0
        assert docs[0].page_content != ""
        assert "score" in docs[0].metadata

    def test_managed_search_with_reranking_model_type(self) -> None:
        """Verify rerankingModelType='MANAGED' is accepted by the API."""
        retriever = AmazonKnowledgeBasesRetriever(
            min_score_confidence=None,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={
                "managedSearchConfiguration": {
                    "numberOfResults": 3,
                    "rerankingModelType": "MANAGED",
                }
            },
            region_name=REGION,
        )

        docs = retriever.invoke("What actions are supported?")

        assert len(docs) > 0

    def test_managed_search_number_of_results(self) -> None:
        """Verify numberOfResults controls result count."""
        retriever = AmazonKnowledgeBasesRetriever(
            min_score_confidence=None,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            retrieval_config={"managedSearchConfiguration": {"numberOfResults": 2}},
            region_name=REGION,
        )

        docs = retriever.invoke("What actions are supported?")

        assert len(docs) <= 2

    def test_agentic_retrieve_returns_results(self) -> None:
        """Verify agentic_retrieve() returns results from the API."""
        result = agentic_retrieve(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            query="What actions are supported?",
            region_name=REGION,
        )

        assert "results" in result
        assert len(result["results"]) > 0
        assert "content" in result["results"][0]

    def test_agentic_retrieve_with_generate_response(self) -> None:
        """Verify agentic_retrieve() with generateResponse returns a cited answer."""
        result = agentic_retrieve(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            query="What actions are supported?",
            region_name=REGION,
            generate_response=True,
        )

        assert "results" in result
        if "generatedResponse" in result:
            assert "answer" in result["generatedResponse"]
