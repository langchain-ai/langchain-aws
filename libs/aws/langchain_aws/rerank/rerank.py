from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import boto3
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import from_env
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self


class BedrockRerank(BaseDocumentCompressor):
    """Document compressor that uses AWS Bedrock Rerank API."""

    client: Any = None
    """Bedrock client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model_id: Optional[str] = "amazon.rerank-v1:0"
    """Model ID to fetch ARN dynamically."""
    aws_region: str = Field(
        default_factory=from_env("AWS_DEFAULT_REGION", default=None)
    )
    """AWS region to initialize the Bedrock client."""
    aws_profile: Optional[str] = Field(
        default_factory=from_env("AWS_PROFILE", default=None)
    )
    """AWS profile for authentication, optional."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def initialize_client(self) -> Self:
        """Initialize the AWS Bedrock client."""
        if not self.client:
            session = self._get_session()
            self.client = session.client("bedrock-agent-runtime")
        return self

    def _get_session(self):
        return (
            boto3.Session(profile_name=self.aws_profile)
            if self.aws_profile
            else boto3.Session()
        )

    def _get_model_arn(self) -> str:
        """Fetch the ARN of the reranker model using the model ID."""
        session = self._get_session()
        client = session.client("bedrock", self.aws_region)
        response = client.get_foundation_model(modelIdentifier=self.model_id)
        return response["modelDetails"]["modelArn"]

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        top_n: Optional[int] = None,
        extra_model_fields: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents based on their relevance to the query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n: The number of top-ranked results to return. Defaults to self.top_n.
            extra_model_fields: A dictionary of additional fields to pass to the model.

        Returns:
            List[Dict[str, Any]]: A list of ranked documents with relevance scores.
        """
        if len(documents) == 0:
            return []

        model_arn = self._get_model_arn()

        # Serialize documents for the Bedrock API
        serialized_documents = [
            {"textDocument": {"text": doc.page_content}, "type": "TEXT"}
            if isinstance(doc, Document)
            else {"textDocument": {"text": doc}, "type": "TEXT"}
            if isinstance(doc, str)
            else {"jsonDocument": doc, "type": "JSON"}
            for doc in documents
        ]

        request_body = {
            "queries": [{"textQuery": {"text": query}, "type": "TEXT"}],
            "rerankingConfiguration": {
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": model_arn,
                        "additionalModelRequestFields": extra_model_fields
                        or {},
                    },
                    "numberOfResults": top_n or self.top_n,
                },
                "type": "BEDROCK_RERANKING_MODEL",
            },
            "sources": [
                {"inlineDocumentSource": doc, "type": "INLINE"}
                for doc in serialized_documents
            ],
        }

        response = self.client.rerank(**request_body)
        response_body = response.get("results", [])

        results = [
            {"index": result["index"], "relevance_score": result["relevanceScore"]}
            for result in response_body
        ]

        return results

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Bedrock's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
