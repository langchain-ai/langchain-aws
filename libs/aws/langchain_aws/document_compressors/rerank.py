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

    model_arn: str
    """The ARN of the reranker model."""
    client: Any = None
    """Bedrock client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    region_name: str = Field(
        default_factory=from_env("AWS_DEFAULT_REGION", default=None)
    )
    """AWS region to initialize the Bedrock client."""
    credentials_profile_name: Optional[str] = Field(
        default_factory=from_env("AWS_PROFILE", default=None)
    )
    """AWS profile for authentication, optional."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def initialize_client(cls, values: Dict[str, Any]) -> Any:
        """Initialize the AWS Bedrock client."""
        if not values.get("client"):
            session = (
                boto3.Session(profile_name=values.get("credentials_profile_name"))
                if values.get("credentials_profile_name", None)
                else boto3.Session()
            )
            values["client"] = session.client(
                "bedrock-agent-runtime",
                region_name=values.get("region_name"),
            )
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        top_n: Optional[int] = None,
        additional_model_request_fields: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents based on their relevance to the query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n: The number of top-ranked results to return. Defaults to self.top_n.
            additional_model_request_fields: A dictionary of additional fields to pass to the model.

        Returns:
            List[Dict[str, Any]]: A list of ranked documents with relevance scores.
        """
        if len(documents) == 0:
            return []

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
                        "modelArn": self.model_arn,
                        "additionalModelRequestFields": additional_model_request_fields
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
