import json
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
    model: Optional[str] = "amazon.rerank-v1:0"
    """Model to use for reranking. Default is amazon.rerank-v1:0."""
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
            session = (
                boto3.Session(profile_name=self.aws_profile)
                if self.aws_profile
                else boto3.Session()
            )
            self.client = session.client("bedrock-runtime", region_name=self.aws_region)
        return self

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents based on their relevance to the query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n: The number of top-ranked results to return. Defaults to self.top_n.
            model: The model to use for reranking. Defaults to self.model.

        Returns:
            List[Dict[str, Any]]: A list of ranked documents with relevance scores.
        """
        if len(documents) == 0:
            return []

        # Serialize documents for the Bedrock API
        serialized_documents = [
            json.dumps(doc)
            if isinstance(doc, dict)
            else doc.page_content
            if isinstance(doc, Document)
            else doc
            for doc in documents
        ]

        body = json.dumps(
            {
                "query": query,
                "documents": serialized_documents,
                "top_n": top_n or self.top_n,
            }
        )

        response = self.client.invoke_model(
            modelId=model or self.model,
            accept="application/json",
            contentType="application/json",
            body=body,
        )

        response_body = json.loads(response.get("body").read())
        results = [
            {"index": result["index"], "relevance_score": result["relevance_score"]}
            for result in response_body["results"]
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
