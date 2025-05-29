from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_aws.utils import create_aws_client


class BedrockRerank(BaseDocumentCompressor):
    """Document compressor that uses AWS Bedrock Rerank API."""

    model_arn: str
    """The ARN of the reranker model."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    """Bedrock client to use for compressing documents."""

    top_n: Optional[int] = 3
    """Number of documents to return."""

    region_name: Optional[str] = None
    """The aws region, e.g., `us-west-2`. 

    Falls back to AWS_REGION or AWS_DEFAULT_REGION env variable or region specified in 
    ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(
        default_factory=from_env("AWS_PROFILE", default=None)
    )
    """AWS profile for authentication, optional."""

    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id. 

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.
    """

    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key. 

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token. 

    If provided, aws_access_key_id and aws_secret_access_key must 
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    endpoint_url: Optional[str] = Field(default=None, alias="base_url")
    """Needed if you don't want to default to us-east-1 endpoint"""

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def initialize_client(cls, values: Dict[str, Any]) -> Any:
        """Initialize the AWS Bedrock client."""
        if not values.get("client"):
            values["client"] = create_aws_client(
                region_name=values.get("region_name"),
                credentials_profile_name=values.get("credentials_profile_name"),
                aws_access_key_id=values.get("aws_access_key_id"),
                aws_secret_access_key=values.get("aws_secret_access_key"),
                aws_session_token=values.get("aws_session_token"),
                endpoint_url=values.get("endpoint_url"),
                config=values.get("config"),
                service_name="bedrock-agent-runtime",
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
            additional_model_request_fields: Additional fields to pass to the model.

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
            doc_copy = Document(**doc.model_dump())
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
