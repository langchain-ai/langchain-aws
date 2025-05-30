import json
from typing import Any, Dict, List, Literal, Optional, Union

from botocore.client import Config
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Annotated

from langchain_aws.utils import create_aws_client

FilterValue = Union[Dict[str, Any], List[Any], int, float, str, bool, None]
Filter = Dict[str, FilterValue]


class SearchFilter(BaseModel):
    """Filter configuration for retrieval."""

    andAll: Optional[List["SearchFilter"]] = None
    orAll: Optional[List["SearchFilter"]] = None
    equals: Optional[Filter] = None
    greaterThan: Optional[Filter] = None
    greaterThanOrEquals: Optional[Filter] = None
    in_: Optional[Filter] = Field(None, alias="in")
    lessThan: Optional[Filter] = None
    lessThanOrEquals: Optional[Filter] = None
    listContains: Optional[Filter] = None
    notEquals: Optional[Filter] = None
    notIn: Optional[Filter] = Field(None, alias="notIn")
    startsWith: Optional[Filter] = None
    stringContains: Optional[Filter] = None

    model_config = ConfigDict(
        populate_by_name=True,
    )


class VectorSearchConfig(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Configuration for vector search."""

    numberOfResults: int = 4
    filter: Optional[SearchFilter] = None
    overrideSearchType: Optional[Literal["HYBRID", "SEMANTIC"]] = None


class RetrievalConfig(BaseModel, extra="allow"):  # type: ignore[call-arg]
    """Configuration for retrieval."""

    vectorSearchConfiguration: VectorSearchConfig
    nextToken: Optional[str] = None


class AmazonKnowledgeBasesRetriever(BaseRetriever):
    """`Amazon Bedrock Knowledge Bases` retrieval.

        See https://aws.amazon.com/bedrock/knowledge-bases for more info.

        Args:
            knowledge_base_id: Knowledge Base ID.

            region_name: The aws region e.g., `us-west-2`.
                Fallback to AWS_REGION/AWS_DEFAULT_REGION env variable or region specified in
                ~/.aws/config.

            credentials_profile_name: The name of the profile in the ~/.aws/credentials
                or ~/.aws/config files, which has either access keys or role information
                specified. If not specified, the default credential profile or, if on an
                EC2 instance, credentials from IMDS will be used.

            aws_access_key_id: AWS access key id. If provided, aws_secret_access_key must
                also be provided. If not specified, the default credential profile or, if
                on an EC2 instance, credentials from IMDS will be used. See:
                https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.

            aws_secret_access_key: AWS secret_access_key. If provided, aws_access_key_id
                must also be provided. If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used. See:
                https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.

            aws_session_token: AWS session token. If provided, aws_access_key_id and
                aws_secret_access_key must also be provided. Not required unless using temporary
                credentials. See:
                https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.

            endpoint_url: Needed if you don't want to default to us-east-1 endpoint.

            config: An optional botocore.config.Config instance to pass to the client.

            client: boto3 client for bedrock agent runtime.

            guardrail_config: Configuration information for a guardrail that you want
                to use in the request.

            retrieval_config: Optional configuration for retrieval specified as a
                Python object (RetrievalConfig) or as a dictionary

        Example:
            .. code-block:: python
                from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
                retriever = AmazonKnowledgeBasesRetriever(
                    knowledge_base_id="<knowledge-base-id>",
                    retrieval_config={
                        "vectorSearchConfiguration": {
                            "numberOfResults": 4
                        }
                    },
                )
    """

    knowledge_base_id: str
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    endpoint_url: Optional[str] = None
    config: Any = None
    client: Any = None
    guardrail_config: Optional[Dict[str, Any]] = Field(
        default=None, alias="guardrails"
    )
    retrieval_config: Optional[Union[RetrievalConfig, Dict[str, Any]]] = None
    min_score_confidence: Annotated[
        Optional[float], Field(ge=0.0, le=1.0, default=None)
    ]

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: Dict[str, Any]) -> Any:
        if "guardrail_config" in values and "guardrails" not in values:
            values["guardrails"] = values.pop("guardrail_config")
        if values.get("client") is None:
            values["client"] = create_aws_client(
                region_name=values.get("region_name"),
                credentials_profile_name=values.get("credentials_profile_name"),
                aws_access_key_id=values.get("aws_access_key_id"),
                aws_secret_access_key=values.get("aws_secret_access_key"),
                aws_session_token=values.get("aws_session_token"),
                endpoint_url=values.get("endpoint_url"),
                config=values.get("config") or Config(
                    connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
                ),
                service_name="bedrock-agent-runtime",
            )

        return values

    def _filter_by_score_confidence(self, docs: List[Document]) -> List[Document]:
        """
        Filter out the records that have a score confidence
        less than the required threshold.
        """
        if not self.min_score_confidence:
            return docs
        filtered_docs = [
            item
            for item in docs
            if (
                item.metadata.get("score") is not None
                and item.metadata.get("score", 0.0) >= self.min_score_confidence
            )
        ]
        return filtered_docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant document from a KnowledgeBase

        :param query: the user's query
        :param run_manager: The callback handler to use
        :return: List of relevant documents
        """
        retrieve_request: Dict[str, Any] = self._get_retrieve_request(query)
        response = self.client.retrieve(**retrieve_request)
        results = response["retrievalResults"]
        documents: List[
            Document
        ] = AmazonKnowledgeBasesRetriever._retrieval_results_to_documents(results)

        return self._filter_by_score_confidence(docs=documents)

    def _get_retrieve_request(self, query: str) -> Dict[str, Any]:
        """
        Build a Retrieve request

        :param query:
        :return:
        """
        request: Dict[str, Any] = {
            "retrievalQuery": {"text": query.strip()},
            "knowledgeBaseId": self.knowledge_base_id,
        }
        if self.guardrail_config:
            if not (self.guardrail_config.get("guardrailId")
                    and self.guardrail_config.get("guardrailVersion")):
                raise TypeError(
                    "Guardrail configuration must be a dictionary with both 'guardrailId' "
                    "and 'guardrailVersion' keys."
                )
            request["guardrailConfiguration"] = self.guardrail_config
        if self.retrieval_config:
            if isinstance(self.retrieval_config, dict):
                request["retrievalConfiguration"] = self.retrieval_config
            else:
                request["retrievalConfiguration"] = self.retrieval_config.model_dump(
                    exclude_none=True, by_alias=True
                )
        return request

    @staticmethod
    def _retrieval_results_to_documents(
        results: List[Dict[str, Any]],
    ) -> List[Document]:
        """
        Convert the Retrieve API results to LangChain Documents

        :param results:  Retrieve API results list
        :return: List of LangChain Documents
        """
        documents = []
        for result in results:
            content = AmazonKnowledgeBasesRetriever._get_content_from_result(result)
            result["type"] = result.get("content", {}).get("type", "TEXT")
            result.pop("content")
            if "score" not in result:
                result["score"] = 0
            if "metadata" in result:
                result["source_metadata"] = result.pop("metadata")
            documents.append(
                Document(
                    page_content=content,
                    metadata=result,
                )
            )
        return documents

    @staticmethod
    def _get_content_from_result(result: Dict[str, Any]) -> Optional[str]:
        """
        Convert the content from one Retrieve API result to string

        :param result: Retrieve API search result
        :return: string representation of the content attribute
        """
        if not result:
            raise ValueError("Invalid search result")
        content: dict = result.get("content")
        if not content:
            raise ValueError(
                "Invalid search result, content is missing from the result"
            )
        if not content.get("type"):
            return content.get("text")
        if content["type"] == "TEXT":
            return content.get("text")
        elif content["type"] == "IMAGE":
            return content.get("byteContent")
        elif content["type"] == "ROW":
            row: Optional[List[dict]] = content.get("row", [])
            return json.dumps(row if row else [])
        else:
            # future proofing this class to prevent code breaks if new types
            # are introduced
            return None
