import asyncio
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws.utils import create_aws_client

logger = logging.getLogger(__name__)


class BedrockEmbeddings(BaseModel, Embeddings):
    """Bedrock embedding models.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Bedrock service.
    """

    """
    Example:
        .. code-block:: python

            from langchain_community.bedrock_embeddings import BedrockEmbeddings
            region_name ="us-east-1"
            credentials_profile_name = "default"
            model_id = "amazon.titan-embed-text-v1"

            be = BedrockEmbeddings(
                credentials_profile_name=credentials_profile_name,
                region_name=region_name,
                model_id=model_id
            )
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    """Bedrock client."""
    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Falls back to AWS_REGION/AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

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

    If provided, aws_access_key_id and aws_secret_access_key must also be provided.
    Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    model_id: str = "amazon.titan-embed-text-v1"
    """Id of the model to call, e.g., amazon.titan-embed-text-v1, this is
    equivalent to the modelId property in the list-foundation-models api"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    provider: Optional[str] = None
    """Name of the provider, e.g., amazon, cohere, etc..
    If not specified, the provider will be inferred from the model_id."""

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to us-east-1 endpoint"""

    normalize: bool = False
    """Whether the embeddings should be normalized to unit vectors"""

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def _inferred_provider(self) -> str:
        """Inferred provider of the model."""
        return self.provider or self.model_id.split(".")[0]

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that AWS credentials to and python package exists in environment."""
        if self.client is None:
            self.client = create_aws_client(
                region_name=self.region_name,
                credentials_profile_name=self.credentials_profile_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                endpoint_url=self.endpoint_url,
                config=self.config,
                service_name="bedrock-runtime",
            )

        return self

    def _embedding_func(self, text: str, input_type: str = "search_document") -> List[float]:
        """Call out to Bedrock embedding endpoint with a single text."""
        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")

        if self._inferred_provider == "cohere":
            # Cohere input_type depends on usage
            # for embedding documents use "search_document"
            # for embedding queries for retrieval use "search_query"
            response_body = self._invoke_model(
                input_body={
                    "input_type": input_type,
                    "texts": [text],
                }
            )
            return response_body.get("embeddings")[0]
        else:
            # includes common provider == "amazon"
            response_body = self._invoke_model(
                input_body={"inputText": text},
            )
            return response_body.get("embedding")

    def _cohere_multi_embedding(self, texts: List[str]) -> List[float]:
        """Call out to Cohere Bedrock embedding endpoint with multiple inputs."""
        # replace newlines, which can negatively affect performance.
        texts = [text.replace(os.linesep, " ") for text in texts]
        results = []

        # Iterate through the list of strings in batches
        for text_batch in _batch_cohere_embedding_texts(texts):
            batch_embeddings = self._invoke_model(
                input_body={
                    "input_type": "search_document",
                    "texts": text_batch,
                }
            ).get("embeddings")

            results += batch_embeddings

        return results

    def _invoke_model(self, input_body: Dict[str, Any] = {}) -> Dict[str, Any]:
        if self.model_kwargs:
            input_body = {**input_body, **self.model_kwargs}

        body = json.dumps(input_body)

        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())
            return response_body
        except Exception as e:
            logger.exception("Error raised by inference endpoint")
            raise e

    def _normalize_vector(self, embeddings: List[float]) -> List[float]:
        """Normalize the embedding to a unit vector."""
        emb = np.array(embeddings)
        norm_emb = emb / np.linalg.norm(emb)
        return norm_emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """

        # If we are able to make use of Cohere's multiple embeddings, use that
        if self._inferred_provider == "cohere":
            return self._embed_cohere_documents(texts)
        else:
            return self._iteratively_embed_documents(texts)

    def _embed_cohere_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._cohere_multi_embedding(texts)

        if self.normalize:
            response = [self._normalize_vector(embedding) for embedding in response]

        return response

    def _iteratively_embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = self._embedding_func(text)

            if self.normalize:
                response = self._normalize_vector(response)

            results.append(response)

        return results

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Bedrock model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        if self._inferred_provider == "cohere":
            embedding = self._embedding_func(text, input_type="search_query")
        else:
            embedding = self._embedding_func(text)

        if self.normalize:
            return self._normalize_vector(embedding)

        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous compute query embeddings using a Bedrock model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        return await run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """

        result = await asyncio.gather(*[self.aembed_query(text) for text in texts])

        return list(result)


def _batch_cohere_embedding_texts(texts: List[str]) -> Generator[List[str], None, None]:
    """Batches a set of texts into chunks that are acceptable for the Cohere embedding API:
    chunks of at most 96 items, or 2048 characters."""

    # Cohere embeddings want a maximum of 96 items and 2048 characters
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
    max_items = 96
    max_chars = 2048

    # Initialize batches
    current_batch = []
    current_chars = 0

    for text in texts:
        text_len = len(text)

        if text_len > max_chars:
            raise ValueError(
                "The Cohere embedding API does not support texts longer than 2048 characters."
            )

        # Check if adding the current string would exceed the limits
        if len(current_batch) >= max_items or current_chars + text_len > max_chars:
            # Process the current batch if limits are exceeded
            yield current_batch
            # Start a new batch
            current_batch = []
            current_chars = 0

        # Otherwise, add the string to the current batch
        current_batch.append(text)
        current_chars += text_len

    if current_batch:
        yield current_batch
