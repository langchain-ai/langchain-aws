import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


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
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    model_id: str = "amazon.titan-embed-text-v1"
    """Id of the model to call, e.g., amazon.titan-embed-text-v1, this is
    equivalent to the modelId property in the list-foundation-models api"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

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

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that AWS credentials to and python package exists in environment."""

        if self.client is not None:
            return self

        try:
            import boto3

            if self.credentials_profile_name is not None:
                session = boto3.Session(profile_name=self.credentials_profile_name)
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if self.region_name:
                client_params["region_name"] = self.region_name

            if self.endpoint_url:
                client_params["endpoint_url"] = self.endpoint_url

            if self.config:
                client_params["config"] = self.config

            self.client = session.client("bedrock-runtime", **client_params)

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                f"profile name are valid. Bedrock error: {e}"
            ) from e

        return self

    def _embedding_func(self, text: str) -> List[float]:
        """Call out to Bedrock embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")

        # format input body for provider
        provider = self.model_id.split(".")[0]
        input_body: Dict[str, Any] = {}
        if provider == "cohere":
            input_body["input_type"] = "search_document"
            input_body["texts"] = [text]
        else:
            # includes common provider == "amazon"
            input_body["inputText"] = text

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
            if provider == "cohere":
                return response_body.get("embeddings")[0]
            else:
                return response_body.get("embedding")

        except Exception as e:
            logging.error(f"Error raised by inference endpoint: {e}")
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
