import asyncio
import base64
import json
import logging
import os
import re
from typing import Any, Dict, Generator, List, Literal, Optional, Union

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws.utils import create_aws_client

logger = logging.getLogger(__name__)

# Type alias for media input (base64 string, file path, S3 URI, or raw bytes)
MediaInput = Union[str, bytes]


class BedrockEmbeddings(BaseModel, Embeddings):
    """Bedrock embedding models.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Bedrock service.

    Example:
        ```python
        from langchain_aws import BedrockEmbeddings

        embeddings = BedrockEmbeddings(
            credentials_profile_name="default",
            region_name="us-east-1",
            model_id="amazon.nova-2-multimodal-embeddings-v1:0",
            dimensions=256,
        )
        ```
    """

    client: Any = Field(default=None, exclude=True)
    """Bedrock client."""

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`.

    Falls back to `AWS_REGION`/`AWS_DEFAULT_REGION` env variable or region
    specified in `~/.aws/config` in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the `~/.aws/credentials` or `~/.aws/config` files,
    which has either access keys or role information specified.
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

    If not provided, will be read from `AWS_ACCESS_KEY_ID` environment variable.
    """

    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key.

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.

    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from `AWS_SECRET_ACCESS_KEY` environment variable.

    """

    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token.

    If provided, `aws_access_key_id` and `aws_secret_access_key` must also be
    provided.

    Not required unless using temporary credentials.

    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from `AWS_SESSION_TOKEN` environment variable.

    """

    model_id: str = "amazon.titan-embed-text-v1"
    """Id of the model to call, e.g., `'amazon.titan-embed-text-v1'`, this is
    equivalent to the `modelId` property in the list-foundation-models api

    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    provider: Optional[str] = None
    """Name of the provider, e.g., amazon, cohere, etc..
    If not specified, the provider will be inferred from the ``model_id``.

    """

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to `'us-east-1'` endpoint"""

    normalize: bool = False
    """Whether the embeddings should be normalized to unit vectors"""

    dimensions: Optional[int] = None
    """The number of dimensions for the output embeddings.

    Only supported by certain models (Titan v2, Cohere, Nova).
    If not specified, uses the model's default.
    """

    config: Any = None
    """An optional `botocore.config.Config` instance to pass to the client."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def _inferred_provider(self) -> str:
        """Inferred provider of the model."""
        if self.provider:
            return self.provider

        regions = ("eu", "us", "us-gov", "apac", "sa", "amer", "global", "jp", "au")
        parts = self.model_id.split(".")
        return parts[1] if parts[0] in regions else parts[0]

    @property
    def _is_cohere_v4(self) -> bool:
        """Check if the model is Cohere Embed v4."""
        return "cohere.embed-v4" in self.model_id

    @property
    def _is_nova_embed(self) -> bool:
        """Check if the model is Amazon Nova Embed."""
        return (
            self._inferred_provider == "amazon"
            and "nova" in self.model_id
            and "embed" in self.model_id
        )

    @property
    def _is_titan_multimodal(self) -> bool:
        """Check if the model is Titan Embed Image (multimodal)."""
        return "titan-embed-image" in self.model_id

    @property
    def _is_marengo(self) -> bool:
        """Check if the model is TwelveLabs Marengo."""
        return "marengo" in self.model_id.lower()

    def _supports_image(self) -> bool:
        """Check if the model supports image embeddings."""
        return (
            self._is_titan_multimodal
            or self._is_nova_embed
            or self._inferred_provider == "cohere"
            or self._is_marengo
        )

    def _supports_audio(self) -> bool:
        """Check if the model supports audio embeddings (sync API)."""
        return self._is_nova_embed

    def _supports_video(self) -> bool:
        """Check if the model supports video embeddings (sync API)."""
        return self._is_nova_embed

    def _supports_s3_input(self) -> bool:
        """Check if the model supports S3 URI input."""
        return self._is_nova_embed or self._is_marengo

    def _get_dimensions_params(self) -> Dict[str, Any]:
        """Get dimensions parameter with provider-specific key name."""
        if self.dimensions is None:
            return {}

        if self._is_cohere_v4:
            return {"output_dimension": self.dimensions}
        elif self._inferred_provider == "cohere":
            return {}
        else:
            return {"dimensions": self.dimensions}

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

    def _embedding_func(
        self, text: str, input_type: str = "search_document"
    ) -> List[float]:
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
                    **self._get_dimensions_params(),
                }
            )
            embeddings = response_body.get("embeddings")
            if embeddings is None:
                raise ValueError("No embeddings returned from model")
            # Embed v3 and v4 schemas
            if isinstance(embeddings, dict) and "float" in embeddings:
                processed_embeddings = embeddings["float"]
            else:
                processed_embeddings = embeddings
            return processed_embeddings[0]
        elif self._is_nova_embed:
            single_embedding_params: Dict[str, Any] = {
                "embeddingPurpose": "GENERIC_INDEX",
                "text": {
                    "truncationMode": "END",
                    "value": text,
                },
            }
            if self.dimensions:
                single_embedding_params["embeddingDimension"] = self.dimensions
            response_body = self._invoke_model(
                input_body={
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": single_embedding_params,
                }
            )
            embeddings = response_body.get("embeddings")
            if not embeddings or not embeddings[0].get("embedding"):
                raise ValueError("No embedding returned from model")
            return embeddings[0]["embedding"]
        else:
            # includes common provider == "amazon"
            response_body = self._invoke_model(
                input_body={"inputText": text, **self._get_dimensions_params()},
            )
            embedding = response_body.get("embedding")
            if embedding is None:
                raise ValueError("No embedding returned from model")
            return embedding

    def _cohere_multi_embedding(self, texts: List[str]) -> List[List[float]]:
        """Call out to Cohere Bedrock embedding endpoint with multiple inputs."""
        # replace newlines, which can negatively affect performance.
        texts = [text.replace(os.linesep, " ") for text in texts]
        results: List[List[float]] = []

        # Iterate through the list of strings in batches
        for text_batch in _batch_cohere_embedding_texts(
            texts, is_v4=self._is_cohere_v4
        ):
            batch_embeddings = self._invoke_model(
                input_body={
                    "input_type": "search_document",
                    "texts": text_batch,
                    **self._get_dimensions_params(),
                }
            ).get("embeddings")
            # Embed v3 and v4 schemas
            if isinstance(batch_embeddings, dict) and "float" in batch_embeddings:
                processed_embeddings = batch_embeddings["float"]
            else:
                processed_embeddings = batch_embeddings

            if processed_embeddings is not None:
                results += processed_embeddings

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

            response_metadata = response.get("ResponseMetadata", {})
            if response_metadata:
                logger.info(
                    f"Successfully invoked model {self.model_id}. "
                    f"ResponseMetadata: {response_metadata}"
                )

            response_body = json.loads(response.get("body").read())
            return response_body
        except Exception as e:
            logger.exception("Error raised by inference endpoint")
            raise e

    def _detect_media_format(self, data: bytes) -> str:
        """Detect media format from magic bytes."""
        if len(data) < 12:
            logger.warning(
                "Media data too short for format detection (%d bytes), "
                "defaulting to JPEG",
                len(data),
            )
            return "jpeg"

        # JPEG: FF D8 FF
        if data[:3] == b"\xff\xd8\xff":
            return "jpeg"
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        # GIF: GIF87a or GIF89a
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        # WEBP: RIFF....WEBP
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        # MP4/MOV: ....ftyp
        if data[4:8] == b"ftyp":
            return "mp4"
        # MP3: FF FB or ID3
        if data[:2] == b"\xff\xfb" or data[:3] == b"ID3":
            return "mp3"
        # WAV: RIFF....WAVE
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        # OGG: OggS
        if data[:4] == b"OggS":
            return "ogg"

        logger.warning(
            "Could not detect media format from magic bytes, defaulting to JPEG"
        )
        return "jpeg"

    def _load_media(
        self,
        data: MediaInput,
        media_type: Literal["image", "audio", "video"],
    ) -> tuple[str, str, Literal["inline", "s3"]]:
        """Normalize media input to (payload, format, source_kind).

        Args:
            data: Base64 string, file path, S3 URI, or raw bytes.
            media_type: Type of media (image, audio, video).

        Returns:
            Tuple of (payload, format, source_kind) where:
            - payload: base64 string or S3 URI
            - format: detected format (jpeg, png, mp3, etc.)
            - source_kind: "inline" or "s3"
        """
        if isinstance(data, bytes):
            fmt = self._detect_media_format(data)
            b64 = base64.b64encode(data).decode("utf-8")
            return b64, fmt, "inline"

        if data.startswith("s3://"):
            if not self._supports_s3_input():
                msg = f"S3 URIs not supported for model {self.model_id}"
                raise ValueError(msg)
            ext_match = re.search(r"\.([a-zA-Z0-9]+)$", data)
            fmt = ext_match.group(1).lower() if ext_match else "jpeg"
            if fmt == "jpg":
                fmt = "jpeg"
            return data, fmt, "s3"

        data_uri_match = re.match(
            r"^data:(?:image|video|audio)/([a-zA-Z0-9]+);base64,(.+)$", data
        )
        if data_uri_match:
            fmt = data_uri_match.group(1).lower()
            b64 = data_uri_match.group(2)
            return b64, fmt, "inline"

        if os.path.isfile(data):
            with open(data, "rb") as f:
                file_bytes = f.read()
            fmt = self._detect_media_format(file_bytes)
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            return b64, fmt, "inline"

        # Assume raw base64 string
        try:
            decoded = base64.b64decode(data)
            fmt = self._detect_media_format(decoded)
            return data, fmt, "inline"
        except Exception:
            msg = "Could not interpret input as base64, file path, or S3 URI"
            raise ValueError(msg)

    def _build_image_request(
        self, payload: str, fmt: str, source_kind: str
    ) -> Dict[str, Any]:
        """Build provider-specific request body for image embedding."""
        if self._is_titan_multimodal:
            body: Dict[str, Any] = {"inputImage": payload}
            if self.dimensions:
                body["embeddingConfig"] = {"outputEmbeddingLength": self.dimensions}
            return body

        if self._is_nova_embed:
            params: Dict[str, Any] = {"embeddingPurpose": "GENERIC_INDEX"}
            if source_kind == "s3":
                params["image"] = {
                    "format": fmt,
                    "source": {"s3Location": {"uri": payload}},
                }
            else:
                params["image"] = {"format": fmt, "source": {"bytes": payload}}
            if self.dimensions:
                params["embeddingDimension"] = self.dimensions
            return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

        if self._inferred_provider == "cohere":
            data_uri = f"data:image/{fmt};base64,{payload}"
            if self._is_cohere_v4:
                return {
                    "input_type": "search_document",
                    "images": [data_uri],
                    **self._get_dimensions_params(),
                }
            return {
                "input_type": "image",
                "images": [data_uri],
                "embedding_types": ["float"],
            }

        if self._is_marengo:
            media_source: Dict[str, Any]
            if source_kind == "s3":
                media_source = {"s3Location": {"uri": payload}}
            else:
                media_source = {"base64String": payload}
            return {"inputType": "image", "image": {"mediaSource": media_source}}

        msg = f"Image embeddings not supported for model {self.model_id}"
        raise ValueError(msg)

    def _build_audio_request(
        self, payload: str, fmt: str, source_kind: str
    ) -> Dict[str, Any]:
        """Build request body for audio embedding (Nova only)."""
        params: Dict[str, Any] = {"embeddingPurpose": "GENERIC_INDEX"}
        if source_kind == "s3":
            params["audio"] = {
                "format": fmt,
                "source": {"s3Location": {"uri": payload}},
            }
        else:
            params["audio"] = {"format": fmt, "source": {"bytes": payload}}
        if self.dimensions:
            params["embeddingDimension"] = self.dimensions
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _build_video_request(
        self, payload: str, fmt: str, source_kind: str
    ) -> Dict[str, Any]:
        """Build request body for video embedding (Nova only)."""
        params: Dict[str, Any] = {"embeddingPurpose": "GENERIC_INDEX"}
        if source_kind == "s3":
            params["video"] = {
                "format": fmt,
                "embeddingMode": "AUDIO_VIDEO_COMBINED",
                "source": {"s3Location": {"uri": payload}},
            }
        else:
            params["video"] = {
                "format": fmt,
                "embeddingMode": "AUDIO_VIDEO_COMBINED",
                "source": {"bytes": payload},
            }
        if self.dimensions:
            params["embeddingDimension"] = self.dimensions
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _extract_media_embedding(self, response: Dict[str, Any]) -> List[float]:
        """Extract embedding from provider-specific response."""
        # Titan format
        if "embedding" in response and isinstance(response["embedding"], list):
            return response["embedding"]

        # Nova format
        if "embeddings" in response and isinstance(response["embeddings"], list):
            if response["embeddings"] and isinstance(response["embeddings"][0], dict):
                embedding = response["embeddings"][0].get("embedding")
                if embedding:
                    return embedding

        # Cohere format
        if "embeddings" in response:
            embeddings = response["embeddings"]
            if isinstance(embeddings, dict) and "float" in embeddings:
                return embeddings["float"][0]
            if isinstance(embeddings, list) and embeddings:
                return embeddings[0]

        # Marengo format: {"data": [{"embedding": [...]}]}
        if "data" in response and isinstance(response["data"], list):
            if response["data"] and isinstance(response["data"][0], dict):
                embedding = response["data"][0].get("embedding")
                if embedding:
                    return embedding

        msg = f"No embedding found in response. Keys: {list(response.keys())}"
        raise ValueError(msg)

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

    def embed_image(self, image: MediaInput) -> List[float]:
        """Embed a single image.

        Args:
            image: Image as base64 string, file path, S3 URI (Nova/Marengo), or bytes.

        Returns:
            Embedding vector for the image.

        Raises:
            ValueError: If model doesn't support image embeddings.
        """
        if not self._supports_image():
            msg = f"Image embeddings not supported for model {self.model_id}"
            raise ValueError(msg)

        payload, fmt, source_kind = self._load_media(image, "image")
        body = self._build_image_request(payload, fmt, source_kind)
        response = self._invoke_model(body)
        embedding = self._extract_media_embedding(response)

        if self.normalize:
            return self._normalize_vector(embedding)
        return embedding

    def embed_images(self, images: List[MediaInput]) -> List[List[float]]:
        """Embed multiple images.

        Args:
            images: List of images as base64 strings, file paths, S3 URIs, or bytes.

        Returns:
            List of embedding vectors, one per image.
        """
        return [self.embed_image(img) for img in images]

    def embed_audio(self, audio: MediaInput) -> List[float]:
        """Embed audio content.

        Only supported by Nova models. Audio must be ≤30 seconds for sync API.

        Args:
            audio: Audio as base64 string, file path, S3 URI, or bytes.

        Returns:
            Embedding vector for the audio.

        Raises:
            ValueError: If model doesn't support audio embeddings.
        """
        if not self._supports_audio():
            msg = (
                f"Audio embeddings not supported for model {self.model_id}. "
                "Only Nova models support sync audio embeddings."
            )
            raise ValueError(msg)

        payload, fmt, source_kind = self._load_media(audio, "audio")
        body = self._build_audio_request(payload, fmt, source_kind)
        response = self._invoke_model(body)
        embedding = self._extract_media_embedding(response)

        if self.normalize:
            return self._normalize_vector(embedding)
        return embedding

    def embed_video(self, video: MediaInput) -> List[float]:
        """Embed video content.

        Only supported by Nova models. Video must be ≤30 seconds for sync API.

        Args:
            video: Video as base64 string, file path, S3 URI, or bytes.

        Returns:
            Embedding vector for the video.

        Raises:
            ValueError: If model doesn't support video embeddings.
        """
        if not self._supports_video():
            msg = (
                f"Video embeddings not supported for model {self.model_id}. "
                "Only Nova models support sync video embeddings."
            )
            raise ValueError(msg)

        payload, fmt, source_kind = self._load_media(video, "video")
        body = self._build_video_request(payload, fmt, source_kind)
        response = self._invoke_model(body)
        embedding = self._extract_media_embedding(response)

        if self.normalize:
            return self._normalize_vector(embedding)
        return embedding

    async def aembed_image(self, image: MediaInput) -> List[float]:
        """Asynchronously embed a single image."""
        return await run_in_executor(None, self.embed_image, image)

    async def aembed_images(self, images: List[MediaInput]) -> List[List[float]]:
        """Asynchronously embed multiple images."""
        result = await asyncio.gather(*[self.aembed_image(img) for img in images])
        return list(result)

    async def aembed_audio(self, audio: MediaInput) -> List[float]:
        """Asynchronously embed audio content."""
        return await run_in_executor(None, self.embed_audio, audio)

    async def aembed_video(self, video: MediaInput) -> List[float]:
        """Asynchronously embed video content."""
        return await run_in_executor(None, self.embed_video, video)


def _batch_cohere_embedding_texts(
    texts: List[str], is_v4: bool = False
) -> Generator[List[str], None, None]:
    """Batches a set of texts into chunks acceptable for the Cohere embedding API.

    For Cohere Embed v3: Chunks of at most 96 items, or 2048 characters.
    For Cohere Embed v4: Chunks of at most 96 items, or ~512,000 characters
    (approx 128K tokens).

    """

    max_items = 96
    if is_v4:
        # Cohere Embed v4 supports up to 128K tokens per input
        # Using conservative estimate of ~4 chars per token = ~512K chars
        # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
        max_chars = 512_000
        char_limit_msg = (
            "The Cohere Embed v4 embedding API does not support texts longer than "
            "approximately 128K tokens (~512,000 characters)."
        )
    else:
        # Cohere Embed v3 limit
        # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
        max_chars = 2048
        char_limit_msg = (
            "The Cohere embedding API does not support texts longer than "
            "2048 characters."
        )

    # Initialize batches
    current_batch: List[str] = []
    current_chars = 0

    for text in texts:
        text_len = len(text)

        if text_len > max_chars:
            raise ValueError(char_limit_msg)

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
