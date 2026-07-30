"""OpenAI-compatible chat model for the Amazon Bedrock Mantle endpoint.

Amazon Bedrock exposes OpenAI-compatible Chat Completions and Responses APIs on
the ``bedrock-mantle`` endpoint (``bedrock-mantle.{region}.api.aws``). Because the
wire format matches OpenAI, this integration is a thin subclass of
``BaseChatOpenAI`` (mirroring the ``AzureChatOpenAI`` pattern) that only resolves
the Bedrock Mantle base URL and authentication; all chat behaviour (tool calling,
structured output, streaming, tracing, multimodal) is inherited unchanged.
"""

import os
from typing import Any, cast

from langchain_core.language_models import ModelProfile, ModelProfileRegistry
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws._version import _add_langchain_aws_version
from langchain_aws.data._profiles import _PROFILES

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Return the static capability profile for a Mantle model, or an empty one."""
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatOpenAIMantle(BaseChatOpenAI):
    """OpenAI-compatible GPT/open-weight models via the Amazon Bedrock Mantle endpoint.

    Talks to the ``bedrock-mantle`` OpenAI-compatible endpoint
    (``bedrock-mantle.{region}.api.aws``) using the OpenAI Python SDK.
    Authentication uses an Amazon Bedrock API key (bearer token) rather than
    AWS SigV4.

    See the [LangChain docs for `ChatOpenAI`](https://docs.langchain.com/oss/python/integrations/chat/openai)
    for tutorials and feature walkthroughs — the same features apply here.

    Example:
        ```python
        # pip install "langchain-aws[openai]"
        # export AWS_BEARER_TOKEN_BEDROCK="your-bedrock-api-key"
        # export AWS_REGION="us-east-1"

        from langchain_aws import ChatOpenAIMantle

        model = ChatOpenAIMantle(
            model="openai.gpt-oss-120b",
            region_name="us-east-1",
        )
        model.invoke("What is 2 + 2?")
        ```

    Note:
        This targets the ``bedrock-mantle`` endpoint, which serves only the
        OpenAI-compatible model catalog (a different, non-superset set of models
        from ``bedrock-runtime``). For Anthropic Claude on Bedrock, use
        ``ChatAnthropicBedrock``.
    """

    model_config = ConfigDict(populate_by_name=True)

    use_responses_api: bool | None = True
    """Whether to use the Responses API instead of the Chat Completions API.

    Defaults to ``True`` for Bedrock Mantle. Set to ``False`` to use the Chat
    Completions API, or ``None`` to infer from the invocation params.
    """

    region_name: str | None = None
    """AWS region for the Bedrock Mantle endpoint, e.g. ``us-east-1``.

    Falls back to the ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` environment
    variable when not provided. Used to build the default ``base_url``.
    """

    bedrock_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_BEARER_TOKEN_BEDROCK", default=None)
    )
    """Amazon Bedrock API key (bearer token) used to authenticate to Mantle.

    If not provided, read from the ``AWS_BEARER_TOKEN_BEDROCK`` environment
    variable. Sent as the OpenAI ``api_key``.
    """

    @model_validator(mode="before")
    @classmethod
    def _set_mantle_defaults(cls, values: Any) -> Any:
        """Resolve the Mantle base URL and bearer key before the client is built.

        Runs before ``BaseChatOpenAI``'s client-construction validator so the
        OpenAI client is created pointing at the Bedrock Mantle endpoint with the
        Bedrock API key.
        """
        if not isinstance(values, dict):
            return values

        region = (
            values.get("region_name")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        # Default the base URL to the region's Mantle endpoint unless the caller
        # supplied one explicitly (either alias form).
        if not values.get("base_url") and not values.get("openai_api_base") and region:
            values["base_url"] = _MANTLE_BASE_URL_TEMPLATE.format(region=region)

        # Route the Bedrock API key into the OpenAI ``api_key`` slot unless the
        # caller already set one explicitly.
        if not values.get("api_key") and not values.get("openai_api_key"):
            key = values.get("bedrock_api_key") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            if key:
                values["api_key"] = key

        return values

    @model_validator(mode="after")
    def _stamp_version(self) -> Self:
        """Record the langchain-aws version in tracing metadata."""
        _add_langchain_aws_version(self)
        return self

    @model_validator(mode="after")
    def _resolve_profile(self) -> Self:
        """Populate the model profile from static data unless one was supplied."""
        if not self.profile:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-mantle-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret field names to environment variables."""
        return {"bedrock_api_key": "AWS_BEARER_TOKEN_BEDROCK"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object."""
        return ["langchain", "chat_models", "openai_mantle"]

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "openai-mantle"
        return params
