"""OpenAI-compatible chat model for the Amazon Bedrock Mantle endpoint.

Amazon Bedrock exposes OpenAI-compatible Chat Completions and Responses APIs on
the ``bedrock-mantle`` endpoint (``bedrock-mantle.{region}.api.aws``). Because the
wire format matches OpenAI, this integration is a thin subclass of
``BaseChatOpenAI`` (mirroring the ``AzureChatOpenAI`` pattern) that only resolves
the Bedrock Mantle base URL and authentication; all chat behaviour (tool calling,
structured output, streaming, tracing, multimodal) is inherited unchanged.
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, cast

from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
    _DictOrPydantic,
    _DictOrPydanticClass,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws._version import _add_langchain_aws_version
from langchain_aws.data._profiles import _PROFILES
from langchain_aws.utils import (
    _BEDROCK_API_KEY_MAX_TTL_SECONDS,
    _MANTLE_GUARDRAILS_ERR_MSG,
    _BedrockApiKeyProvider,
    _check_no_mantle_guardrail_headers,
)

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Return the static capability profile for a Mantle model, or an empty one."""
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _plain_secret(value: Any) -> str | None:
    """Return the plain string for a ``SecretStr``/``str``/``None`` value."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return str(value)


class ChatOpenAIMantle(BaseChatOpenAI):
    """OpenAI-compatible GPT/open-weight models via the Amazon Bedrock Mantle endpoint.

    Talks to the ``bedrock-mantle`` OpenAI-compatible endpoint
    (``bedrock-mantle.{region}.api.aws``) using the OpenAI Python SDK.
    Authentication uses an Amazon Bedrock API key (bearer token) rather than
    AWS SigV4. Provide it directly (``bedrock_api_key`` /
    ``AWS_BEARER_TOKEN_BEDROCK``), or omit it and let short-term keys be derived
    from your AWS credentials and refreshed transparently.

    See the [LangChain docs for `ChatOpenAI`](https://docs.langchain.com/oss/python/integrations/chat/openai)
    for tutorials and feature walkthroughs — the same features apply here.

    Example (static bearer token):
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

    Example (derive short-term keys from AWS credentials):
        ```python
        # pip install "langchain-aws[openai]"
        # Uses the standard boto3 credential chain (env, profile, role, IRSA).

        from langchain_aws import ChatOpenAIMantle

        model = ChatOpenAIMantle(
            model="openai.gpt-oss-120b",
            region_name="us-east-1",
            # optionally: credentials_profile_name="my-profile"
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

    When no static key is available (neither this field nor
    ``AWS_BEARER_TOKEN_BEDROCK``), short-term keys are derived from AWS
    credentials instead — see ``aws_access_key_id`` below.
    """

    aws_access_key_id: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id used to derive short-term Bedrock API keys.

    Only used when no static ``bedrock_api_key``/``AWS_BEARER_TOKEN_BEDROCK`` is
    set. If provided, ``aws_secret_access_key`` must also be provided. When
    omitted, the standard boto3 credential chain (profile, env, IMDS, IRSA, ...)
    is used. Read from ``AWS_ACCESS_KEY_ID`` when not provided.
    """

    aws_secret_access_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret access key used to derive short-term Bedrock API keys.

    Read from ``AWS_SECRET_ACCESS_KEY`` when not provided.
    """

    aws_session_token: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token for temporary credentials.

    Read from ``AWS_SESSION_TOKEN`` when not provided.
    """

    credentials_profile_name: str | None = None
    """Named AWS profile to resolve credentials from when deriving short-term keys."""

    api_key_ttl_seconds: int = _BEDROCK_API_KEY_MAX_TTL_SECONDS
    """Requested lifetime (seconds) for derived short-term Bedrock API keys.

    Capped by Bedrock's server-side maximum and by the underlying credential
    lifetime for refreshable credentials (AssumeRole, SSO, IRSA, ...).
    """

    @model_validator(mode="before")
    @classmethod
    def _reject_guardrails(cls, values: Any) -> Any:
        # TODO: remove after Mantle adds guardrails support
        if isinstance(values, dict):
            if any(
                values.get(key) is not None
                for key in ("guardrail_config", "guardrails")
            ):
                raise ValueError(_MANTLE_GUARDRAILS_ERR_MSG)
            _check_no_mantle_guardrail_headers(values.get("default_headers"))
        return values

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        # TODO: remove after Mantle adds guardrails support
        if kwargs.get("guardrail_config") is not None:
            raise ValueError(_MANTLE_GUARDRAILS_ERR_MSG)
        _check_no_mantle_guardrail_headers(kwargs.get("extra_headers"))
        return super()._get_request_payload(input_, stop=stop, **kwargs)

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
        # supplied one explicitly (either alias form). If no base URL is given and
        # no region can be resolved, fail here rather than let ``BaseChatOpenAI``
        # fall back to the default OpenAI host — otherwise the Bedrock bearer token
        # copied into ``api_key`` below would be sent to ``api.openai.com``.
        has_explicit_base_url = bool(
            values.get("base_url") or values.get("openai_api_base")
        )
        if not has_explicit_base_url:
            if not region:
                msg = (
                    "ChatOpenAIMantle could not resolve an AWS region for the "
                    "Bedrock Mantle endpoint. Set `region_name`, the `AWS_REGION` "
                    "or `AWS_DEFAULT_REGION` environment variable, or pass an "
                    "explicit `base_url`. Refusing to default to the OpenAI host "
                    "so the Bedrock API key is never sent to api.openai.com."
                )
                raise ValueError(msg)
            values["base_url"] = _MANTLE_BASE_URL_TEMPLATE.format(region=region)

        # Route the Bedrock API key into the OpenAI ``api_key`` slot unless the
        # caller already set one explicitly. Precedence:
        #   1. explicit ``api_key``/``openai_api_key`` -> untouched
        #   2. static bearer key (``bedrock_api_key`` / ``AWS_BEARER_TOKEN_BEDROCK``)
        #   3. AWS credentials -> a ``_BedrockApiKeyProvider`` callable that mints
        #      and transparently refreshes short-term keys. ``langchain_openai``
        #      accepts a callable ``api_key`` and resolves it per request.
        if not values.get("api_key") and not values.get("openai_api_key"):
            static_key = values.get("bedrock_api_key") or os.getenv(
                "AWS_BEARER_TOKEN_BEDROCK"
            )
            if static_key:
                values["api_key"] = static_key
            elif region:
                # No static token: derive short-term keys from AWS credentials.
                # Explicitly-passed creds/profile are forwarded; otherwise the
                # provider falls back to the default boto3 credential chain
                # (which also covers the AWS_* env vars) at first use.
                values["api_key"] = _BedrockApiKeyProvider(
                    region=region,
                    ttl_seconds=values.get(
                        "api_key_ttl_seconds", _BEDROCK_API_KEY_MAX_TTL_SECONDS
                    ),
                    aws_access_key_id=_plain_secret(values.get("aws_access_key_id")),
                    aws_secret_access_key=_plain_secret(
                        values.get("aws_secret_access_key")
                    ),
                    aws_session_token=_plain_secret(values.get("aws_session_token")),
                    credentials_profile_name=values.get("credentials_profile_name"),
                )

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

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to the Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to the Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Identical to ``BaseChatOpenAI.with_structured_output`` except that
        ``method`` defaults to ``'json_schema'`` (matching ``ChatOpenAI``).
        """
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )
