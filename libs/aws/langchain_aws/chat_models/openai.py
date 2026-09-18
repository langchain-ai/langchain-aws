"""OpenAI-compatible chat models for Amazon Bedrock endpoints.

Amazon Bedrock exposes OpenAI-compatible Chat Completions and Responses APIs on
two endpoints:

* ``bedrock-mantle`` (``bedrock-mantle.{region}.api.aws``) — served by
  :class:`ChatOpenAIMantle`, which defaults to the ``/v1`` route. Mantle also
  serves some models under ``/openai/v1``; override ``base_url`` for those.
* ``bedrock-runtime`` (``bedrock-runtime.{region}.amazonaws.com/openai/v1``) —
  served by :class:`ChatOpenAIBedrock`.

Because the wire format matches OpenAI, both integrations are thin subclasses of
``BaseChatOpenAI`` (mirroring the ``AzureChatOpenAI`` pattern) that only resolve
the endpoint base URL and authentication; all chat behavior (tool calling,
structured output, streaming, tracing, multimodal) is inherited unchanged. The
shared logic lives in :class:`_BaseChatBedrockOpenAI`; each public class only
sets a handful of endpoint-specific class attributes.
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar, Literal, cast

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
    _reject_guardrail_config_values,
    _reject_guardrail_request_kwargs,
    _strip_cross_region_prefix,
)

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"
_BEDROCK_RUNTIME_BASE_URL_TEMPLATE = (
    "https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
)

# Guardrails are not accepted by the OpenAI models currently served on the
# bedrock-runtime OpenAI-compatible endpoint; used only by ChatOpenAIBedrock.
_BEDROCK_RUNTIME_OAI_GUARDRAILS_ERR_MSG = (
    "Amazon Bedrock Guardrails are not supported by the current OpenAI models "
    "on the bedrock-runtime OpenAI-compatible endpoint. Please use "
    "``ChatAnthropicBedrock`` or ``ChatBedrockConverse`` instead, which support "
    "guardrails via the bedrock-runtime endpoint."
)

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Return the static capability profile for a model, or an empty one.

    Tries the exact model id first (so an explicitly-registered profile such as
    ``global.openai.gpt-5.6-sol`` is used verbatim), then retries with any
    geographic/global prefix stripped (so ``us.openai.gpt-5.6-sol`` inherits the
    base ``openai.gpt-5.6-sol`` profile).
    """
    default = _MODEL_PROFILES.get(model_name)
    if default is None:
        default = _MODEL_PROFILES.get(_strip_cross_region_prefix(model_name))
    return (default or {}).copy()


def _plain_secret(value: Any) -> str | None:
    """Return the plain string for a ``SecretStr``/``str``/``None`` value."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return str(value)


class _BaseChatBedrockOpenAI(BaseChatOpenAI):
    """Shared implementation for Bedrock OpenAI-compatible chat models.

    Not intended for direct use. Concrete subclasses set the endpoint-specific
    class attributes below (``_base_url_template``, ``_endpoint_label``,
    ``_ls_provider_name``, ``_lc_namespace``, ``_llm_type_name``,
    ``_guardrails_err_msg``); everything else — region/base-URL resolution,
    Bedrock API-key auth (static bearer token or short-term keys derived from
    AWS credentials), guardrail rejection, profile resolution, tracing, and the
    Chat Completions/Responses routing — is shared here.
    """

    # --- Endpoint-specific configuration (set by subclasses) ---------------
    _base_url_template: ClassVar[str]
    """``str.format(region=...)`` template for the default endpoint base URL."""

    _endpoint_label: ClassVar[str]
    """Human-readable endpoint name used in error messages."""

    _ls_provider_name: ClassVar[str]
    """Value reported as ``ls_provider`` in tracing metadata."""

    _lc_namespace: ClassVar[list[str]]
    """LangChain serialization namespace for this class."""

    _llm_type_name: ClassVar[str]
    """Value returned by ``_llm_type``."""

    _guardrails_err_msg: ClassVar[str]
    """Error raised when guardrails are configured on an unsupported endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    region_name: str | None = None
    """AWS region for the Bedrock endpoint, e.g. ``us-east-1``.

    Falls back to the ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` environment
    variable when not provided. Used to build the default ``base_url``.
    """

    bedrock_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_BEARER_TOKEN_BEDROCK", default=None)
    )
    """Amazon Bedrock API key (bearer token) used to authenticate.

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

    @classmethod
    def _ensure_concrete(cls) -> None:
        """Guard against instantiating the abstract base directly.

        The endpoint-specific ``ClassVar``s are only set on concrete subclasses,
        so constructing ``_BaseChatBedrockOpenAI`` itself would otherwise fail
        with an opaque ``AttributeError`` deep inside a validator.
        """
        if cls is _BaseChatBedrockOpenAI:
            msg = (
                "_BaseChatBedrockOpenAI is an internal base class and cannot be "
                "instantiated directly. Use ChatOpenAIMantle (bedrock-mantle) or "
                "ChatOpenAIBedrock (bedrock-runtime)."
            )
            raise TypeError(msg)

    @model_validator(mode="before")
    @classmethod
    def _reject_guardrails(cls, values: Any) -> Any:
        cls._ensure_concrete()
        _reject_guardrail_config_values(values, cls._guardrails_err_msg)
        return values

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        _reject_guardrail_request_kwargs(kwargs, self._guardrails_err_msg)
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _set_endpoint_defaults(cls, values: Any) -> Any:
        """Resolve the endpoint base URL and bearer key before the client is built.

        Runs before ``BaseChatOpenAI``'s client-construction validator so the
        OpenAI client is created pointing at the Bedrock endpoint with the
        Bedrock API key.
        """
        cls._ensure_concrete()
        if not isinstance(values, dict):
            return values

        region = (
            values.get("region_name")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        # Default the base URL to the region's endpoint unless the caller
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
                    f"{cls.__name__} could not resolve an AWS region for the "
                    f"{cls._endpoint_label} endpoint. Set `region_name`, the "
                    "`AWS_REGION` or `AWS_DEFAULT_REGION` environment variable, "
                    "or pass an explicit `base_url`. Refusing to default to the "
                    "OpenAI host so the Bedrock API key is never sent to "
                    "api.openai.com."
                )
                raise ValueError(msg)
            values["base_url"] = cls._base_url_template.format(region=region)

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
        return self._llm_type_name

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret field names to environment variables."""
        return {"bedrock_api_key": "AWS_BEARER_TOKEN_BEDROCK"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object."""
        return list(cls._lc_namespace)

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = self._ls_provider_name
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


class ChatOpenAIMantle(_BaseChatBedrockOpenAI):
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
        from ``bedrock-runtime``). For the ``bedrock-runtime`` OpenAI-compatible
        endpoint (e.g. GPT-5.x via cross-Region inference profiles), use
        ``ChatOpenAIBedrock``. For Anthropic Claude on Bedrock, use
        ``ChatAnthropicBedrock``.

        The default ``base_url`` uses Mantle's ``/v1`` route (used by
        open-weight models such as gpt-oss). Some Mantle models are served
        under ``/openai/v1`` instead; pass an explicit ``base_url`` for those.

        As on ``bedrock-runtime``, tool calling with the GPT-5.x reasoning
        models is rejected on the Chat Completions path while reasoning is
        active; use ``use_responses_api=True`` or ``reasoning_effort="none"``
        to bind tools. Open-weight models such as gpt-oss are unaffected.
    """

    _base_url_template = _MANTLE_BASE_URL_TEMPLATE
    _endpoint_label = "Bedrock Mantle"
    _ls_provider_name = "openai-mantle"
    _lc_namespace = ["langchain", "chat_models", "openai_mantle"]
    _llm_type_name = "openai-mantle-chat"
    _guardrails_err_msg = _MANTLE_GUARDRAILS_ERR_MSG


class ChatOpenAIBedrock(_BaseChatBedrockOpenAI):
    """OpenAI-compatible models via the Amazon Bedrock ``bedrock-runtime`` endpoint.

    Talks to the ``bedrock-runtime`` OpenAI-compatible endpoint
    (``bedrock-runtime.{region}.amazonaws.com/openai/v1``) using the OpenAI
    Python SDK. This is the recommended endpoint for OpenAI models (e.g. the
    GPT-5.x family) on Amazon Bedrock, and it supports both the Chat Completions
    and Responses APIs plus cross-Region inference.

    Authentication uses an Amazon Bedrock API key (bearer token). Provide it
    directly (``bedrock_api_key`` / ``AWS_BEARER_TOKEN_BEDROCK``), or omit it and
    let short-term keys be derived from your AWS credentials and refreshed
    transparently.

    See the [LangChain docs for `ChatOpenAI`](https://docs.langchain.com/oss/python/integrations/chat/openai)
    for tutorials and feature walkthroughs — the same features apply here.

    Example:
        ```python
        # pip install "langchain-aws[openai]"
        # export AWS_REGION="us-west-2"

        from langchain_aws import ChatOpenAIBedrock

        # bedrock-runtime requires a cross-Region inference profile id
        # (e.g. "us." or "global." prefix), not a bare model id.
        model = ChatOpenAIBedrock(
            model="us.openai.gpt-5.6-sol",
            region_name="us-west-2",
        )
        model.invoke("What is 2 + 2?")
        ```

    Note:
        Cross-Region inference is selected purely via the ``model`` id — use a
        geographic (``us.``, ``eu.``, ...) or ``global.`` inference-profile id.
        Bare foundation-model ids are rejected by this endpoint. For the
        ``bedrock-mantle`` endpoint (open-weight catalog, server-side tools) use
        ``ChatOpenAIMantle``. For Anthropic Claude on Bedrock, use
        ``ChatAnthropicBedrock``.

        Amazon Bedrock Guardrails are not currently accepted by the OpenAI models
        on this endpoint; configuring them raises an error. Use
        ``ChatAnthropicBedrock`` or ``ChatBedrockConverse`` for guardrails.

        Tool calling with the GPT-5.x reasoning models is rejected on the Chat
        Completions path when reasoning is active. To bind tools (e.g. in a
        LangGraph agent), either use the Responses API
        (``ChatOpenAIBedrock(..., use_responses_api=True)``) or set
        ``reasoning_effort="none"``.
    """

    _base_url_template = _BEDROCK_RUNTIME_BASE_URL_TEMPLATE
    _endpoint_label = "Bedrock Runtime"
    _ls_provider_name = "openai-bedrock"
    _lc_namespace = ["langchain", "chat_models", "openai_bedrock"]
    _llm_type_name = "openai-bedrock-chat"
    _guardrails_err_msg = _BEDROCK_RUNTIME_OAI_GUARDRAILS_ERR_MSG
