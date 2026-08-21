"""Anthropic Bedrock chat models."""

import os
from functools import cached_property
from typing import Any, cast

import anthropic
from anthropic import (
    AnthropicBedrock,
    AnthropicBedrockMantle,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
)
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.language_models import ModelProfile, ModelProfileRegistry
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws._version import _add_langchain_aws_version
from langchain_aws.chat_models._anthropic_utils import _create_bedrock_client_params
from langchain_aws.data._profiles import _PROFILES
from langchain_aws.utils import MODEL_ID_GEO_PREFIXES

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Look up the default profile for a model ID."""
    default = _MODEL_PROFILES.get(model_name)
    if default is None:
        prefix, _, rest = model_name.partition(".")
        if rest and prefix.lower() in MODEL_ID_GEO_PREFIXES:
            default = _MODEL_PROFILES.get(rest)
    return (default or {}).copy()


def _guardrail_config_to_headers(
    guardrail_config: dict[str, Any] | None,
) -> dict[str, str]:
    """Translate a guardrail_config dict into Bedrock guardrail request headers.

    The native Bedrock ``InvokeModel`` / ``InvokeModelWithResponseStream`` APIs
    attach guardrails via HTTP headers instead of a request body field, so the
    Converse-style config is mapped onto those headers here.

    Args:
        guardrail_config: Dict with ``guardrailIdentifier``,
            ``guardrailVersion``, and optional ``trace``.

    Returns:
        Header dict; empty when no guardrail is configured.

    Raises:
        ValueError: If ``guardrail_config`` is missing
            ``guardrailIdentifier`` or ``guardrailVersion``.
    """
    if not guardrail_config:
        return {}
    identifier = guardrail_config.get("guardrailIdentifier")
    version = guardrail_config.get("guardrailVersion")
    if not identifier or not version:
        msg = (
            "guardrail_config requires both 'guardrailIdentifier' and "
            "'guardrailVersion'."
        )
        raise ValueError(msg)
    headers: dict[str, str] = {
        "X-Amzn-Bedrock-GuardrailIdentifier": str(identifier),
        "X-Amzn-Bedrock-GuardrailVersion": str(version),
    }
    if trace := guardrail_config.get("trace"):
        headers["X-Amzn-Bedrock-Trace"] = str(trace).upper()
    return headers


class ChatAnthropicBedrock(ChatAnthropic):
    """Anthropic Claude via AWS Bedrock.

    Uses the `AnthropicBedrock` clients in the `anthropic` SDK.

    See the [LangChain docs for `ChatAnthropic`](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
    for tutorials, feature walkthroughs, and examples.

    See the [Claude Platform docs](https://platform.claude.com/docs/en/about-claude/models/overview)
    for a list of the latest models, their capabilities, and pricing.

    Example:
        ```python
        # pip install -U langchain-anthropic
        # export AWS_ACCESS_KEY_ID="your-access-key"
        # export AWS_SECRET_ACCESS_KEY="your-secret-key"
        # export AWS_REGION="us-east-1"  # or AWS_DEFAULT_REGION

        from langchain_anthropic import ChatAnthropicBedrock

        model = ChatAnthropicBedrock(
            model="us.anthropic.claude-sonnet-4-6",
            # other params...
        )
        ```

    Note:
        Any param which is not explicitly supported will be passed directly to
        [`AnthropicBedrock.messages.create(...)`](https://docs.anthropic.com/en/api/messages)
        each time the model is invoked.
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    region_name: str | None = None
    """The aws region, e.g., `us-west-2`.

    Falls back to AWS_REGION or AWS_DEFAULT_REGION env variable or region specified in
    ~/.aws/config in case it is not provided here.
    """

    aws_access_key_id: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id.

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.

    """

    aws_secret_access_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key.

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token.

    If provided, aws_access_key_id and aws_secret_access_key must
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    guardrail_config: dict[str, Any] | None = Field(default=None, alias="guardrails")
    """Configuration for an Amazon Bedrock Guardrail.

    The native Bedrock ``InvokeModel`` APIs used by the Anthropic SDK attach
    guardrails as HTTP request headers rather than a request body field, so
    this config is translated into ``X-Amzn-Bedrock-GuardrailIdentifier``,
    ``X-Amzn-Bedrock-GuardrailVersion`` and ``X-Amzn-Bedrock-Trace`` headers
    injected per-request via the SDK's ``extra_headers`` parameter.

    When set at construction, applies to every request as the default.
    Can be overridden per-request by passing ``guardrail_config=`` as an
    invoke kwarg (mirroring ``ChatBedrockConverse`` behaviour)::

        model.invoke("hello", guardrail_config={
            "guardrailIdentifier": "gr-other",
            "guardrailVersion": "2",
        })
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-bedrock-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret keys to environment variables."""
        return {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
            "mcp_servers": "ANTHROPIC_MCP_SERVERS",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
        }

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "anthropic-bedrock"]`
        """
        return ["langchain", "chat_models", "anthropic_bedrock"]

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Override to inject additional Bedrock configs via ``extra_headers``."""

        req_guardrail = kwargs.pop("guardrail_config", None)
        guardrail = req_guardrail or self.guardrail_config

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if guardrail:
            guardrail_headers = _guardrail_config_to_headers(guardrail)
            existing_extra = payload.get("extra_headers") or {}
            payload["extra_headers"] = {
                **existing_extra,
                **guardrail_headers,
            }

        return payload

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        """Get client parameters for AnthropicBedrock."""
        region_name = (
            self.region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or None  # let boto3 resolve
        )
        return _create_bedrock_client_params(
            region_name=region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            timeout=self.default_request_timeout,
        )

    @cached_property
    def _client(self) -> Any:  # type: ignore[type-arg]
        """Get synchronous AnthropicBedrock client."""
        return AnthropicBedrock(**self._client_params)

    @cached_property
    def _async_client(self) -> Any:  # type: ignore[type-arg]
        """Get asynchronous AnthropicBedrock client."""
        return AsyncAnthropicBedrock(**self._client_params)

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic-bedrock",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @model_validator(mode="before")
    @classmethod
    def _set_anthropic_api_key(cls, values: dict[str, Any]) -> Any:
        if not values.get("anthropic_api_key"):
            values["anthropic_api_key"] = ""
        return values

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model)
        if self.guardrail_config:
            _guardrail_config_to_headers(self.guardrail_config)
        _add_langchain_aws_version(self)
        return self

    def _make_message_chunk_from_anthropic_event(
        self,
        event: anthropic.types.RawMessageStreamEvent,
        *,
        stream_usage: bool = True,
        coerce_content_to_string: bool,
        block_start_event: anthropic.types.RawMessageStreamEvent | None = None,
    ) -> tuple[AIMessageChunk | None, anthropic.types.RawMessageStreamEvent | None]:
        """Override to capture input_tokens from message_start events.

        Bedrock reports input_tokens on message_start (not message_delta),
        so we attach usage_metadata to the message_start chunk and zero out
        input_tokens on the message_delta chunk to avoid double-counting.
        """
        msg, block_start_event = super()._make_message_chunk_from_anthropic_event(
            event,
            stream_usage=stream_usage,
            coerce_content_to_string=coerce_content_to_string,
            block_start_event=block_start_event,
        )
        if msg is None or not stream_usage:
            return msg, block_start_event

        if event.type == "message_start":
            usage = getattr(event.message, "usage", None)
            if usage is not None:
                input_token_details: dict = {
                    "cache_read": getattr(usage, "cache_read_input_tokens", None),
                    "cache_creation": getattr(
                        usage, "cache_creation_input_tokens", None
                    ),
                }
                input_tokens = (
                    (getattr(usage, "input_tokens", 0) or 0)
                    + (input_token_details["cache_read"] or 0)
                    + (input_token_details["cache_creation"] or 0)
                )
                msg.usage_metadata = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=0,
                    total_tokens=input_tokens,
                    input_token_details=InputTokenDetails(
                        **{
                            k: v
                            for k, v in input_token_details.items()
                            if v is not None
                        },
                    ),
                )
        elif event.type == "message_delta" and msg.usage_metadata is not None:
            output_tokens = msg.usage_metadata.get("output_tokens", 0)
            msg.usage_metadata = UsageMetadata(
                input_tokens=0,
                output_tokens=output_tokens,
                total_tokens=output_tokens,
            )

        return msg, block_start_event


_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/anthropic"


class ChatAnthropicMantle(ChatAnthropic):
    """Anthropic Claude via the Amazon Bedrock Mantle endpoint.

    Amazon Bedrock exposes the native Anthropic Messages API on the
    ``bedrock-mantle`` endpoint (``bedrock-mantle.{region}.api.aws/anthropic``).
    Because the wire format is the standard Anthropic Messages API, this
    integration is a thin subclass of
    [`ChatAnthropic`](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
    (mirroring the ``AzureChatOpenAI`` pattern) that only resolves the Bedrock
    Mantle base URL and authentication; all chat behaviour (tool calling,
    structured output, streaming, tracing, thinking, multimodal) is inherited
    from ``ChatAnthropic`` and stays in sync with ``langchain-anthropic``.

    Uses the ``AnthropicBedrockMantle`` clients in the ``anthropic`` SDK, which
    support both authentication modes Mantle accepts:

    - **Amazon Bedrock API key** (bearer token), from ``bedrock_api_key`` or the
      ``AWS_BEARER_TOKEN_BEDROCK`` environment variable. Explicit SigV4
      credentials take precedence over an environment-sourced API key.
    - **AWS SigV4** with standard AWS credentials — explicit keys, a named
      profile, or the default credential chain (environment, instance profile,
      SSO, etc.). Used automatically whenever no API key is provided.

    See the [Claude Platform docs](https://platform.claude.com/docs/en/about-claude/models/overview)
    for the latest models, their capabilities, and pricing.

    Example:
        ```python
        # pip install "langchain-aws[anthropic]"
        # export AWS_BEARER_TOKEN_BEDROCK="your-bedrock-api-key"
        # export AWS_REGION="us-east-1"

        from langchain_aws import ChatAnthropicMantle

        model = ChatAnthropicMantle(
            model="anthropic.claude-sonnet-5",
            region_name="us-east-1",
        )
        model.invoke("What is 2 + 2?")
        ```

    Note:
        This targets the ``bedrock-mantle`` endpoint. For Claude via
        ``bedrock-runtime`` with AWS SigV4 credentials and invocation logging,
        use ``ChatAnthropicBedrock`` instead.
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
    """Amazon Bedrock API key used to authenticate to Mantle.

    If not provided, read from the ``AWS_BEARER_TOKEN_BEDROCK`` environment
    variable. An explicitly supplied API key takes precedence over SigV4
    credentials. An environment-sourced API key is ignored when a profile or
    either static credential is explicitly supplied and both resolve to values,
    allowing callers to select SigV4 authentication. Otherwise, the client falls
    back to the default AWS credential chain when no API key is available.
    """

    aws_access_key_id: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id.

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.

    """

    aws_secret_access_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key.

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token.

    If provided, aws_access_key_id and aws_secret_access_key must
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    credentials_profile_name: str | None = None
    """AWS profile name from ``~/.aws/credentials`` for SigV4 authentication."""

    @model_validator(mode="before")
    @classmethod
    def _set_anthropic_api_key(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("anthropic_api_key"):
            values["anthropic_api_key"] = ""
        return values

    @property
    def _client_params(self) -> dict[str, Any]:
        """Get client parameters for AnthropicBedrockMantle."""
        region_name = (
            self.region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or None  # SDK will resolve
        )
        client_params: dict[str, Any] = {
            "aws_region": region_name,
            "max_retries": self.max_retries,
            "default_headers": (self.default_headers or None),
        }
        if self.anthropic_api_url and "api.anthropic.com" not in self.anthropic_api_url:
            client_params["base_url"] = self.anthropic_api_url
        explicit_sigv4_credentials = (
            "credentials_profile_name" in self.model_fields_set
            and bool(self.credentials_profile_name)
        ) or (
            bool({"aws_access_key_id", "aws_secret_access_key"} & self.model_fields_set)
            and self.aws_access_key_id is not None
            and self.aws_secret_access_key is not None
        )
        if self.bedrock_api_key and (
            "bedrock_api_key" in self.model_fields_set or not explicit_sigv4_credentials
        ):
            client_params["api_key"] = self.bedrock_api_key.get_secret_value()
        if self.aws_access_key_id:
            client_params["aws_access_key"] = self.aws_access_key_id.get_secret_value()
        if self.aws_secret_access_key:
            client_params["aws_secret_key"] = (
                self.aws_secret_access_key.get_secret_value()
            )
        if self.aws_session_token:
            client_params["aws_session_token"] = (
                self.aws_session_token.get_secret_value()
            )
        if self.credentials_profile_name:
            client_params["aws_profile"] = self.credentials_profile_name
        if (
            self.default_request_timeout is not None
            and self.default_request_timeout > 0
        ):
            client_params["timeout"] = self.default_request_timeout
        return client_params

    @cached_property
    def _client(self) -> Any:  # type: ignore[type-arg]
        """Get synchronous AnthropicBedrockMantle client."""
        return AnthropicBedrockMantle(**self._client_params)

    @cached_property
    def _async_client(self) -> Any:  # type: ignore[type-arg]
        """Get asynchronous AnthropicBedrockMantle client."""
        return AsyncAnthropicBedrockMantle(**self._client_params)

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Resolve the model profile and record the langchain-aws version."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model)
        _add_langchain_aws_version(self)
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-mantle-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret field names to environment variables."""
        return {
            "bedrock_api_key": "AWS_BEARER_TOKEN_BEDROCK",
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is LangChain-serializable.

        ``False`` because this partner class is not registered in
        ``langchain-core``'s deserialization allowlist, so a dumped instance
        cannot be round-tripped via ``langchain_core.load.load``. The
        ``ChatAnthropic`` base returns ``True``, so this override is required.
        """
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "anthropic_mantle"]`
        """
        return ["langchain", "chat_models", "anthropic_mantle"]

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic-mantle",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params
