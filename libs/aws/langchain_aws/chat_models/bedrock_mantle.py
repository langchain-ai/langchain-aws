from __future__ import annotations

import json
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.base import LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws.utils import _BEDROCK_API_KEY_MAX_TTL_SECONDS

logger = logging.getLogger(__name__)

# Keys that only exist in the Responses API — their presence forces that path.
_RESPONSES_ONLY_KEYS = frozenset(
    {
        "context_management",
        "include",
        "previous_response_id",
        "reasoning",
        "text",
        "truncation",
    }
)

# Keys that only exist in Chat Completions — their presence forces that path.
_CHAT_COMPLETIONS_ONLY_KEYS = frozenset(
    {
        "response_format",
    }
)

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"


def _openai_response_to_ai_message(choice: Any) -> AIMessage:
    """Convert an OpenAI ChatCompletion choice to an AIMessage."""
    message = choice.message
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            try:
                parsed = json.loads(args)
            except (ValueError, TypeError):
                parsed = args if isinstance(args, dict) else {"raw": args}
            if not isinstance(parsed, dict):
                parsed = {"raw": parsed}
            tool_calls.append(
                {
                    "name": tc.function.name,
                    "args": parsed,
                    "id": tc.id,
                    "type": "tool_call",
                }
            )

    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        additional_kwargs={
            k: v
            for k, v in {
                "refusal": getattr(message, "refusal", None),
            }.items()
            if v is not None
        },
        response_metadata={
            "finish_reason": choice.finish_reason,
        },
    )


def _handle_openai_error(error: Exception) -> None:
    """Handle OpenAI SDK errors, re-raising the original exception.

    Preserves the original error type hierarchy so callers can catch
    specific OpenAI exceptions (AuthenticationError, RateLimitError, etc.).
    Only wraps errors when additional Mantle-specific context is needed.
    """
    try:
        from openai import AuthenticationError, BadRequestError
    except ImportError:
        raise error

    # Some models (openai.gpt-5.*, google.gemma-4-*, xai.grok-4.3) are
    # served at /openai/v1/* instead of /v1/*. Hitting the wrong base
    # path returns "Berm is not enabled for this account" as an
    # AuthenticationError. Log a routing hint; the exception is re-raised
    # unchanged so type-based handling continues to work.
    if isinstance(error, AuthenticationError) and "Berm is not enabled" in str(error):
        logger.warning(
            "Bedrock Mantle returned 'Berm is not enabled for this "
            "account'. This typically means the model requires the "
            "'/openai/v1' base path instead of '/v1'. Affected families "
            "include openai.gpt-5.*, google.gemma-4-*, and xai.grok-4.3. "
            "Set base_url to include the '/openai/v1' prefix — e.g. "
            "'https://bedrock-mantle.<region>.api.aws/openai/v1'."
        )

    # Only add context for BadRequestError cases where Mantle-specific
    # guidance helps the user. All other errors are re-raised unchanged.
    if isinstance(error, BadRequestError):
        msg = str(error)
        if "tool_choice" in msg:
            raise ValueError(
                "Bedrock Mantle rejected the tool_choice value. "
                f"Original error: {error}"
            ) from error
        if "does not support" in msg:
            raise ValueError(
                "Bedrock Mantle: the requested feature is not supported "
                f"for this model. Original error: {error}"
            ) from error

    raise error


def _parse_responses_tool_args(args: Any) -> Dict[str, Any]:
    """Parse tool-call arguments from the Responses API into a dict.

    LangChain expects tool-call ``args`` to be a ``dict``.  The Responses
    API may return a JSON string, a dict, a list, or ``None``.
    """
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (ValueError, TypeError):
        parsed = args if isinstance(args, dict) else {"raw": args}
    if not isinstance(parsed, dict):
        parsed = {"raw": parsed}
    return parsed


def _extract_responses_output(response: Any) -> tuple[str, list]:
    """Extract content text and tool calls from a Responses API response.

    The Responses API emits heterogeneous output items — text messages,
    tool/function calls, reasoning summaries, and others. Some item kinds
    (notably ``reasoning`` items on gpt-5.x) expose a ``content``
    attribute whose value is ``None``, so we guard against that when
    iterating.
    """
    content = ""
    tool_calls: list = []
    if not response.output:
        return content, tool_calls
    for item in response.output:
        item_content = getattr(item, "content", None)
        if item_content is not None:
            for block in item_content:
                block_text = getattr(block, "text", None)
                if block_text:
                    content += block_text
        elif hasattr(item, "name"):
            args = getattr(item, "arguments", "") or ""
            tool_calls.append(
                {
                    "name": item.name,
                    "args": _parse_responses_tool_args(args),
                    "id": getattr(item, "call_id", None) or item.id,
                    "type": "tool_call",
                }
            )
    return content, tool_calls


_ApiFormat = Literal["chat", "responses"]


def _reduce_content_to_text(content: Any) -> str:
    """Reduce a message ``content`` to a plain text string.

    ``content`` may be a plain ``str``, ``None``, or a list of typed
    blocks (``{"type": "text", "text": ...}``, ``{"type": "tool_call",
    ...}``, etc). We keep only text-shaped blocks and drop typed
    non-text blocks — tool calls are re-emitted separately as Responses
    API ``function_call`` items, and multimodal (image / audio / PDF)
    input isn't currently in scope for this class.

    Follow-up: ``MANTLE_FOLLOWUPS.md`` — extend to preserve typed blocks
    when multimodal support is added.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in (None, "text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                # Skip tool_call / tool_use / thinking / image_url /
                # refusal / annotation / other typed blocks.
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _convert_message_to_dict(
    msg: BaseMessage, api: _ApiFormat
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Serialize a LangChain message to the requested API's shape.

    Mirrors the architecture of ``ChatOpenAI._convert_message_to_dict``:
    one helper dispatches on the ``api`` flag to produce either a Chat
    Completions message dict OR a Responses API input-item dict/list.

    Chat Completions returns a single dict per message. Responses API
    may return a **list** of items for one ``AIMessage`` (an assistant
    text turn plus one ``function_call`` item per tool call), or a
    ``function_call_output`` item for a ``ToolMessage``.

    Args:
        msg: A LangChain ``BaseMessage`` subclass.
        api: ``"chat"`` for Chat Completions, ``"responses"`` for the
            Responses API.

    Returns:
        A dict for Chat Completions; a dict or list of dicts for
        Responses API.

    Notes:
        Multimodal content blocks (image URL, audio, PDF), refusal
        blocks, and annotation blocks are currently reduced to plain
        text via ``_reduce_content_to_text``. Extending this helper is
        the canonical place to add multimodal support — tracked in
        ``MANTLE_FOLLOWUPS.md``.
    """
    text = _reduce_content_to_text(msg.content)

    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": text}

    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": text}

    if isinstance(msg, ToolMessage):
        if api == "responses":
            return {
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": text,
            }
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": text,
        }

    if isinstance(msg, AIMessage):
        if api == "chat":
            out: Dict[str, Any] = {"role": "assistant", "content": text}
            if msg.tool_calls:
                out["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.get("id"),
                        "function": {
                            "name": tc.get("name"),
                            "arguments": (
                                tc["args"]
                                if isinstance(tc.get("args"), str)
                                else json.dumps(tc.get("args", {}))
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return out

        # api == "responses": emit text (if any), then a
        # ``function_call`` item per tool call.
        items: List[Dict[str, Any]] = []
        if text:
            items.append({"role": "assistant", "content": text})
        for tc in msg.tool_calls:
            args = tc.get("args", {})
            args_str = args if isinstance(args, str) else json.dumps(args)
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc.get("id"),
                    "name": tc.get("name"),
                    "arguments": args_str,
                }
            )
        return items

    # Unknown message subclass — fall back to langchain-core's
    # canonical Chat-Completions adapter.
    fallback = convert_to_openai_messages([msg])
    if isinstance(fallback, list) and fallback:
        return fallback[0]
    if isinstance(fallback, dict):
        return fallback
    return {"role": "user", "content": text}


def _messages_to_api_input(
    messages: Sequence[BaseMessage], api: _ApiFormat
) -> List[Dict[str, Any]]:
    """Convert a sequence of LangChain messages to the target API's input list.

    Flattens the per-message result of ``_convert_message_to_dict``:
    Responses API messages that fan out into multiple items (assistant
    text + one item per tool call) are appended in order.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        result = _convert_message_to_dict(msg, api)
        if isinstance(result, list):
            out.extend(result)
        else:
            out.append(result)
    return out


def _get_last_messages_and_response_id(
    messages: Sequence[BaseMessage],
) -> tuple[list[BaseMessage], Optional[str]]:
    """Extract the most recent ``previous_response_id`` from message history.

    Scans backward for the last ``AIMessage`` whose ``response_metadata["id"]``
    starts with ``resp_``. Returns the messages after that point and the
    response ID. If no such message is found, returns all messages and
    ``None``.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            resp_id = msg.response_metadata.get("id", "") or ""
            if resp_id.startswith("resp_"):
                return list(messages[i + 1 :]), resp_id
    return list(messages), None


def _build_chat_completions_result(response: Any) -> ChatResult:
    """Build a ``ChatResult`` from a Chat Completions API response."""
    choice = response.choices[0]
    ai_msg = _openai_response_to_ai_message(choice)

    if response.usage:
        ai_msg.usage_metadata = UsageMetadata(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    ai_msg.response_metadata["model"] = response.model
    ai_msg.response_metadata["model_name"] = response.model
    ai_msg.response_metadata["model_provider"] = "bedrock_mantle"
    ai_msg.response_metadata["id"] = response.id
    return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def _build_responses_result(response: Any) -> ChatResult:
    """Build a ``ChatResult`` from a Responses API response object."""
    content, tool_calls = _extract_responses_output(response)

    ai_msg = AIMessage(
        content=content,
        tool_calls=tool_calls,
        id=response.id,
        response_metadata={
            "id": response.id,
            "model": response.model,
            "model_name": response.model,
            "model_provider": "bedrock_mantle",
            "status": getattr(response, "status", None),
            # ``incomplete_details`` is only populated when the model
            # hit ``max_output_tokens`` or the content filter — mirrors
            # ChatOpenAI's Responses API metadata (base.py:4434-4448).
            **(
                {
                    "incomplete_details": (
                        response.incomplete_details.model_dump(exclude_none=True)
                        if hasattr(response.incomplete_details, "model_dump")
                        else response.incomplete_details
                    )
                }
                if getattr(response, "incomplete_details", None) is not None
                else {}
            ),
        },
    )

    if hasattr(response, "usage") and response.usage:
        ai_msg.usage_metadata = UsageMetadata(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

    return ChatResult(generations=[ChatGeneration(message=ai_msg)])


class ChatBedrockMantle(BaseChatModel):
    """LangChain chat model for Amazon Bedrock via the Mantle inference engine.

    Supports models available on the ``bedrock-mantle`` endpoint using the
    OpenAI-compatible Responses API and Chat Completions API.

    Setup:
        Install the optional Mantle dependencies::

            pip install "langchain-aws[mantle]"

    Instantiate:
        .. code-block:: python

            from langchain_aws import ChatBedrockMantle

            llm = ChatBedrockMantle(
                model="deepseek.v3.2",
                region_name="us-east-1",
            )

    Invoke:
        .. code-block:: python

            response = llm.invoke("What is 2+2?")

    Stream:
        .. code-block:: python

            for chunk in llm.stream("Say hello in 3 languages"):
                print(chunk.content)

    Async:
        .. code-block:: python

            response = await llm.ainvoke("What is 2+2?")

    Conversation state with previous_response_id:
        .. code-block:: python

            llm = ChatBedrockMantle(
                model="deepseek.v3.2",
                region_name="us-east-1",
            )
            response = llm.invoke("Hello")
            response_id = response.response_metadata["id"]
            followup = llm.invoke(
                "Tell me more",
                previous_response_id=response_id,
            )
    """

    model: str = Field(..., alias="model")
    """Mantle model ID, e.g. ``deepseek.v3.2``, ``qwen.qwen3-32b``."""

    region_name: Optional[str] = None
    """AWS region. Used to construct the Mantle endpoint URL."""

    base_url: Optional[str] = None
    """Override the Mantle endpoint URL."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional parameters passed to the underlying SDK."""

    use_responses_api: Optional[bool] = None
    """Whether to use the Responses API instead of Chat Completions.

    Applies to OpenAI-compatible models only.

    - ``True``: always use the Responses API.
    - ``False``: always use the Chat Completions API.
    - ``None`` (default): auto-detect per request. Defaults to the
      Responses API unless Chat-Completions-only parameters (e.g.
      ``response_format``) are present.
    """

    use_previous_response_id: bool = False
    """If ``True``, automatically pass ``previous_response_id`` using the ID
    of the most recent ``AIMessage`` in the input message sequence. Messages
    up to that response are dropped from the request payload. Applies to
    OpenAI-compatible models only (Responses API).

    Trade-offs to be aware of:

    - **Server-side state dependency.** The Responses API stores conversation
      state server-side and looks it up by response ID. If Mantle's retention
      window has elapsed between calls (e.g. a LangGraph graph resumed after
      a long delay), the ``previous_response_id`` will be rejected. The
      client does not retry with the full message history; the error
      propagates to the caller. If you cannot tolerate this failure mode
      (e.g. long-running agentic workflows), keep this ``False`` and send
      the full message history each time.
    - **Not portable across endpoints.** ``resp_*`` IDs are bound to the
      region and endpoint that produced them. Resuming in a different region
      or with a different ``base_url`` will fail.
    - **Intermediate tool messages are dropped.** When enabled, only messages
      after the last ``AIMessage`` with a ``resp_*`` id are sent; earlier
      tool-call / tool-result messages are assumed to be reconstructible
      from server-side state.
    - **Redundant storage with checkpointers.** If a LangGraph checkpointer
      is also in use, it will store the full message history but those
      messages are never sent to the API when this flag is on. That is
      redundant storage cost with no functional benefit.
    """

    stream_usage: Optional[bool] = None
    """Whether to include usage metadata in streaming output.

    Applies to OpenAI-compatible models only.

    If ``True``, ``stream_options={"include_usage": True}`` is sent.
    If ``False``, usage is not requested.  If ``None`` (default),
    usage is included.
    """

    max_retries: int = 2
    """Maximum number of retries for transient errors (5xx, connection, timeout)."""

    timeout: Optional[float] = None
    """Request timeout in seconds. Applies to both connect and read."""

    bedrock_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Static Bedrock API key. If provided, short-term key generation is skipped."""

    # TODO: Consolidate the AWS credential fields below with the identical
    # declarations on ChatBedrockConverse and BedrockLLM into a shared
    # mixin. Deferred to keep this PR scoped to reviewer comment #21.
    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id used to generate short-term Bedrock API keys.

    If provided, ``aws_secret_access_key`` must also be provided. Takes
    precedence over ``credentials_profile_name`` and the default AWS
    credential chain. Not used when a static ``bedrock_api_key`` /
    ``api_key`` is supplied.

    Falls back to the ``AWS_ACCESS_KEY_ID`` environment variable.
    """

    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret access key used to generate short-term Bedrock API keys.

    Required when ``aws_access_key_id`` is set. Falls back to the
    ``AWS_SECRET_ACCESS_KEY`` environment variable.
    """

    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """Optional AWS session token for temporary credentials.

    Only meaningful when ``aws_access_key_id`` and ``aws_secret_access_key``
    are also provided. Falls back to the ``AWS_SESSION_TOKEN`` environment
    variable.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """Named AWS profile from ``~/.aws/credentials`` / ``~/.aws/config``.

    Used only when explicit ``aws_access_key_id`` / ``aws_secret_access_key``
    are not set. If neither explicit creds nor a profile name are
    provided, the default AWS credential chain is used (env vars, IAM
    instance role, container credentials, etc.).
    """

    bedrock_api_key_ttl_seconds: int = _BEDROCK_API_KEY_MAX_TTL_SECONDS
    """Requested TTL (in seconds) for auto-refreshed short-term Bedrock API keys.

    Only applies when short-term keys are auto-generated from AWS
    credentials (not when a static ``bedrock_api_key`` / ``api_key`` is
    supplied).

    - Must be greater than 0 and less than or equal to ``43200`` (12
      hours). Bedrock rejects longer expiries at the server side.
    - Default: ``43200`` (12 hours) — the longest lifetime Bedrock
      allows; minimizes refresh overhead.
    - The effective TTL is further capped by the underlying IAM
      credentials' remaining lifetime, so passing a value longer than
      your STS session lifetime has no effect beyond the credentials'
      natural expiry.

    Set a lower value to trade slightly higher refresh overhead for a
    tighter security posture (shorter-lived tokens on the wire).
    """

    disabled_params: Optional[Dict[str, Any]] = None
    """Parameters to silently drop before sending to the API.

    Useful when a model does not support certain parameters that are
    set internally (e.g., by ``with_structured_output``).

    Format: ``{"param_name": None}`` to always drop, or
    ``{"param_name": ["val1", "val2"]}`` to drop only specific values.

    Example::

        llm = ChatBedrockMantle(
            model="some-model",
            disabled_params={"parallel_tool_calls": None},
        )
    """

    _sync_client: Any = None
    _async_client: Any = None
    _resolved_region: str = ""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_environment(self) -> Self:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Could not import openai. "
                'Please install it via: pip install "langchain-aws[mantle]"'
            ) from exc

        if bool(self.aws_access_key_id) != bool(self.aws_secret_access_key):
            msg = (
                "aws_access_key_id and aws_secret_access_key must both be "
                "provided, or both left unset."
            )
            raise ValueError(msg)
        if not 0 < self.bedrock_api_key_ttl_seconds <= _BEDROCK_API_KEY_MAX_TTL_SECONDS:
            msg = (
                "bedrock_api_key_ttl_seconds must be between 1 and "
                f"{_BEDROCK_API_KEY_MAX_TTL_SECONDS} "
                "(12 hours, Bedrock's server-side max); got "
                f"{self.bedrock_api_key_ttl_seconds}."
            )
            raise ValueError(msg)

        region = (
            self.region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        if not region and not self.base_url:
            try:
                import botocore.session

                region = botocore.session.Session().get_config_variable("region")
            except (ImportError, OSError, ValueError) as exc:
                # ImportError: botocore not installed
                # OSError: config file not readable
                # ValueError: malformed config
                logger.debug("Could not resolve region from botocore: %s", exc)
        if not region and not self.base_url:
            raise ValueError(
                "region_name must be provided, or set AWS_REGION / "
                "AWS_DEFAULT_REGION, or provide base_url."
            )
        self._resolved_region = region or ""

        url = self.base_url or _MANTLE_BASE_URL_TEMPLATE.format(
            region=self._resolved_region
        )

        if self.bedrock_api_key:
            api_key: Any = self.bedrock_api_key.get_secret_value()
        else:
            try:
                from importlib.util import find_spec

                if find_spec("aws_bedrock_token_generator") is None:
                    raise ImportError("aws_bedrock_token_generator not found")
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "Could not import aws-bedrock-token-generator. "
                    'Please install it via: pip install "langchain-aws[mantle]"'
                ) from exc

            from langchain_aws.utils import _BedrockApiKeyProvider

            api_key = _BedrockApiKeyProvider(
                self._resolved_region,
                ttl_seconds=self.bedrock_api_key_ttl_seconds,
                aws_access_key_id=(
                    self.aws_access_key_id.get_secret_value()
                    if self.aws_access_key_id
                    else None
                ),
                aws_secret_access_key=(
                    self.aws_secret_access_key.get_secret_value()
                    if self.aws_secret_access_key
                    else None
                ),
                aws_session_token=(
                    self.aws_session_token.get_secret_value()
                    if self.aws_session_token
                    else None
                ),
                credentials_profile_name=self.credentials_profile_name,
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": url,
            "max_retries": self.max_retries,
            "default_headers": {
                "x-client-framework": "langchain-aws",
            },
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        self._sync_client = OpenAI(**client_kwargs)
        async_kwargs = {**client_kwargs}
        if callable(api_key) and hasattr(api_key, "async_call"):
            async_kwargs["api_key"] = api_key.async_call
        self._async_client = AsyncOpenAI(**async_kwargs)  # type: ignore[arg-type]
        return self

    @property
    def _llm_type(self) -> str:
        return "bedrock-mantle"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters for tracing and callbacks."""
        return {
            "model": self.model,
            "region_name": self._resolved_region,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_responses_api": self.use_responses_api,
        }

    def _filter_disabled_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Remove parameters listed in ``disabled_params``.

        If a param is mapped to ``None``, it is always removed.
        If mapped to a list of values, it is removed only when its
        value is in that list.
        """
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
            if k in self.disabled_params:
                disabled_values = self.disabled_params[k]
                if disabled_values is None or v in disabled_values:
                    continue
            filtered[k] = v
        return filtered

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for LangSmith tracing."""
        ls_params = LangSmithParams(
            ls_provider="amazon_bedrock_mantle",
            ls_model_name=kwargs.get("model") or self.model,
            ls_model_type="chat",
            ls_temperature=self.temperature,
        )
        if self.max_tokens:
            ls_params["ls_max_tokens"] = self.max_tokens
        if stop:
            ls_params["ls_stop"] = stop
        return ls_params

    def _use_responses_api(self, params: Dict[str, Any]) -> bool:
        """Decide whether to route this request to the Responses API.

        If ``use_responses_api`` is set explicitly (``True`` / ``False``),
        that value is honored.  Otherwise defaults to the Responses API
        unless Chat-Completions-only parameters are present.
        """
        if isinstance(self.use_responses_api, bool):
            return self.use_responses_api

        # If Responses-API-only keys are present, always use Responses API
        if _RESPONSES_ONLY_KEYS.intersection(params):
            return True

        # Fall back to Chat Completions for Chat-Completions-only keys
        if _CHAT_COMPLETIONS_ONLY_KEYS.intersection(params):
            return False

        # Default: Responses API
        return True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.use_previous_response_id:
            messages, prev_id = _get_last_messages_and_response_id(messages)
            if prev_id:
                kwargs.setdefault("previous_response_id", prev_id)
        params = self._build_params(stop=stop, stream=False, **kwargs)
        logger.debug("ChatBedrockMantle input messages: %s", messages)
        logger.debug("ChatBedrockMantle params: %s", params)

        try:
            if self._use_responses_api(params):
                return self._generate_responses_api(messages, params)
            return self._generate_chat_completions(messages, params)
        except Exception as e:
            _handle_openai_error(e)
            raise  # unreachable, but satisfies type checker

    def _generate_chat_completions(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
    ) -> ChatResult:
        """Generate via the Chat Completions API."""
        openai_msgs = _messages_to_api_input(messages, "chat")
        response = self._sync_client.chat.completions.create(
            messages=openai_msgs, **params
        )
        logger.debug("ChatBedrockMantle response: %s", response)
        return _build_chat_completions_result(response)

    def _generate_responses_api(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
    ) -> ChatResult:
        """Generate via the Responses API (``/v1/responses``)."""
        resp_params = self._prepare_responses_params(params)
        input_items = _messages_to_api_input(messages, "responses")

        response = self._sync_client.responses.create(input=input_items, **resp_params)
        return _build_responses_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.use_previous_response_id:
            messages, prev_id = _get_last_messages_and_response_id(messages)
            if prev_id:
                kwargs.setdefault("previous_response_id", prev_id)
        params = self._build_params(stop=stop, stream=False, **kwargs)

        try:
            if self._use_responses_api(params):
                return await self._agenerate_responses_api(messages, params)
            return await self._agenerate_chat_completions(messages, params)
        except Exception as e:
            _handle_openai_error(e)
            raise

    async def _agenerate_chat_completions(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
    ) -> ChatResult:
        """Async generate via the Chat Completions API."""
        openai_msgs = _messages_to_api_input(messages, "chat")
        response = await self._async_client.chat.completions.create(
            messages=openai_msgs, **params
        )
        return _build_chat_completions_result(response)

    async def _agenerate_responses_api(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
    ) -> ChatResult:
        """Async generate via the Responses API (``/v1/responses``)."""
        resp_params = self._prepare_responses_params(params)
        input_items = _messages_to_api_input(messages, "responses")

        response = await self._async_client.responses.create(
            input=input_items, **resp_params
        )
        return _build_responses_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.use_previous_response_id:
            messages, prev_id = _get_last_messages_and_response_id(messages)
            if prev_id:
                kwargs.setdefault("previous_response_id", prev_id)
        params = self._build_params(stop=stop, stream=True, **kwargs)
        if self._use_responses_api(params):
            yield from self._stream_responses(messages, params, run_manager)
        else:
            yield from self._stream_chat_completions(messages, params, run_manager)

    def _stream_chat_completions(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream via the Chat Completions API."""
        openai_msgs = _messages_to_api_input(messages, "chat")
        logger.debug("ChatBedrockMantle streaming (chat) with params: %s", params)

        try:
            context_manager = self._sync_client.chat.completions.create(
                messages=openai_msgs, **params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        with context_manager as stream:
            for chunk in stream:
                result = self._process_chat_stream_chunk(chunk)
                if result is None:
                    continue
                token, gen_chunk = result
                # Tag the chunk with the provider identifier — mirrors
                # ChatBedrockConverse / ChatOpenAI: ``model_provider`` is
                # a client-side constant emitted on every chunk so
                # downstream consumers can identify the source in
                # real time. ``model_name`` is set once (in the
                # terminal-chunk branch of the event handlers).
                gen_chunk.message.response_metadata["model_provider"] = "bedrock_mantle"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=gen_chunk)
                yield gen_chunk

    def _stream_responses(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream via the Responses API (``/v1/responses``)."""
        resp_params = self._prepare_responses_params(params)
        input_items = _messages_to_api_input(messages, "responses")
        logger.debug(
            "ChatBedrockMantle streaming (responses) with params: %s", resp_params
        )

        try:
            context_manager = self._sync_client.responses.create(
                input=input_items, stream=True, **resp_params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        with context_manager as stream:
            for event in stream:
                result = self._process_responses_stream_event(event)
                if result is None:
                    continue
                token, gen_chunk = result
                # Tag the chunk with the provider identifier — mirrors
                # ChatBedrockConverse / ChatOpenAI: ``model_provider`` is
                # a client-side constant emitted on every chunk so
                # downstream consumers can identify the source in
                # real time. ``model_name`` is set once (in the
                # terminal-chunk branch of the event handlers).
                gen_chunk.message.response_metadata["model_provider"] = "bedrock_mantle"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=gen_chunk)
                yield gen_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.use_previous_response_id:
            messages, prev_id = _get_last_messages_and_response_id(messages)
            if prev_id:
                kwargs.setdefault("previous_response_id", prev_id)
        params = self._build_params(stop=stop, stream=True, **kwargs)
        if self._use_responses_api(params):
            async for chunk in self._astream_responses(messages, params, run_manager):
                yield chunk
        else:
            async for chunk in self._astream_chat_completions(
                messages, params, run_manager
            ):
                yield chunk

    async def _astream_chat_completions(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream via the Chat Completions API."""
        openai_msgs = _messages_to_api_input(messages, "chat")

        try:
            context_manager = await self._async_client.chat.completions.create(
                messages=openai_msgs, **params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        async with context_manager as stream:
            async for chunk in stream:
                result = self._process_chat_stream_chunk(chunk)
                if result is None:
                    continue
                token, gen_chunk = result
                gen_chunk.message.response_metadata["model_provider"] = "bedrock_mantle"
                if run_manager:
                    await run_manager.on_llm_new_token(token, chunk=gen_chunk)
                yield gen_chunk

    async def _astream_responses(
        self,
        messages: List[BaseMessage],
        params: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream via the Responses API (``/v1/responses``)."""
        resp_params = self._prepare_responses_params(params)
        input_items = _messages_to_api_input(messages, "responses")

        try:
            context_manager = await self._async_client.responses.create(
                input=input_items, stream=True, **resp_params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        async with context_manager as stream:
            async for event in stream:
                result = self._process_responses_stream_event(event)
                if result is None:
                    continue
                token, gen_chunk = result
                gen_chunk.message.response_metadata["model_provider"] = "bedrock_mantle"
                if run_manager:
                    await run_manager.on_llm_new_token(token, chunk=gen_chunk)
                yield gen_chunk

    def _process_chat_stream_chunk(
        self, chunk: Any
    ) -> Optional[tuple[str, ChatGenerationChunk]]:
        """Convert a Chat Completions streaming chunk to a (token, chunk) pair.

        With ``stream_options.include_usage=True``, the OpenAI spec emits a
        terminal chunk with an empty ``choices`` array and populated
        ``usage``. That chunk is surfaced here as an empty-content
        ``AIMessageChunk`` carrying ``usage_metadata``.
        """
        has_usage = hasattr(chunk, "usage") and chunk.usage

        if not chunk.choices:
            if has_usage:
                # Terminal chunk carrying only ``usage`` (emitted when
                # ``stream_options.include_usage=True``). Populate the
                # same identifying fields the non-streaming path sets so
                # the merged final AIMessage carries them.
                msg_chunk = AIMessageChunk(
                    content="",
                    id=chunk.id,
                    usage_metadata=UsageMetadata(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    ),
                    response_metadata={
                        "id": chunk.id,
                        "model": chunk.model,
                        "model_name": chunk.model,
                    },
                )
                return "", ChatGenerationChunk(message=msg_chunk)
            return None

        delta = chunk.choices[0].delta
        content: str = delta.content or ""

        tc_chunks = []
        if delta.tool_calls:
            for tc in delta.tool_calls:
                tc_chunks.append(
                    tool_call_chunk(
                        name=tc.function.name if tc.function else None,
                        args=tc.function.arguments if tc.function else None,
                        id=tc.id,
                        index=tc.index,
                    )
                )

        msg_chunk = AIMessageChunk(
            content=content,
            tool_call_chunks=tc_chunks,
        )

        if chunk.choices[0].finish_reason:
            msg_chunk.response_metadata["finish_reason"] = chunk.choices[
                0
            ].finish_reason

        if has_usage:
            msg_chunk.usage_metadata = UsageMetadata(
                input_tokens=chunk.usage.prompt_tokens or 0,
                output_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )

        return content, ChatGenerationChunk(message=msg_chunk)

    def _process_responses_stream_event(
        self, event: Any
    ) -> Optional[tuple[str, ChatGenerationChunk]]:
        """Convert a Responses API SSE event to a (token, chunk) pair.

        Handled event types:

        - ``response.output_text.delta`` — text token
        - ``response.output_item.added`` (type ``function_call``) — tool-call start
        - ``response.function_call_arguments.delta`` — tool-call arg fragment
        - ``response.completed`` — final usage metadata
        """
        if event.type == "response.output_text.delta":
            text: str = event.delta or ""
            msg_chunk = AIMessageChunk(content=text)
            return text, ChatGenerationChunk(message=msg_chunk)

        if (
            event.type == "response.output_item.added"
            and hasattr(event, "item")
            and getattr(event.item, "type", None) == "function_call"
        ):
            tc = tool_call_chunk(
                name=event.item.name,
                args="",
                id=event.item.call_id,
                index=event.output_index if hasattr(event, "output_index") else 0,
            )
            msg_chunk = AIMessageChunk(content="", tool_call_chunks=[tc])
            return "", ChatGenerationChunk(message=msg_chunk)

        if event.type == "response.function_call_arguments.delta":
            tc = tool_call_chunk(
                name=None,
                args=event.delta,
                id=None,
                index=event.output_index if hasattr(event, "output_index") else 0,
            )
            msg_chunk = AIMessageChunk(content="", tool_call_chunks=[tc])
            return "", ChatGenerationChunk(message=msg_chunk)

        if event.type in ("response.completed", "response.incomplete"):
            # ``response.incomplete`` fires when the model hit
            # ``max_output_tokens`` (typically during reasoning) before
            # producing text. We still emit a final chunk so callers
            # never see "No generation chunks were returned".
            #
            # Match ChatOpenAI's Responses API convention: surface
            # ``status`` and ``incomplete_details`` verbatim from the
            # SDK rather than synthesizing a ``finish_reason`` field.
            # ``finish_reason`` is a Chat Completions concept —
            # ``ChatOpenAI`` deliberately doesn't invent it for
            # Responses, and callers of both classes drill into
            # ``response_metadata["incomplete_details"]["reason"]`` for
            # the same information.
            response = event.response
            usage_metadata = None
            if hasattr(response, "usage") and response.usage:
                usage_metadata = UsageMetadata(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            metadata: Dict[str, Any] = {
                "id": response.id,
                "model": response.model,
                "model_name": response.model,
                "status": getattr(response, "status", None),
            }
            incomplete = getattr(response, "incomplete_details", None)
            if incomplete is not None:
                # Pydantic model on the SDK — dump to a plain dict so it
                # serializes cleanly through LangSmith / model_dump().
                if hasattr(incomplete, "model_dump"):
                    metadata["incomplete_details"] = incomplete.model_dump(
                        exclude_none=True
                    )
                else:
                    metadata["incomplete_details"] = incomplete
            msg_chunk = AIMessageChunk(
                content="",
                id=response.id,
                usage_metadata=usage_metadata,
                response_metadata=metadata,
            )
            return "", ChatGenerationChunk(message=msg_chunk)

        return None

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[str | Dict | bool] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool definitions for function calling.

        Args:
            tools: A list of tool definitions to bind.
            tool_choice: Which tool to require. Options:
                - str tool name: forces that specific tool
                - ``"auto"``: model decides (default)
                - ``"none"``: no tool calling
                - ``"required"`` or ``True``: force at least one tool call
                - dict: raw tool_choice object
            strict: If ``True``, model output is guaranteed to match the
                tool's JSON Schema exactly.
            parallel_tool_calls: If ``False``, disable parallel tool use.
            kwargs: Additional parameters passed to ``bind()``.
        """
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted = [convert_to_openai_tool(t, strict=strict) for t in tools]
        if tool_choice:
            if isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, str):
                # Map tool names to the dict format
                tool_names = [
                    t["function"]["name"] for t in formatted if "function" in t
                ]
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice == "any":
                    tool_choice = "required"
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Return a runnable that produces structured output.

        Args:
            schema: A Pydantic model, TypedDict, or JSON Schema dict
                describing the desired output shape.
            method: One of ``"json_schema"`` (default — uses the Responses
                API ``text.format``), ``"function_calling"`` (uses tool
                calling), or ``"json_mode"`` (uses Chat Completions
                ``response_format={"type": "json_object"}``).
            include_raw: If ``True``, return a dict with ``raw``,
                ``parsed``, and ``parsing_error`` keys.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            A ``Runnable`` that outputs structured data.
        """
        from operator import itemgetter

        from langchain_core.output_parsers import (
            JsonOutputParser,
            PydanticOutputParser,
        )
        from langchain_core.output_parsers.openai_tools import (
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )
        from langchain_core.runnables import RunnableMap, RunnablePassthrough
        from langchain_core.utils.pydantic import is_basemodel_subclass

        is_pydantic = isinstance(schema, type) and is_basemodel_subclass(schema)

        if method == "function_calling":
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs = self._filter_disabled_params(
                tool_choice=tool_name,
                parallel_tool_calls=False,
                **kwargs,
            )
            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic:
                output_parser: Any = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"}, **kwargs)
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic
                else JsonOutputParser()
            )
        elif method == "json_schema":
            # ``convert_to_json_schema`` handles Pydantic v2, Pydantic v1,
            # and TypedDict uniformly. Plain-dict inputs are treated as
            # already-formed JSON Schema and deep-copied — passing them
            # through ``convert_to_json_schema`` would silently rewrite
            # openai-tool-shaped dicts (``{name, description, parameters}``)
            # in ways the caller didn't ask for. Matches
            # ``ChatBedrockConverse._with_structured_output_prompt_prefill``
            # (bedrock_converse.py:1701-1706).
            import copy

            from langchain_core.utils.function_calling import (
                convert_to_json_schema,
            )

            if isinstance(schema, dict):
                json_schema = copy.deepcopy(schema)
            else:
                json_schema = convert_to_json_schema(schema)

            schema_name = json_schema.get(
                "title", getattr(schema, "__name__", "Output")
            )
            # Always bind in Chat Completions shape. ``_prepare_responses_params``
            # transparently remaps to the Responses API's ``text.format`` if
            # the request is routed there. Callers on Responses-only models
            # (e.g. ``openai.gpt-5.x``) must set ``use_responses_api=True``
            # so the router doesn't force Chat Completions.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                },
            }
            # LangSmith tracing hint: identifies the structured-output
            # method/schema for callback consumers (matches the shape used
            # by ChatBedrockConverse and ChatOpenAI).
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": json_schema,
                },
                **kwargs,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method '{method}'. Expected one of "
                f"'function_calling', 'json_mode', or 'json_schema'."
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the Mantle ``/v1/models`` endpoint."""
        try:
            response = self._sync_client.models.list()
        except Exception as e:
            _handle_openai_error(e)
            raise
        return [
            {"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in response
        ]

    def _build_params(
        self,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build parameters dict for the OpenAI SDK call."""
        params: Dict[str, Any] = {
            "model": self.model,
            "stream": stream,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if stop:
            params["stop"] = stop

        merged = {**self.model_kwargs, **kwargs}
        if "tools" in merged:
            params["tools"] = merged.pop("tools")
        if "tool_choice" in merged:
            params["tool_choice"] = merged.pop("tool_choice")
        if "previous_response_id" in merged:
            params["previous_response_id"] = merged.pop("previous_response_id")

        for k, v in merged.items():
            if k not in params:
                params[k] = v

        if stream:
            should_include_usage = (
                self.stream_usage if self.stream_usage is not None else True
            )
            if should_include_usage:
                params["stream_options"] = {"include_usage": True}

        return params

    def _prepare_responses_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt ``_build_params`` output for the Responses API.

        Pops/remaps keys that differ between Chat Completions and the
        Responses API so callers don't need to worry about the difference.
        """
        resp_params = dict(params)
        resp_params.pop("stream", None)
        stop = resp_params.pop("stop", None)
        if stop:
            logger.warning(
                "Bedrock Mantle Responses API does not support 'stop' "
                "sequences. The stop=%r parameter will be ignored. Use "
                "use_responses_api=False to route through Chat Completions "
                "which supports stop sequences.",
                stop,
            )
        resp_params.pop("stream_options", None)
        max_tokens = resp_params.pop("max_tokens", None)
        if max_tokens is not None:
            resp_params["max_output_tokens"] = max_tokens

        # Remap response_format to text (Responses API equivalent).
        # Chat Completions:
        #   response_format={"type": "json_schema", "json_schema": {...}}
        # Responses API:
        #   text={"format": {"type": "json_schema", ...}}
        if "response_format" in resp_params:
            rf = resp_params.pop("response_format")
            if isinstance(rf, dict) and rf.get("type") == "json_schema":
                json_schema_config = rf.get("json_schema", {})
                resp_params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema_config.get("name", "Output"),
                        "schema": json_schema_config.get("schema", {}),
                        "strict": json_schema_config.get("strict", True),
                    }
                }
            elif isinstance(rf, dict) and rf.get("type") == "json_object":
                resp_params["text"] = {"format": {"type": "json_object"}}

        # Convert Chat Completions tool format to Responses API format.
        # Chat Completions: {"type": "function", "function": {"name": ..., ...}}
        # Responses API:    {"type": "function", "name": ..., ...}
        if "tools" in resp_params:
            converted: List[Dict[str, Any]] = []
            for tool in resp_params["tools"]:
                if (
                    isinstance(tool, dict)
                    and tool.get("type") == "function"
                    and "function" in tool
                ):
                    converted.append({"type": "function", **tool["function"]})
                else:
                    converted.append(tool)
            resp_params["tools"] = converted

        # Convert tool_choice from Chat Completions dict format to Responses API format.
        if "tool_choice" in resp_params:
            tc = resp_params["tool_choice"]
            if (
                isinstance(tc, dict)
                and tc.get("type") == "function"
                and "function" in tc
            ):
                resp_params["tool_choice"] = {
                    "type": "function",
                    **tc["function"],
                }

        return resp_params
