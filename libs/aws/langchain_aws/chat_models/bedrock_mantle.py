"""ChatBedrockMantle — LangChain chat model for Amazon Bedrock Mantle.

Wraps the OpenAI Python SDK to talk to Mantle's Chat Completions and Responses
APIs at ``bedrock-mantle.<region>.api.aws``.  Supports both long-term API keys
and auto-refreshing short-term API keys generated from AWS credentials.

Install with::

    pip install "langchain-aws[mantle]"
"""

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
    Optional,
    Sequence,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Keys that only exist in the Responses API — their presence forces that path.
_RESPONSES_ONLY_KEYS = frozenset(
    {
        "previous_response_id",
        "text",
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
    """Handle OpenAI SDK errors with enhanced messages for Mantle.

    Translates OpenAI SDK exceptions into actionable error messages
    specific to the Bedrock Mantle context.
    """
    try:
        from openai import (
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            RateLimitError,
        )
    except ImportError:
        raise error

    if isinstance(error, AuthenticationError):
        raise ValueError(
            "Bedrock Mantle authentication failed. Your API key or "
            "short-term token may be invalid or expired. If using "
            "short-term tokens, ensure your AWS credentials are valid. "
            "If using a static API key, verify it in the Bedrock Console. "
            f"Original error: {error}"
        ) from error

    if isinstance(error, RateLimitError):
        raise ValueError(
            "Bedrock Mantle rate limit exceeded. Consider reducing request "
            "frequency or requesting a quota increase via AWS Support. "
            f"Original error: {error}"
        ) from error

    if isinstance(error, NotFoundError):
        raise ValueError(
            "Bedrock Mantle model not found. Verify the model ID is correct "
            "and available in your region. Use list_models() to see available "
            f"models. Original error: {error}"
        ) from error

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
        raise ValueError(
            f"Bedrock Mantle bad request. Original error: {error}"
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
    """Extract content text and tool calls from a Responses API response."""
    content = ""
    tool_calls: list = []
    if response.output:
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        content += block.text
            elif hasattr(item, "name"):
                args = item.arguments if hasattr(item, "arguments") else ""
                tool_calls.append(
                    {
                        "name": item.name,
                        "args": _parse_responses_tool_args(args),
                        "id": (item.call_id if hasattr(item, "call_id") else item.id),
                        "type": "tool_call",
                    }
                )
    return content, tool_calls


def _get_last_messages_and_response_id(
    messages: Sequence[BaseMessage],
) -> tuple[list[BaseMessage], Optional[str]]:
    """Extract the most recent ``previous_response_id`` from message history.

    Scans backward for the last ``AIMessage`` whose ``id`` starts with
    ``resp_``.  Returns the messages after that point and the response ID.
    If no such message is found, returns all messages and ``None``.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            resp_id = getattr(msg, "id", None) or ""
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
            "status": response.status if hasattr(response, "status") else None,
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
    """If ``True``, automatically pass ``previous_response_id``.

    Uses the ID of the most recent ``AIMessage`` in the input message
    sequence. Messages up to that response are dropped from the request
    payload. Applies to OpenAI-compatible models only (Responses API).
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

            api_key = _BedrockApiKeyProvider(self._resolved_region)

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

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for LangSmith tracing."""
        ls_params = LangSmithParams(
            ls_provider="amazon_bedrock_mantle",
            ls_model_name=self.model,
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
        openai_msgs = convert_to_openai_messages(messages)
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
        openai_msgs = convert_to_openai_messages(messages)

        response = self._sync_client.responses.create(input=openai_msgs, **resp_params)
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
        openai_msgs = convert_to_openai_messages(messages)
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
        openai_msgs = convert_to_openai_messages(messages)

        response = await self._async_client.responses.create(
            input=openai_msgs, **resp_params
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
        openai_msgs = convert_to_openai_messages(messages)
        logger.debug("ChatBedrockMantle streaming (chat) with params: %s", params)

        try:
            stream = self._sync_client.chat.completions.create(
                messages=openai_msgs, **params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        for chunk in stream:
            result = self._process_chat_stream_chunk(chunk)
            if result is None:
                continue
            token, gen_chunk = result
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
        openai_msgs = convert_to_openai_messages(messages)
        logger.debug(
            "ChatBedrockMantle streaming (responses) with params: %s", resp_params
        )

        try:
            stream = self._sync_client.responses.create(
                input=openai_msgs, stream=True, **resp_params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        for event in stream:
            result = self._process_responses_stream_event(event)
            if result is None:
                continue
            token, gen_chunk = result
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
        openai_msgs = convert_to_openai_messages(messages)

        try:
            stream = await self._async_client.chat.completions.create(
                messages=openai_msgs, **params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        async for chunk in stream:
            result = self._process_chat_stream_chunk(chunk)
            if result is None:
                continue
            token, gen_chunk = result
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
        openai_msgs = convert_to_openai_messages(messages)

        try:
            stream = await self._async_client.responses.create(
                input=openai_msgs, stream=True, **resp_params
            )
        except Exception as e:
            _handle_openai_error(e)
            raise

        async for event in stream:
            result = self._process_responses_stream_event(event)
            if result is None:
                continue
            token, gen_chunk = result
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=gen_chunk)
            yield gen_chunk

    def _process_chat_stream_chunk(
        self, chunk: Any
    ) -> Optional[tuple[str, ChatGenerationChunk]]:
        """Convert a Chat Completions streaming chunk to a (token, chunk) pair."""
        if not chunk.choices:
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

        if hasattr(chunk, "usage") and chunk.usage:
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

        if event.type == "response.completed":
            response = event.response
            usage_metadata = None
            if hasattr(response, "usage") and response.usage:
                usage_metadata = UsageMetadata(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            msg_chunk = AIMessageChunk(
                content="",
                id=response.id,
                usage_metadata=usage_metadata,
                response_metadata={
                    "id": response.id,
                    "model": response.model,
                    "finish_reason": "stop",
                },
            )
            return "", ChatGenerationChunk(message=msg_chunk)

        return None

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[str | Dict] = None,
        **kwargs: Any,
    ) -> "ChatBedrockMantle":
        """Bind tool definitions for function calling."""
        formatted = [convert_to_openai_tool(t) for t in tools]
        bind_kwargs: Dict[str, Any] = {"tools": formatted, **kwargs}
        if tool_choice:
            bind_kwargs["tool_choice"] = tool_choice
        return super().bind(**bind_kwargs)  # type: ignore[return-value]

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
        from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
        from langchain_core.utils.pydantic import is_basemodel_subclass

        is_pydantic = isinstance(schema, type) and is_basemodel_subclass(schema)

        if method == "function_calling":
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm: Runnable = self.bind_tools([schema], tool_choice=tool_name, **kwargs)
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
            if self.use_responses_api is False:
                logger.warning(
                    "with_structured_output(method='json_schema') uses the "
                    "Responses API text.format parameter, but "
                    "use_responses_api=False is set. The text parameter "
                    "will be ignored by Chat Completions. Use "
                    "method='json_mode' or method='function_calling' instead."
                )
            if is_pydantic:
                json_schema = schema.model_json_schema()
            elif isinstance(schema, dict):
                json_schema = schema
            else:
                msg = (
                    f"Unsupported schema type for json_schema method: "
                    f"{type(schema)}. Use a Pydantic model or dict."
                )
                raise ValueError(msg)

            schema_name = json_schema.get(
                "title", getattr(schema, "__name__", "Output")
            )
            text_format = {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
            llm = self.bind(text={"format": text_format}, **kwargs)
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

        # Validate tool_choice — Mantle may only support "auto" or "none".
        if "tool_choice" in resp_params:
            tc = resp_params["tool_choice"]
            # Convert Chat Completions dict format to Responses API format
            if (
                isinstance(tc, dict)
                and tc.get("type") == "function"
                and "function" in tc
            ):
                resp_params["tool_choice"] = {
                    "type": "function",
                    **tc["function"],
                }
                tc = resp_params["tool_choice"]
            if isinstance(tc, str) and tc not in ("auto", "none"):
                logger.warning(
                    "Bedrock Mantle Responses API may only support "
                    "tool_choice='auto' or 'none'. Got %r — the API "
                    "may reject this.",
                    tc,
                )

        return resp_params
