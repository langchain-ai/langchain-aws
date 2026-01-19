"""Bedrock Anthropic prompt caching middleware.

Requires:
    - langchain: For agent middleware framework
    - langchain-aws: For ChatBedrock/ChatBedrockConverse models (already a dependency)
"""

from collections.abc import Awaitable, Callable
from typing import Literal, Union
from warnings import warn

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "Prompt caching middleware requires 'langchain' to be installed. "
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


def _is_anthropic_model(model: Union[ChatBedrock, ChatBedrockConverse]) -> bool:
    """Check if the model is an Anthropic model on Bedrock."""
    model_id = getattr(model, "model_id", "") or getattr(model, "model", "")
    return "anthropic" in model_id.lower()


class BedrockPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware for ChatBedrock (InvokeModel API).

    Optimizes API usage by caching conversation prefixes for Anthropic models
    on AWS Bedrock. Adds cache_control to the last message, caching the entire
    conversation prefix (including system prompt) for improved cache hits in
    multi-turn conversations.

    Requires both 'langchain' and 'langchain-aws' packages to be installed.

    Learn more about Anthropic prompt caching
    [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
    """

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",  # noqa: A002
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize the middleware with cache control settings.

        Args:
            type: The type of cache to use, only "ephemeral" is supported.
            ttl: The time to live for the cache, only "5m" and "1h" are
                supported.
            min_messages_to_cache: The minimum number of messages until the
                cache is used, default is 0.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used. "ignore" will ignore the unsupported
                model and continue without caching. "warn" will warn the user
                and continue without caching. "raise" will raise an error and
                stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Check if caching should be applied to the request."""
        if not isinstance(request.model, ChatBedrock):
            msg = (
                f"BedrockPromptCachingMiddleware only supports ChatBedrock, "
                f"not {type(request.model).__name__}. "
                f"For ChatBedrockConverse, use BedrockConversePromptCachingMiddleware."
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        if not _is_anthropic_model(request.model):
            model_id = getattr(request.model, "model_id", "unknown")
            msg = f"Prompt caching only supported for Anthropic models: {model_id}"
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        messages_count = (
            len(request.messages) + 1
            if request.system_prompt
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def _apply_cache_control(self, request: ModelRequest) -> None:
        """Apply cache control to the last message."""
        if not request.messages:
            return

        cache_control = {"type": self.type, "ttl": self.ttl}
        last_msg = request.messages[-1]
        content = getattr(last_msg, "content", None)

        if isinstance(content, str):
            new_content = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ]
            request.messages[-1] = last_msg.model_copy(update={"content": new_content})
        elif isinstance(content, list) and content:
            new_content = list(content)
            last_block = new_content[-1]
            if isinstance(last_block, dict):
                if "cache_control" not in last_block:
                    new_block = dict(last_block)
                    new_block["cache_control"] = cache_control
                    new_content[-1] = new_block
            elif isinstance(last_block, str):
                new_content[-1] = {
                    "type": "text",
                    "text": last_block,
                    "cache_control": cache_control,
                }
            request.messages[-1] = last_msg.model_copy(update={"content": new_content})

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Wrap the model call to add cache control."""
        if self._should_apply_caching(request):
            self._apply_cache_control(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap the model call to add cache control (async version)."""
        if self._should_apply_caching(request):
            self._apply_cache_control(request)
        return await handler(request)


class BedrockConversePromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware for ChatBedrockConverse (Converse API).

    Optimizes API usage by caching conversation prefixes for Anthropic models
    on AWS Bedrock via the Converse API.

    Requires both 'langchain' and 'langchain-aws' packages to be installed.

    Learn more about AWS Bedrock prompt caching
    [here](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).
    """

    def __init__(
        self,
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize the middleware with cache settings.

        Args:
            min_messages_to_cache: The minimum number of messages until the
                cache is used, default is 0.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used. "ignore" will ignore the unsupported
                model and continue without caching. "warn" will warn the user
                and continue without caching. "raise" will raise an error and
                stop the agent.
        """
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Check if caching should be applied to the request."""
        if not isinstance(request.model, ChatBedrockConverse):
            msg = (
                "BedrockConversePromptCachingMiddleware only supports "
                f"ChatBedrockConverse, not {type(request.model).__name__}. "
                "For ChatBedrock, use BedrockPromptCachingMiddleware."
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        if not _is_anthropic_model(request.model):
            model_id = getattr(request.model, "model", "unknown")
            msg = f"Prompt caching only supported for Anthropic models: {model_id}"
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        messages_count = (
            len(request.messages) + 1
            if request.system_prompt
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def _apply_cache_point(self, request: ModelRequest) -> None:
        """Apply cachePoint to the system prompt."""
        cache_point = {"cachePoint": {"type": "default"}}

        if isinstance(request.system_prompt, str) and request.system_prompt.strip():
            request.system_prompt = [
                {"text": request.system_prompt},
                cache_point,
            ]
        elif isinstance(request.system_prompt, list):
            has_cache_point = any(
                isinstance(item, dict) and "cachePoint" in item
                for item in request.system_prompt
            )
            if not has_cache_point:
                request.system_prompt = list(request.system_prompt) + [cache_point]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Wrap the model call to add cache point."""
        if self._should_apply_caching(request):
            self._apply_cache_point(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap the model call to add cache point (async version)."""
        if self._should_apply_caching(request):
            self._apply_cache_point(request)
        return await handler(request)
