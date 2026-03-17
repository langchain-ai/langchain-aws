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


def _is_supported_model(model: Union[ChatBedrock, ChatBedrockConverse]) -> bool:
    """Check if the model supports prompt caching on Bedrock."""
    model_id = getattr(model, "model_id", "") or getattr(model, "model", "")
    return any(name in model_id.lower() for name in ("anthropic", "amazon.nova"))


class BedrockPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware for ChatBedrock and ChatBedrockConverse.

    Optimizes API usage by caching conversation prefixes for supported models
    on AWS Bedrock. Supports Anthropic Claude and Amazon Nova models.

    For ChatBedrock (InvokeModel API), adds ``cache_control`` to the last
    message's content block. For ChatBedrockConverse (Converse API), appends
    ``cachePoint`` blocks to the system prompt and last message.

    Requires both 'langchain' and 'langchain-aws' packages to be installed.

    Learn more about prompt caching at:
    - `Anthropic <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>`
    - `AWS Bedrock <https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html>`
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
            type: The type of cache to use. For ChatBedrock, only
                ``"ephemeral"`` is supported. For ChatBedrockConverse,
                this value is ignored as the Converse API always uses
                ``"default"`` cache type.
            ttl: The time to live for the cache, only ``"5m"`` and ``"1h"``
                are supported, default is``"5m"``.
            min_messages_to_cache: The minimum number of messages until the
                cache is used, default is 0.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used. ``"ignore"`` will ignore the
                unsupported model and continue without caching. ``"warn"``
                will warn the user and continue without caching. ``"raise"``
                will raise an error and stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Check if caching should be applied to the request."""
        if not isinstance(request.model, (ChatBedrock, ChatBedrockConverse)):
            err = (
                f"BedrockPromptCachingMiddleware only supports ChatBedrock and "
                f"ChatBedrockConverse, not {type(request.model).__name__}."
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(err)
            if self.unsupported_model_behavior == "warn":
                warn(err, stacklevel=3)
            return False

        if not _is_supported_model(request.model):
            model_id = getattr(request.model, "model_id", "unknown")
            err = (
                f"Prompt caching is only supported for Anthropic and "
                f"Amazon Nova models, got: {model_id}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(err)
            if self.unsupported_model_behavior == "warn":
                warn(err, stacklevel=3)
            return False

        messages_count = (
            len(request.messages) + 1
            if request.system_prompt
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def _get_cache_control_settings(self) -> dict:
        """Get cache control settings to pass via model_settings."""
        return {"type": self.type, "ttl": self.ttl}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Wrap the model call to add cache control via model_settings."""
        if self._should_apply_caching(request):
            # Pass cache_control through model_settings instead of modifying messages.
            # This prevents cache_control from accumulating in checkpoints across turns.
            # The model class applies this at API call time.
            new_model_settings = {
                **request.model_settings,
                "cache_control": self._get_cache_control_settings(),
            }
            return handler(request.override(model_settings=new_model_settings))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap the model call to add cache control via model_settings (async)."""
        if self._should_apply_caching(request):
            # Pass cache_control through model_settings instead of modifying messages.
            # This prevents cache_control from accumulating in checkpoints across turns.
            # The model class applies this at API call time.
            new_model_settings = {
                **request.model_settings,
                "cache_control": self._get_cache_control_settings(),
            }
            return await handler(request.override(model_settings=new_model_settings))
        return await handler(request)
