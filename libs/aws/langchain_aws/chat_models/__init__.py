from typing import TYPE_CHECKING, Any

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse

if TYPE_CHECKING:
    from langchain_aws.chat_models.anthropic import ChatAnthropicBedrock

__all__ = [
    "ChatAnthropicBedrock",
    "ChatBedrock",
    "ChatBedrockConverse",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "ChatAnthropicBedrock":
        try:
            from langchain_aws.chat_models.anthropic import ChatAnthropicBedrock

            return ChatAnthropicBedrock
        except ImportError as e:
            msg = (
                f"Cannot import {name}. "
                'Please install it with `pip install "langchain-aws[anthropic]"`.'
            )
            raise ImportError(msg) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
