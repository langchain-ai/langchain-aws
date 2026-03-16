from typing import TYPE_CHECKING, Any

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse

if TYPE_CHECKING:
    from langchain_aws.chat_models.anthropic import ChatAnthropicBedrock
    from langchain_aws.chat_models.bedrock_nova_sonic import ChatBedrockNovaSonic

__all__ = [
    "ChatAnthropicBedrock",
    "ChatBedrock",
    "ChatBedrockConverse",
    "ChatBedrockNovaSonic",
]

# Mapping of class name to (module path, install hint) for chat models that
# depend on optional extras.  ``__getattr__`` uses this to defer the import
# until the class is actually accessed, so the base package stays lightweight.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ChatAnthropicBedrock": (
        "langchain_aws.chat_models.anthropic",
        'pip install "langchain-aws[anthropic]"',
    ),
    "ChatBedrockNovaSonic": (
        "langchain_aws.chat_models.bedrock_nova_sonic",
        'pip install "langchain-aws[nova-sonic]"',
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name in _LAZY_IMPORTS:
        module_path, install_hint = _LAZY_IMPORTS[name]
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return getattr(mod, name)
        except ImportError as e:
            msg = f"Cannot import {name}. Please install it with `{install_hint}`."
            raise ImportError(msg) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
