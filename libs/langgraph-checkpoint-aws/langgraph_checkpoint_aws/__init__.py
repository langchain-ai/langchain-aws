"""
LangGraph Checkpoint AWS - A LangChain checkpointer implementation using
Bedrock Session Management Service and Valkey.
"""

from importlib.metadata import version

from langgraph_checkpoint_aws.agentcore.saver import (
    AgentCoreMemorySaver,
)
from langgraph_checkpoint_aws.agentcore.store import (
    AgentCoreMemoryStore,
)

# Conditional imports for Valkey functionality
try:
    from langgraph_checkpoint_aws.cache import ValkeyCache
    from langgraph_checkpoint_aws.checkpoint import AsyncValkeySaver, ValkeySaver

    valkey_available = True
except ImportError:
    # If checkpoint dependencies are not available, create placeholder classes
    from typing import Any

    def _missing_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Valkey functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        )

    # Create placeholder classes that raise helpful errors
    AsyncValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyCache: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    valkey_available = False

try:
    __version__ = version("langgraph-checkpoint-aws")
except Exception:
    # Fallback version if package is not installed
    __version__ = "1.0.0"
SDK_USER_AGENT = f"LangGraphCheckpointAWS#{__version__}"

# Expose the saver class at the package level
__all__ = [
    "AgentCoreMemorySaver",
    "AgentCoreMemoryStore",
    "AsyncValkeySaver",
    "ValkeyCache",
    "ValkeySaver",
    "SDK_USER_AGENT",
    "valkey_available",
]
