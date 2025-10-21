"""
LangGraph Checkpoint AWS - A LangChain checkpointer implementation using
Bedrock Session Management Service.
"""

from langgraph_checkpoint_aws.agentcore.saver import (
    AgentCoreMemorySaver,
)
from langgraph_checkpoint_aws.agentcore.store import (
    AgentCoreMemoryStore,
)

# Conditional imports for Valkey cache functionality
try:
    from langgraph_checkpoint_aws.cache.valkey import ValkeyCache

    valkey_available = True
except ImportError:
    # If Valkey dependencies are not available, create placeholder class
    from typing import Any

    def _missing_valkey_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Valkey cache functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        )

    # Create placeholder class that raises helpful error
    ValkeyCache: type[Any] = _missing_valkey_dependencies_error  # type: ignore[assignment,no-redef]
    valkey_available = False

__version__ = "0.2.0"
SDK_USER_AGENT = f"LangGraphCheckpointAWS#{__version__}"

# Expose the saver class at the package level
__all__ = [
    "AgentCoreMemorySaver",
    "AgentCoreMemoryStore",
    "ValkeyCache",
    "SDK_USER_AGENT",
    "valkey_available",
]
