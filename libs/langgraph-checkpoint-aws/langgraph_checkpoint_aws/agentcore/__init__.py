from typing import Any

from langgraph_checkpoint_aws.agentcore.saver import AgentCoreMemorySaver
from langgraph_checkpoint_aws.agentcore.store import AgentCoreMemoryStore

# Store the import error for later use
_import_error: ImportError | None = None

# Conditional imports for optional dependencies
try:
    from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

    valkey_available = True
except ImportError as e:
    # Store the error for later use
    _import_error = e
    valkey_available = False

    # If dependencies are not available, provide helpful error message
    def _missing_valkey_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "AgentCore Valkey functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        ) from _import_error

    # Create placeholder class that raises helpful error
    AgentCoreValkeySaver: type[Any] = _missing_valkey_dependencies_error  # type: ignore[assignment,no-redef]

__all__ = [
    "AgentCoreMemorySaver",
    "AgentCoreMemoryStore",
    "AgentCoreValkeySaver",
    "valkey_available",
]
