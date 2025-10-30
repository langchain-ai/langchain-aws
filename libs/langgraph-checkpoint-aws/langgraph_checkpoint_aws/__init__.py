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
    from langgraph_checkpoint_aws.store import (
        AsyncValkeyStore,
        ValkeyIndexConfig,
        ValkeyStore,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        DocumentParsingError as ValkeyDocumentParsingError,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        EmbeddingGenerationError as ValkeyEmbeddingGenerationError,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        SearchIndexError as ValkeySearchIndexError,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        TTLConfigurationError as ValkeyTTLConfigurationError,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        ValidationError as ValkeyValidationError,
    )
    from langgraph_checkpoint_aws.store.valkey.exceptions import (
        ValkeyConnectionError,
        ValkeyStoreError,
    )

except ImportError:
    # If checkpoint dependencies are not available, create placeholder classes
    from typing import Any

    def _missing_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Valkey functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        )

    # Create placeholder classes that raise helpful errors
    AsyncValkeySaver: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    AsyncValkeyStore: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyCache: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeySaver: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyStore: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyIndexConfig: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyConnectionError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyDocumentParsingError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyEmbeddingGenerationError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeySearchIndexError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyStoreError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyTTLConfigurationError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyValidationError: type[Any] = _missing_checkpoint_dependencies_error  # type: ignore[assignment,no-redef]

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
    "AsyncValkeyStore",
    "SDK_USER_AGENT",
    "ValkeyConnectionError",
    "ValkeyDocumentParsingError",
    "ValkeyEmbeddingGenerationError",
    "ValkeyIndexConfig",
    "ValkeySearchIndexError",
    "ValkeyStore",
    "ValkeyStoreError",
    "ValkeyTTLConfigurationError",
    "ValkeyValidationError",
    "ValkeySaver",
    "ValkeyCache",
    "SDK_USER_AGENT",
]
