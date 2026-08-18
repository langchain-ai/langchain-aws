"""
LangGraph Checkpoint AWS - A LangChain checkpointer implementation using
Bedrock Session Management Service and Valkey.
"""

from typing import Any

from langgraph_checkpoint_aws.checkpoint.agentcore.saver import (
    AgentCoreMemorySaver,
)
from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.async_saver import (
    AsyncBedrockSessionSaver,
)
from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.async_session import (
    AsyncBedrockAgentRuntimeSessionClient,
)
from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.saver import (
    BedrockSessionSaver,
)
from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.session import (
    BedrockAgentRuntimeSessionClient,
)
from langgraph_checkpoint_aws.checkpoint.deferred_saver import (
    AsyncBatchFlushable,
    DeferredCheckpointSaver,
    PendingWrite,
    SyncBatchFlushable,
)
from langgraph_checkpoint_aws.checkpoint.dynamodb import (
    DynamoDBSaver,
)
from langgraph_checkpoint_aws.store.agentcore.store import (
    AgentCoreMemoryStore,
)
from langgraph_checkpoint_aws.store.dynamodb import (
    DynamoDBStore,
)

# Conditional imports for Valkey functionality
try:
    from langgraph_checkpoint_aws.cache import ValkeyCache
    from langgraph_checkpoint_aws.checkpoint import (
        AgentCoreValkeySaver,
        AsyncValkeySaver,
        ValkeySaver,
    )
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

    valkey_available = True
except ImportError as e:
    # Store the import error for better debugging
    _import_error = e

    from typing import Any

    def _missing_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Valkey functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        ) from _import_error

    # Create placeholder classes that raise helpful errors
    AgentCoreValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    AsyncValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    AsyncValkeyStore: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyCache: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyStore: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyIndexConfig: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyConnectionError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyDocumentParsingError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyEmbeddingGenerationError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeySearchIndexError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyStoreError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyTTLConfigurationError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    ValkeyValidationError: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]

    valkey_available = False

from langgraph_checkpoint_aws._version import (  # noqa: E402, F401
    SDK_USER_AGENT,
    __version__,
)

# Expose the saver class at the package level
__all__ = [
    "AgentCoreMemorySaver",
    "AgentCoreMemoryStore",
    "AgentCoreValkeySaver",
    "AsyncBatchFlushable",
    "AsyncBedrockAgentRuntimeSessionClient",
    "AsyncBedrockSessionSaver",
    "AsyncValkeySaver",
    "AsyncValkeyStore",
    "BedrockAgentRuntimeSessionClient",
    "BedrockSessionSaver",
    "DeferredCheckpointSaver",
    "DynamoDBSaver",
    "DynamoDBStore",
    "PendingWrite",
    "SDK_USER_AGENT",
    "SyncBatchFlushable",
    "ValkeyCache",
    "ValkeyConnectionError",
    "ValkeyDocumentParsingError",
    "ValkeyEmbeddingGenerationError",
    "ValkeyIndexConfig",
    "ValkeySaver",
    "ValkeySearchIndexError",
    "ValkeyStore",
    "ValkeyStoreError",
    "ValkeyTTLConfigurationError",
    "ValkeyValidationError",
    "valkey_available",
]
