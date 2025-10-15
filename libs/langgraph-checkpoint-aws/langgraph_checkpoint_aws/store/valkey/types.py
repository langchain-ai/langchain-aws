"""Enhanced type definitions for Valkey store."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

# Type aliases for embedding functions from LangGraph documentation
EmbeddingsFunc = Callable[[list[str]], list[list[float]]]
AEmbeddingsFunc = Callable[[list[str]], Any]  # Awaitable[list[list[float]]]

# Union type for embed field as per LangGraph IndexConfig
# Any represents Embeddings class
EmbedType = Any | EmbeddingsFunc | AEmbeddingsFunc | str


class ValkeyIndexConfigTyped(TypedDict):
    """Typed dictionary for Valkey index configuration with comprehensive type hints."""

    # Required fields
    collection_name: str
    dims: int
    embed: EmbedType  # Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    fields: list[str]

    # Optional fields with defaults
    timezone: NotRequired[str]  # Default: "UTC"
    index_type: NotRequired[Literal["hnsw", "flat"]]  # Default: "hnsw"
    distance_metric: NotRequired[Literal["COSINE", "L2", "IP"]]  # Default: "COSINE"
    algorithm: NotRequired[Literal["HNSW", "FLAT"]]  # Default: "HNSW"

    # HNSW specific parameters
    hnsw_m: NotRequired[int]  # Default: 16 - Number of connections per layer
    # Default: 200 - Search width during construction
    hnsw_ef_construction: NotRequired[int]
    hnsw_ef_runtime: NotRequired[int]  # Default: 10 - Search width during queries


class TTLConfigTyped(TypedDict):
    """Typed dictionary for TTL configuration."""

    default_ttl: NotRequired[float]  # TTL in minutes


class DocumentHashFields(TypedDict):
    """Typed dictionary for document hash fields stored in Valkey."""

    value: str  # JSON-encoded document value
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    vector: NotRequired[str]  # JSON-encoded vector data (optional)


class ParsedDocument(TypedDict):
    """Typed dictionary for parsed document structure."""

    value: dict[str, Any]  # Parsed document value
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    vector: NotRequired[list[float]]  # Vector data (optional)


class SearchResultMetadata(TypedDict):
    """Typed dictionary for search result metadata."""

    doc_id: str  # Document ID from search results
    score: float  # Similarity score
    namespace: tuple[str, ...]  # Parsed namespace
    key: str  # Document key


class ConnectionPoolConfig(TypedDict):
    """Typed dictionary for connection pool configuration."""

    min_size: NotRequired[int]  # Minimum pool size
    max_size: NotRequired[int]  # Maximum pool size
    timeout: NotRequired[float]  # Connection timeout in seconds


class ValkeyStoreConfig(TypedDict):
    """Comprehensive configuration for Valkey store."""

    # Connection configuration
    connection_string: NotRequired[str]  # Valkey connection string
    pool_config: NotRequired[ConnectionPoolConfig]  # Connection pool settings

    # Index configuration
    index: NotRequired[ValkeyIndexConfigTyped]  # Vector index configuration

    # TTL configuration
    ttl: NotRequired[TTLConfigTyped]  # TTL settings


class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents synchronously."""
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents asynchronously."""
        ...


class ValkeyResponseProtocol(Protocol):
    """Protocol for Valkey response types."""

    def __await__(self) -> Any:
        """Make response awaitable for async contexts."""
        ...


# Type aliases for better readability
ValkeyResponse = Any  # Can be actual result or ResponseT
NamespaceTuple = tuple[str, ...]
FilterDict = dict[str, Any]
HashFieldsDict = dict[str, str | bytes]
VectorData = list[float] | None

# Search strategy types
SearchStrategyType = Literal["vector", "hash", "key_pattern"]


class SearchStrategyResult(TypedDict):
    """Result from a search strategy execution."""

    strategy_used: SearchStrategyType
    items: list[Any]  # SearchItem objects
    execution_time: NotRequired[float]  # Execution time in seconds
    error: NotRequired[str]  # Error message if strategy failed


class ValkeyOperationResult(TypedDict):
    """Result from a Valkey operation with metadata."""

    success: bool
    operation: str  # Operation name (get, put, search, etc.)
    key: NotRequired[str]  # Key involved in operation
    namespace: NotRequired[NamespaceTuple]  # Namespace involved
    error: NotRequired[str]  # Error message if operation failed
    execution_time: NotRequired[float]  # Execution time in seconds


# Validation types
class ValidationResult(TypedDict):
    """Result from input validation."""

    valid: bool
    errors: list[str]  # List of validation error messages
    field_errors: NotRequired[dict[str, str]]  # Field-specific errors


# Index management types
class IndexStatus(TypedDict):
    """Status of a search index."""

    exists: bool
    name: str
    index_type: NotRequired[str]  # hnsw or flat
    document_count: NotRequired[int]  # Number of indexed documents
    vector_dims: NotRequired[int]  # Vector dimensions
    last_updated: NotRequired[str]  # ISO timestamp of last update


class IndexCreationParams(TypedDict):
    """Parameters for creating a search index."""

    index_name: str
    prefix: str  # Key prefix for indexing
    vector_dims: int
    index_type: Literal["hnsw", "flat"]
    distance_metric: Literal["COSINE", "L2", "IP"]
    hnsw_params: NotRequired[dict[str, int]]  # HNSW-specific parameters
    schema_fields: list[str]  # Additional schema fields to index
