"""Base Valkey store implementation with common functionality."""

from __future__ import annotations

import logging
from typing import Any, NotRequired

import orjson
from langgraph.store.base import (
    BaseStore,
    IndexConfig,
    TTLConfig,
)
from langgraph.store.base.embed import ensure_embeddings
from valkey import Valkey
from valkey.connection import ConnectionPool

from ...checkpoint.valkey.utils import set_client_info
from .constants import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TIMEZONE,
    DEFAULT_INDEX_TYPE,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_HNSW_M,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_RUNTIME,
)
from .document_utils import DocumentProcessor, FilterProcessor, ScoreCalculator
from .exceptions import (
    ValkeyStoreError,
    ValkeyConnectionError,
    DocumentParsingError,
    SearchIndexError,
    EmbeddingGenerationError,
    ValidationError,
    TTLConfigurationError,
)
from .search_strategies import SearchStrategyManager
logger = logging.getLogger(__name__)


class ValkeyIndexConfig(IndexConfig):
    """Valkey-specific index configuration extending IndexConfig.

    Includes all fields from IndexConfig plus additional Valkey-specific
    configuration options for vector search and indexing.
    """

    collection_name: str
    timezone: NotRequired[str]  # Default: "UTC"
    index_type: NotRequired[str]  # Default: "hnsw", can be 'hnsw' or 'flat'
    distance_metric: NotRequired[str]  # Default: "COSINE", supports L2, IP, COSINE
    # HNSW specific parameters with recommended defaults
    hnsw_m: NotRequired[int]  # Default: 16 - Number of connections per layer
    hnsw_ef_construction: NotRequired[
        int
    ]  # Default: 200 - Search width during construction
    hnsw_ef_runtime: NotRequired[int]  # Default: 10 - Search width during queries


class BaseValkeyStore(BaseStore):
    """Base Valkey store implementation with common functionality.

    This class contains all the shared functionality between the sync and async
    Valkey store implementations, including initialization, configuration,
    search index setup, and utility methods.
    """

    supports_ttl = True

    def __init__(
        self,
        client: Valkey,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        """Initialize Valkey store.

        Args:
            client: Valkey client instance
            index: Optional Valkey-specific vector indexing configuration
            ttl: Optional TTL configuration
        """
        self.client = client
        self.ttl_config = ttl
        self.index = index
        self._search_available = None  # Cache search availability check

        # Initialize architectural components
        self._document_processor = DocumentProcessor()
        self._filter_processor = FilterProcessor()
        self._score_calculator = ScoreCalculator()
        
        if index:
            embed_config = index.get("embed")
            if embed_config:
                try:
                    self.embeddings = ensure_embeddings(embed_config)
                except Exception as e:
                    raise EmbeddingGenerationError(f"Failed to initialize embeddings: {e}") from e
            else:
                self.embeddings = None
            self.index_fields = index.get("fields", ["$"])
            self.dims = index.get("dims")

            # Initialize Valkey-specific configuration with defaults from constants
            self.collection_name = index.get("collection_name", DEFAULT_COLLECTION_NAME)
            self.timezone = index.get("timezone", DEFAULT_TIMEZONE)
            self.index_type = index.get("index_type", DEFAULT_INDEX_TYPE)
            self.distance_metric = index.get("distance_metric", DEFAULT_DISTANCE_METRIC)
            self.hnsw_m = index.get("hnsw_m", DEFAULT_HNSW_M)
            self.hnsw_ef_construction = index.get("hnsw_ef_construction", DEFAULT_HNSW_EF_CONSTRUCTION)
            self.hnsw_ef_runtime = index.get("hnsw_ef_runtime", DEFAULT_HNSW_EF_RUNTIME)
        else:
            self.embeddings = None
            self.index_fields = None
            self.dims = None
            # Set default values when no index config is provided
            self.collection_name = DEFAULT_COLLECTION_NAME
            self.timezone = DEFAULT_TIMEZONE
            self.index_type = DEFAULT_INDEX_TYPE
            self.distance_metric = DEFAULT_DISTANCE_METRIC
            self.hnsw_m = DEFAULT_HNSW_M
            self.hnsw_ef_construction = DEFAULT_HNSW_EF_CONSTRUCTION
            self.hnsw_ef_runtime = DEFAULT_HNSW_EF_RUNTIME

        # Initialize search strategy manager (will be set up by subclasses)
        self._search_strategy_manager = None

        # Set client info for identification
        # Check if this is an async client by looking for async methods
        if hasattr(client, "aclose") or hasattr(client, "__aenter__"):
            # This is likely an async client, skip sync set_client_info
            pass
        else:
            # This is a sync client, safe to call set_client_info
            try:
                set_client_info(client)
            except Exception as e:
                raise ValkeyConnectionError(f"Failed to set client info: {e}") from e

    def _is_search_available(self) -> bool:
        """Check if Valkey Search module is available."""
        if self._search_available is not None:
            return self._search_available

        try:
            # Try to execute a simple FT.INFO command to check if search is available
            self.client.execute_command("FT._LIST")
            self._search_available = True
            return True
        except Exception as e:
            logger.debug(f"Valkey Search not available: {e}")
            self._search_available = False
            return False

    def _create_index_command(self, index_name: str, prefix: str) -> list[str]:
        """Create the FT.CREATE command for vector search index."""
        # Valkey requires at least one vector field for search indices
        # Use default dimensions if not specified
        dims = self.dims or 128

        # Build the vector field configuration based on index type
        if self.index_type == "hnsw":
            vector_config = [
                "vector",
                "VECTOR",
                "HNSW",
                "12",  # Attribute count: TYPE, FLOAT32, DIM, dims, DISTANCE_METRIC, metric, M, m, EF_CONSTRUCTION, ef_construction, EF_RUNTIME, ef_runtime
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dims),
                "DISTANCE_METRIC",
                self.distance_metric,
                "M",
                str(self.hnsw_m),
                "EF_CONSTRUCTION",
                str(self.hnsw_ef_construction),
                "EF_RUNTIME",
                str(self.hnsw_ef_runtime),
            ]
        elif self.index_type == "flat":
            vector_config = [
                "vector",
                "VECTOR",
                "FLAT",
                "6",  # Attribute count: TYPE, FLOAT32, DIM, dims, DISTANCE_METRIC, metric
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dims),
                "DISTANCE_METRIC",
                self.distance_metric,
            ]
        else:
            # Fallback to HNSW if invalid type
            vector_config = [
                "vector",
                "VECTOR",
                "HNSW",
                "12",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dims),
                "DISTANCE_METRIC",
                self.distance_metric,
                "M",
                str(self.hnsw_m),
                "EF_CONSTRUCTION",
                str(self.hnsw_ef_construction),
                "EF_RUNTIME",
                str(self.hnsw_ef_runtime),
            ]

        # Build schema fields - start with basic fields
        schema_fields = [
            "namespace",
            "TAG",
            "key", 
            "TAG",
            "value",
            "TAG",
        ]

        # Add configured searchable fields from index configuration
        if self.index_fields and self.index_fields != ["$"]:
            for field in self.index_fields:
                if field != "$":  # Skip root field marker
                    # Add each field as a searchable TAG field
                    # Store the field directly without "value_" prefix for proper indexing
                    schema_fields.extend([field, "TAG"])

        # Build the complete command
        return [
            "FT.CREATE",
            index_name,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            f"{prefix}:",
            "SCHEMA",
        ] + schema_fields + vector_config

    def _execute_command(self, *args) -> Any:
        """Execute a command on the Valkey client."""
        return self.client.execute_command(*args)

    def _setup_search_index_sync(self) -> None:
        """Setup vector search index for the store (synchronous version)."""
        if not self._is_search_available():
            logger.warning(
                "Valkey Search module not available, vector search will be disabled"
            )
            return

        try:
            # Use collection_name for index_name, fallback to default if not available
            index_name = self.collection_name or "langgraph_store_idx"

            # Check if index already exists
            try:
                self._execute_command("FT.INFO", index_name)
                logger.debug(f"Search index {index_name} already exists")
                return
            except Exception:
                # Index doesn't exist, create it
                pass

            # Create the index using raw FT.CREATE command with correct syntax
            cmd = self._create_index_command(index_name, "langgraph")
            self._execute_command(*cmd)
            logger.info(f"Created search index {index_name} with TAG fields")

        except Exception as e:
            logger.error(f"Failed to setup search index: {e}")
            # Don't raise the error, just log it and continue without search


    def _validate_put_operation(
        self, namespace: tuple[str, ...], value: dict[str, Any] | None
    ) -> None:
        """Validate put operation parameters."""
        try:
            # Validate namespace
            if not namespace:
                raise ValidationError("Namespace cannot be empty")

            # Validate value type
            if value is not None and not isinstance(value, dict):
                raise ValidationError("Value must be a dictionary or None")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Validation failed: {e}") from e


    def _build_key(self, namespace: tuple[str, ...], key: str) -> str:
        """Build a storage key from namespace and key."""
        return "langgraph:" + "/".join(namespace + (key,))

    def _parse_key(self, key: str, prefix: str = "") -> tuple[tuple[str, ...], str]:
        """Parse a storage key into namespace and key components."""
        key_suffix = key
        if prefix and key.startswith(prefix):
            key_suffix = key[len(prefix) :].lstrip("/")

        parts = key_suffix.split("/")
        if len(parts) >= 1:
            namespace = tuple(parts[:-1]) if len(parts) > 1 else tuple()
            item_key = parts[-1]
            return namespace, item_key
        else:
            return tuple(), key_suffix

    def _calculate_simple_score(
        self, query: str | None, value: dict[str, Any]
    ) -> float:
        """Calculate a simple text-based relevance score."""
        return self._score_calculator.calculate_text_similarity_score(query, value)


    def _apply_filter(
        self, value: dict[str, Any], filter_dict: dict[str, Any] | None
    ) -> bool:
        """Apply filter conditions to a value."""
        return self._filter_processor.apply_filters(value, filter_dict)

    def _extract_namespaces_from_keys(
        self, keys: list[bytes | str], max_depth: int | None = None
    ) -> set[tuple[str, ...]]:
        """Extract unique namespaces from a list of keys."""
        namespaces = set()

        for key_bytes in keys:
            # Convert bytes key to string
            if isinstance(key_bytes, bytes):
                key = key_bytes.decode("utf-8")
            elif isinstance(key_bytes, str):
                key = key_bytes
            else:
                # Handle other types by converting to string
                key = str(key_bytes)

            # Extract namespace from key
            parts = key.split("/")

            # Apply max_depth if specified
            if max_depth and len(parts) > max_depth:
                parts = parts[:max_depth]

            # Create namespace tuple (excluding the last part which is the key)
            if len(parts) > 1:
                namespace = tuple(parts[:-1])
                namespaces.add(namespace)
            elif len(parts) == 1:
                # Single level key, add empty namespace
                namespaces.add(tuple())

        return namespaces

    def _handle_response_t(self, result: Any) -> Any:
        """Handle ResponseT type from Valkey client.

        ResponseT can be either the actual result or an awaitable.
        In sync context, we should only get the actual result.
        """
        if hasattr(result, "__await__"):
            # This shouldn't happen in sync context, but handle it gracefully
            logger.error(f"Received awaitable result in sync context: {type(result)}")
            return None
        return result

    def _safe_parse_keys(self, keys_result: Any) -> list[str]:
        """Safely parse keys result handling ResponseT type."""
        # Handle ResponseT type - ensure we get the actual result
        keys_result = self._handle_response_t(keys_result)
        if keys_result is None:
            return []

        if hasattr(keys_result, "__iter__") and not isinstance(
            keys_result, (str, bytes)
        ):
            # Convert all keys to strings
            result = []
            for key in keys_result:
                if isinstance(key, bytes):
                    result.append(key.decode("utf-8"))
                elif isinstance(key, str):
                    result.append(key)
                else:
                    result.append(str(key))
            return result
        else:
            return []
