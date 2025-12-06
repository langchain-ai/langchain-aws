"""Valkey store implementation."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, Literal

from langgraph.store.base import (
    NOT_PROVIDED,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    _ensure_ttl,
)
from langgraph.store.base.embed import get_text_at_path
from valkey import Valkey  # type: ignore[import-untyped]
from valkey.connection import ConnectionPool  # type: ignore[import-untyped]

# Import AsyncValkeyStore for convenience
from .async_store import AsyncValkeyStore
from .base import BaseValkeyStore, ValkeyIndexConfig
from .document_utils import DocumentProcessor
from .exceptions import EmbeddingGenerationError
from .search import SearchStrategyManager

logger = logging.getLogger(__name__)


class ValkeyStore(BaseValkeyStore):
    """Synchronous Valkey store implementation for LangGraph.

    Features:
        - Vector similarity search with configurable embeddings
        - JSON document storage with TTL support
        - Connection pool support for better performance
        - Namespace organization
        - Batch operations

    Examples:
        Basic usage with BedrockEmbeddings:

        ```python
        from langgraph_checkpoint_aws.store.valkey import ValkeyStore
        from langchain_aws import BedrockEmbeddings

        # Create BedrockEmbeddings instance
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name="us-east-1"
        )

        # Using connection string with BedrockEmbeddings
        with ValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            index={
                "collection_name": "my_documents",
                "dims": 1536,
                "embed": embeddings,
                "fields": ["text"],
                "timezone": "UTC",
                "index_type": "hnsw"
            },
            ttl={"default_ttl": 60.0},  # 1 hour TTL
            pool_size=10  # Connection pool size
        ) as store:
            # Use store...

        # Advanced HNSW config with Cohere embeddings
        cohere_embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name="us-east-1"
        )
        with ValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            index={
                "collection_name": "embeddings_store",
                "dims": 1024,
                "embed": cohere_embeddings,
                "fields": ["text", "title"],
                "timezone": "America/New_York",
                "index_type": "hnsw",
                "hnsw_m": 32,
                "hnsw_ef_construction": 400,
                "hnsw_ef_runtime": 20,
                "algorithm": "HNSW",
                "distance_metric": "COSINE"
            }
        ) as store:
            # Store with optimized HNSW parameters

        # Using connection pool with FLAT index and OpenAI embeddings
        pool = ConnectionPool(
            "valkey://localhost:6379",
            min_size=5,
            max_size=20,
            timeout=30
        )
        with ValkeyStore.from_pool(
            pool,
            index={
                "collection_name": "exact_search_store",
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
                "index_type": "flat",
                "algorithm": "FLAT",
                "distance_metric": "L2"
            }
        ) as store:
            # Use store with exact search...

        # Direct initialization with BedrockEmbeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name="us-east-1"
        )
        store = ValkeyStore(
            Valkey("valkey://localhost:6379"),
            index={
                "collection_name": "langgraph_store_idx",
                "dims": 1536,
                "embed": embeddings,
                "fields": ["text"]
            }
        )
        ```

    Thread Safety:
        The store is thread-safe for concurrent operations. Multiple threads can
        safely share a single store instance. The underlying Valkey client uses
        connection pooling to handle concurrent requests efficiently.

        For high-concurrency applications, configure appropriate pool size:

        ```python
        store = ValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            pool_size=20,  # Size based on expected concurrent threads
            ...
        )
        ```

        Note: If providing custom embeddings, ensure your embeddings object is
        thread-safe. Standard embedding providers (BedrockEmbeddings, OpenAI,
        Cohere) are thread-safe by default.

    Note:
        Semantic search is disabled by default. You can enable it by providing an
        `index` config when creating the store. Without this configuration, all
        `index` arguments passed to `put` or `aput` will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and
        indexes.
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
            index: Optional vector indexing configuration
            ttl: Optional TTL configuration
        """
        super().__init__(client, index=index, ttl=ttl)
        # Initialize search strategy manager
        self._search_manager = SearchStrategyManager(client, self)

    def setup(self) -> None:
        """Setup the store, including creating vector search index if configured."""
        if self.index and self.dims and self.embeddings:
            self._setup_search_index_sync()

    def _setup_search_index(self) -> None:
        """Setup vector search index for the store (sync version)."""
        return self._setup_search_index_sync()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        pool_size: int | None = None,
        pool_timeout: float | None = None,
    ) -> Generator[ValkeyStore, None, None]:
        """Create a ValkeyStore from a connection string.

        Args:
            conn_string: Valkey connection string (e.g. "valkey://localhost:6379")
            index: Optional vector indexing configuration
            ttl: Optional TTL configuration
            pool_size: Optional connection pool size
            pool_timeout: Optional pool timeout in seconds

        Example:
            ```python
            from langchain_aws import BedrockEmbeddings

            embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                region_name="us-east-1"
            )

            with ValkeyStore.from_conn_string(
                "valkey://localhost:6379",
                index={
                    "collection_name": "ml_reports",
                    "dims": 1536,
                    "embed": embeddings,
                    "fields": ["text"],
                    "timezone": "UTC",
                    "index_type": "hnsw",
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 200,
                    "hnsw_ef_runtime": 10
                },
                ttl={"default_ttl": 60.0},
                pool_size=10
            ) as store:
                # Store with BedrockEmbeddings vector indexing
                store.put(
                    ("docs", "user123"),
                    "report",
                    {
                        "text": "Machine learning report...",
                        "tags": ["ml", "report"]
                    }
                )
            ```
        """
        try:
            if pool_size:
                # Create connection pool
                pool = ConnectionPool.from_url(
                    url=conn_string,
                    max_connections=pool_size,
                    timeout=pool_timeout or 30.0,
                )
                client = Valkey.from_pool(pool)
            else:
                # Single connection
                client = Valkey.from_url(conn_string)

            # Set client info for library identification
            from ...checkpoint.valkey.utils import set_client_info

            set_client_info(client)

            store = cls(client, index=index, ttl=ttl)
            yield store
        finally:
            # Cleanup will be handled by pool/client
            pass

    @classmethod
    @contextmanager
    def from_pool(
        cls,
        pool: ConnectionPool,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Generator[ValkeyStore, None, None]:
        """Create a ValkeyStore from an existing connection pool.

        This allows reusing an existing pool across multiple stores or
        sharing a pool with other components.

        Args:
            pool: Existing Valkey connection pool
            index: Optional vector indexing configuration
            ttl: Optional TTL configuration

        Example:
            ```python
            from langchain_aws import BedrockEmbeddings

            # Create custom pool
            pool = ConnectionPool(
                "valkey://localhost:6379",
                min_size=5,
                max_size=20,
                timeout=30
            )

            # Create BedrockEmbeddings instance
            embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v2",
                region_name="us-west-2"
            )

            # Use pool with BedrockEmbeddings
            with ValkeyStore.from_pool(
                pool,
                index={
                    "collection_name": "shared_documents",
                    "dims": 1024,
                    "embed": embeddings,
                    "fields": ["text", "title"],
                    "timezone": "America/Los_Angeles",
                    "index_type": "hnsw",
                    "hnsw_m": 24,
                    "hnsw_ef_construction": 300,
                    "hnsw_ef_runtime": 15
                }
            ) as store:
                # Store with BedrockEmbeddings vector indexing
                store.put(
                    ("docs", "user123"),
                    "report",
                    {
                        "text": "Machine learning report...",
                        "tags": ["ml", "report"]
                    }
                )
            ```
        """
        try:
            client = Valkey.from_pool(connection_pool=pool)
            # Set client info for library identification
            from ...checkpoint.valkey.utils import set_client_info

            set_client_info(client)

            store = cls(client, index=index, ttl=ttl)
            yield store
        finally:
            # Pool cleanup handled by owner
            pass

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations synchronously."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                result = self._handle_get(op)
                results.append(result)
            elif isinstance(op, PutOp):
                self._handle_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                search_result = self._handle_search(op)
                results.append(search_result)  # type: ignore[arg-type]
            elif isinstance(op, ListNamespacesOp):
                list_result = self._handle_list(op)
                results.append(list_result)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use batch() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Get an item from the store asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use get() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Put an item in the store asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use put() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete an item from the store asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use delete() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search for items in the store asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use search() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List namespaces in the store asynchronously.

        Note:
            This async method is not supported by the ValkeyStore class.
            Use list_namespaces() instead, or consider using AsyncValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyStore does not support async methods. "
            "Consider using AsyncValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore\n"
            "See the documentation for more information."
        )

    def _handle_get(self, op: GetOp) -> Item | None:
        """Handle get operation."""
        try:
            key = self._build_key(op.namespace, op.key)
            result = self.client.hgetall(key)
            if not result:
                return None

            # Handle ResponseT type using base class method
            result = self._handle_response_t(result)
            if result is None:
                return None

            # Use shared core logic
            item = self._handle_get_core(op, result)

            # Refresh TTL if configured
            if item and op.refresh_ttl and self.ttl_config:
                ttl = self.ttl_config.get("default_ttl")
                if ttl:
                    self.client.expire(key, int(ttl * 60))

            return item
        except Exception as e:
            logger.error(f"Error in get operation: {e}")
            return None

    def _handle_put(self, op: PutOp) -> None:
        """Handle put operation."""
        # Use shared core logic from base class
        key, hash_fields = self._handle_put_core(op)

        if op.value is None:
            # Handle deletion
            try:
                self.client.delete(key)
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}")
            return

        # Generate embeddings synchronously if needed
        # Note: base class _handle_put_core doesn't generate embeddings,
        # so we need to do that here and update hash_fields
        if hash_fields and self.embeddings and op.index is not False:
            vector = self._generate_embeddings_sync(op)
            if vector:
                # Update hash_fields with vector
                import struct

                vector_bytes = struct.pack(f"{len(vector)}f", *vector)
                hash_fields["vector"] = vector_bytes

        try:
            # Use HSET to store as hash fields for vector search compatibility
            self.client.hset(key, mapping=hash_fields)

            # Set TTL if specified
            if op.ttl is not None:
                ttl_seconds = int(op.ttl * 60)  # Convert minutes to seconds
                self.client.expire(key, ttl_seconds)
        except Exception as e:
            logger.error(f"Error in put operation: {e}")
            raise

    def _generate_embeddings_sync(self, op: PutOp) -> list[float] | None:
        """Generate embeddings synchronously."""
        try:
            fields = op.index or self.index_fields
            if not fields:
                return None

            texts: list[str] = []
            for field in fields:
                field_value = get_text_at_path(op.value, field)
                if isinstance(field_value, list):
                    texts.extend(str(v) for v in field_value)
                elif field_value:
                    texts.append(str(field_value))

            if not texts:
                return None

            # Use sync embeddings method
            if hasattr(self.embeddings, "embed_documents"):
                vectors = self.embeddings.embed_documents(texts)
                return vectors[0] if vectors else None
            else:
                raise EmbeddingGenerationError(
                    "Cannot generate embeddings: embeddings object only has "
                    "async methods (aembed_documents). "
                    "Use AsyncValkeyStore for async embedding generation.",
                    text_content=" ".join(texts[:3]) if texts else None,
                )
        except EmbeddingGenerationError:
            # Re-raise EmbeddingGenerationError
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle search operation using search strategy pattern."""
        try:
            return self._search_manager.search(op)
        except Exception as e:
            logger.error(f"Error in search operation: {e}")
            return []

    def _convert_to_search_items(
        self, results: list[tuple[tuple[str, ...], str, float]]
    ) -> list[SearchItem]:
        """Convert hash search results to SearchItem objects."""
        items = []
        for namespace, key, score in results:
            try:
                # Get full document data using HGETALL since data is stored
                full_key = self._build_key(namespace, key)
                hash_data = self.client.hgetall(full_key)
                if not hash_data:
                    continue

                hash_data = self._handle_response_t(hash_data)
                if not hash_data or not isinstance(hash_data, dict):
                    continue

                # Use DocumentProcessor to convert hash fields back to document
                document = DocumentProcessor.convert_hash_to_document(hash_data)
                if document is None:
                    continue

                # Parse the JSON-encoded value using DocumentProcessor
                parsed_value = DocumentProcessor.parse_document_value(document)
                if parsed_value is None:
                    continue

                # Parse timestamps using DocumentProcessor
                created_at, updated_at = DocumentProcessor.parse_timestamps(document)

                items.append(
                    SearchItem(
                        namespace=namespace,
                        key=key,
                        value=parsed_value,
                        created_at=created_at,
                        updated_at=updated_at,
                        score=score,
                    )
                )
            except Exception as e:
                logger.debug(f"Error converting result {namespace}/{key}: {e}")
                continue
        return items

    def _refresh_ttl_for_items(self, items: list[SearchItem]) -> None:
        """Refresh TTL for a list of items."""
        if not self.ttl_config:
            return

        ttl_seconds = self.ttl_config.get("default_ttl")
        if ttl_seconds:
            for item in items:
                item_key = self._build_key(item.namespace, item.key)
                try:
                    self.client.expire(item_key, int(ttl_seconds * 60))
                except Exception as e:
                    logger.error(f"Error refreshing TTL for {item_key}: {e}")

    def _handle_list(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle list namespaces operation."""
        try:
            # Build patterns based on match conditions with langgraph: prefix
            patterns = ["langgraph:*"]  # Default pattern with prefix

            if op.match_conditions:
                patterns = []
                for condition in op.match_conditions:
                    path = "/".join(str(p) for p in condition.path)
                    if condition.match_type == "prefix":
                        patterns.append(f"langgraph:{path}*")
                    else:  # suffix
                        patterns.append(f"langgraph:*{path}")

            # Collect all keys from patterns
            all_keys = []
            for pattern in patterns:
                try:
                    keys_result = self.client.keys(pattern)
                    # Use base class method to safely parse keys
                    keys = self._safe_parse_keys(keys_result)
                    all_keys.extend(keys)
                except Exception as e:
                    logger.error(f"Error scanning pattern {pattern}: {e}")
                    continue

            # Remove langgraph: prefix from keys before extracting namespaces
            cleaned_keys = []
            for key in all_keys:
                if key.startswith("langgraph:"):
                    cleaned_keys.append(key[10:])  # Remove "langgraph:" prefix
                else:
                    cleaned_keys.append(key)

            # Extract namespaces using base class method
            # Type ignore: list[str] is compatible with list[bytes | str] at runtime
            namespaces = self._extract_namespaces_from_keys(cleaned_keys, op.max_depth)  # type: ignore[arg-type]

            # Convert to sorted list and apply pagination
            namespace_list = sorted(list(namespaces))

            # Apply offset and limit
            start_idx = op.offset or 0
            end_idx = start_idx + (op.limit or len(namespace_list))

            return namespace_list[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return []

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Get an item from the store."""
        from langgraph.store.base import GetOp

        op = GetOp(namespace=namespace, key=key, refresh_ttl=refresh_ttl or False)
        return self._handle_get(op)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Put an item in the store."""
        from langgraph.store.base import PutOp

        resolved_ttl = _ensure_ttl(self.ttl_config, ttl)
        op = PutOp(
            namespace=namespace, key=key, value=value, index=index, ttl=resolved_ttl
        )
        self._handle_put(op)

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete an item from the store."""
        from langgraph.store.base import PutOp

        op = PutOp(namespace=namespace, key=key, value=None, ttl=None)
        self._handle_put(op)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search for items in the store."""
        from langgraph.store.base import SearchOp

        op = SearchOp(
            namespace_prefix=namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl or False,
        )
        return self._handle_search(op)

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List namespaces in the store."""
        from langgraph.store.base import ListNamespacesOp

        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(path=prefix, match_type="prefix"))
        if suffix:
            match_conditions.append(MatchCondition(path=suffix, match_type="suffix"))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions) if match_conditions else None,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        return self._handle_list(op)


__all__ = ["ValkeyStore", "AsyncValkeyStore"]
