"""Async Valkey store implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
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
from valkey import Valkey
from valkey.connection import ConnectionPool

from ...checkpoint.valkey.utils import aset_client_info
from .base import BaseValkeyStore, ValkeyIndexConfig
from .document_utils import DocumentProcessor, FilterProcessor, ScoreCalculator
from .exceptions import EmbeddingGenerationError
from .search import AsyncSearchStrategyManager

logger = logging.getLogger(__name__)


class AsyncValkeyStore(BaseValkeyStore):
    """Asynchronous Valkey store implementation for LangGraph.

    Features:
        - Vector similarity search with configurable embeddings
        - JSON document storage with TTL support
        - Connection pool support for better performance
        - Async operations
        - Namespace organization
        - Batch operations

    Examples:
        Basic usage with BedrockEmbeddings:

        ```python
        from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore
        from langchain_aws import BedrockEmbeddings

        # Create BedrockEmbeddings instance
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name="us-east-1"
        )

        # Using connection string with BedrockEmbeddings
        async with AsyncValkeyStore.from_conn_string(
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

        # Advanced HNSW configuration with Cohere embeddings
        cohere_embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name="us-east-1"
        )
        async with AsyncValkeyStore.from_conn_string(
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
        async with AsyncValkeyStore.from_pool(
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
        store = AsyncValkeyStore(
            Valkey("valkey://localhost:6379"),
            index={
                "collection_name": "langgraph_store_idx",
                "dims": 1536,
                "embed": embeddings,
                "fields": ["text"]
            }
        )
        ```

    Concurrency:
        AsyncValkeyStore is designed for asyncio concurrency (cooperative multitasking).
        Multiple coroutines can safely share a single store instance within the same
        event loop. The underlying async Valkey client handles concurrent requests
        efficiently.

        For high-concurrency applications, configure appropriate pool size:

        ```python
        store = await AsyncValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            pool_size=20,  # Size based on expected concurrent operations
            ...
        )
        ```

        Note: If providing custom embeddings, ensure your embeddings object supports
        async operations. Standard embedding providers (BedrockEmbeddings with async
        methods, OpenAI, Cohere) support async operations by default.

    Note:
        Semantic search is disabled by default. You can enable it by providing
        an `index` configuration when creating the store. Without this configuration,
        all `index` arguments passed to
        `put` or `aput` will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary
        tables and indexes.
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
        super().__init__(client, index=index, ttl=ttl)

        # Initialize architectural components
        self._document_processor = DocumentProcessor()
        self._filter_processor = FilterProcessor()
        self._score_calculator = ScoreCalculator()
        self._search_manager = AsyncSearchStrategyManager(self)

        # Detect if this is an async client (like fakeredis.aioredis.FakeRedis)
        self._is_async_client = self._detect_async_client(client)

    def _detect_async_client(self, client: Any) -> bool:
        """Detect if the client is an async client."""
        # Check for async methods that are specific to async Redis clients
        return (
            hasattr(client, "aclose")
            or hasattr(client, "__aenter__")
            or "aioredis" in str(type(client))
            or "FakeRedis" in str(type(client))
            and hasattr(client, "hgetall")
            and callable(getattr(client, "hgetall", None))
            and asyncio.iscoroutinefunction(client.hgetall)
        )

    async def _execute_client_method(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a client method, handling both sync and async clients."""
        method = getattr(self.client, method_name)

        if self._is_async_client:
            # For async clients, call the method directly
            return await method(*args, **kwargs)
        else:
            # For sync clients, use run_in_executor
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: method(*args, **kwargs)
            )

    async def _execute_command(self, *args) -> Any:
        """Execute a command on the async Valkey client."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.client.execute_command, *args
        )

    async def setup(self) -> None:
        """Setup the store, including creating vector search index if configured."""
        if self.index:
            await self._setup_search_index_async()

    async def _is_search_available_async(self) -> bool:
        """Check if Valkey Search module is available (async version)."""
        if self._search_available is not None:
            return self._search_available

        try:
            # Try to execute a simple FT.INFO command to check if search is available
            await self._execute_command("FT._LIST")
            self._search_available = True  # type: ignore[assignment]
            return True
        except Exception as e:
            logger.debug(f"Valkey Search not available: {e}")
            self._search_available = False  # type: ignore[assignment]
            return False

    async def _setup_search_index_async(self) -> None:
        """Setup vector search index for the store (asynchronous version)."""
        if not await self._is_search_available_async():
            logger.warning(
                "Valkey Search module not available, vector search will be disabled"
            )
            return

        try:
            # Use collection_name for index_name, fallback to default if not available
            index_name = self.collection_name or "langgraph_store_idx"

            # Check if index already exists
            try:
                await self._execute_command("FT.INFO", index_name)
                logger.debug(f"Search index {index_name} already exists")
                return
            except Exception:
                # Index doesn't exist, create it
                pass

            # Create the index using raw FT.CREATE command with correct syntax
            cmd = self._create_index_command(index_name, "langgraph")
            await self._execute_command(*cmd)
            logger.info(f"Created search index {index_name} with TAG fields")

        except Exception as e:
            logger.error(f"Failed to setup search index: {e}")
            # Don't raise the error, just log it and continue without search

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        pool_size: int | None = None,
        pool_timeout: float | None = None,
    ) -> AsyncGenerator[AsyncValkeyStore, None]:
        """Create an AsyncValkeyStore from a connection string."""
        client = None
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

            await aset_client_info(client)
            store = cls(client, index=index, ttl=ttl)
            yield store
        finally:
            if client:
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception as e:
                    logger.debug(f"Error closing client: {e}")

    @classmethod
    @asynccontextmanager
    async def from_pool(
        cls,
        pool: ConnectionPool,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncGenerator[AsyncValkeyStore, None]:
        """Create an AsyncValkeyStore from an existing connection pool."""
        client = None
        try:
            client = Valkey.from_pool(connection_pool=pool)
            await aset_client_info(client)
            store = cls(client, index=index, ttl=ttl)
            yield store
        finally:
            if client:
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception as e:
                    logger.debug(f"Error closing client: {e}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations synchronously."""
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Use ValkeyStore for synchronous operations."
        )

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations asynchronously."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                result = await self._handle_get_async(op)
                results.append(result)
            elif isinstance(op, PutOp):
                await self._handle_put_async(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                search_result = await self._handle_search_async(op)
                results.append(search_result)  # type: ignore[arg-type]
            elif isinstance(op, ListNamespacesOp):
                list_result = await self._handle_list_async(op)
                results.append(list_result)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def _handle_get_async(self, op: GetOp) -> Item | None:
        """Handle get operation asynchronously."""
        try:
            key = self._build_key(op.namespace, op.key)
            result = await self._execute_client_method("hgetall", key)

            if not result:
                return None

            # Handle ResponseT type for async context
            result = await self._handle_response_t_async(result)
            if result is None:
                return None

            # Use shared core logic from base class
            item = self._handle_get_core(op, result)
            if item is None:
                return None

            # Refresh TTL if configured (async version)
            if op.refresh_ttl and self.ttl_config:
                ttl = self.ttl_config.get("default_ttl")
                if ttl:
                    await self._execute_client_method("expire", key, int(ttl * 60))

            return item
        except Exception as e:
            logger.error(f"Error in get operation: {e}")
            return None

    async def _handle_put_async(self, op: PutOp) -> None:
        """Handle put operation asynchronously."""
        # Use shared core logic from base class
        key, hash_fields = self._handle_put_core(op)

        if op.value is None:
            # Handle deletion
            try:
                await self._execute_client_method("delete", key)
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}")
            return

        # Generate embeddings asynchronously if needed
        if hash_fields and self.embeddings and op.index is not False:
            vector = await self._generate_embeddings_async(op)
            if vector:
                # Update hash_fields with vector - convert to base64 string for storage
                import base64
                import struct

                vector_bytes = b"".join(struct.pack("<f", x) for x in vector)
                hash_fields["vector"] = base64.b64encode(vector_bytes).decode("utf-8")

        try:
            # Use HSET to store as hash fields for better vector search compatibility
            await self._execute_client_method("hset", key, mapping=hash_fields)

            # Set TTL if specified
            if op.ttl is not None:
                ttl_seconds = int(op.ttl * 60)  # Convert minutes to seconds
                await self._execute_client_method("expire", key, ttl_seconds)
        except Exception as e:
            logger.error(f"Error in put operation: {e}")
            raise

    async def _generate_embeddings_async(self, op: PutOp) -> list[float] | None:
        """Generate embeddings asynchronously for the given operation."""
        if not self.embeddings:
            return None

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

            # Use async embeddings method
            # Type checker and runtime will ensure embeddings has aembed_documents
            vectors = await self.embeddings.aembed_documents(texts)
            return vectors[0] if vectors else None

        except EmbeddingGenerationError:
            # Re-raise EmbeddingGenerationError
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def _handle_search_async(self, op: SearchOp) -> list[SearchItem]:
        """Handle search operation using search strategy pattern asynchronously."""
        try:
            return await self._search_manager.search(op)
        except Exception as e:
            logger.error(f"Error in search operation: {e}")
            return []

    async def _handle_list_async(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle list namespaces operation asynchronously."""
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
                    keys_result = await self._execute_client_method("keys", pattern)
                    # Handle ResponseT type for async context
                    keys_result = await self._handle_response_t_async(keys_result)
                    if keys_result is None:
                        continue

                    # Convert keys to strings
                    keys = []
                    if hasattr(keys_result, "__iter__") and not isinstance(
                        keys_result, (str, bytes)
                    ):
                        for key in keys_result:
                            if isinstance(key, bytes):
                                keys.append(key.decode("utf-8"))
                            elif isinstance(key, str):
                                keys.append(key)
                            else:
                                keys.append(str(key))
                    all_keys.extend(keys)
                except Exception as e:
                    logger.error(f"Error scanning pattern {pattern}: {e}")
                    continue

            # Use shared core logic from base class
            return self._handle_list_core(op, all_keys)

        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return []

    async def _refresh_ttl_for_items_async(self, items: list[SearchItem]) -> None:
        """Refresh TTL for a list of items."""
        if not self.ttl_config:
            return

        ttl_seconds = self.ttl_config.get("default_ttl")
        if ttl_seconds:
            for item in items:
                item_key = self._build_key(item.namespace, item.key)
                try:
                    await self._execute_client_method(
                        "expire", item_key, int(ttl_seconds * 60)
                    )
                except Exception as e:
                    logger.error(f"Error refreshing TTL for {item_key}: {e}")

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Get an item from the store."""
        from langgraph.store.base import GetOp

        op = GetOp(namespace=namespace, key=key, refresh_ttl=refresh_ttl or False)
        return await self._handle_get_async(op)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Put an item in the store."""
        resolved_ttl = _ensure_ttl(self.ttl_config, ttl)
        op = PutOp(
            namespace=namespace, key=key, value=value, index=index, ttl=resolved_ttl
        )
        await self._handle_put_async(op)

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete an item from the store."""
        op = PutOp(namespace=namespace, key=key, value=None)
        await self._handle_put_async(op)

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
        return await self._handle_search_async(op)

    async def alist_namespaces(
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
        return await self._handle_list_async(op)

    async def _handle_response_t_async(self, result: Any) -> Any:
        """Handle ResponseT type from Valkey client in async context.

        ResponseT can be either the actual result or an awaitable.
        In async context, we need to await if it's awaitable.
        """
        if hasattr(result, "__await__"):
            try:
                return await result
            except Exception as e:
                logger.error(f"Error awaiting result: {e}")
                return None
        return result

    async def _safe_parse_keys_async(self, keys_result: Any) -> list[str]:
        """Safely parse keys result handling ResponseT type in async context."""
        # Handle ResponseT type - ensure we get the actual result
        keys_result = await self._handle_response_t_async(keys_result)
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

    # Sync methods that raise NotImplementedError
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Get an item from the store synchronously.

        Note:
            This sync method is not supported by the AsyncValkeyStore class.
            Use aget() instead, or consider using ValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Consider using ValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import ValkeyStore\n"
            "See the documentation for more information."
        )

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Put an item in the store synchronously.

        Note:
            This sync method is not supported by the AsyncValkeyStore class.
            Use aput() instead, or consider using ValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Consider using ValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import ValkeyStore\n"
            "See the documentation for more information."
        )

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete an item from the store synchronously.

        Note:
            This sync method is not supported by the AsyncValkeyStore class.
            Use adelete() instead, or consider using ValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Consider using ValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import ValkeyStore\n"
            "See the documentation for more information."
        )

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
        """Search for items in the store synchronously.

        Note:
            This sync method is not supported by the AsyncValkeyStore class.
            Use asearch() instead, or consider using ValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Consider using ValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import ValkeyStore\n"
            "See the documentation for more information."
        )

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List namespaces in the store synchronously.

        Note:
            This sync method is not supported by the AsyncValkeyStore class.
            Use alist_namespaces() instead, or consider using ValkeyStore.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeyStore does not support sync methods. "
            "Consider using ValkeyStore instead.\n"
            "from langgraph_checkpoint_aws.store.valkey import ValkeyStore\n"
            "See the documentation for more information."
        )
