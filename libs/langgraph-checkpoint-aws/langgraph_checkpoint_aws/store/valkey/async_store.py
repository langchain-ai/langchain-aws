"""Async Valkey store implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

import orjson
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
from .search_strategies import SearchStrategyManager

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
        self._search_strategy_manager = SearchStrategyManager(client, self)  # type: ignore[assignment]

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

    def _generate_embeddings(self, texts: list[str]) -> list[float] | None:
        """Override base class method to handle async embedding generation."""
        # This method should not be called directly in async context
        # Use _generate_embeddings_async instead
        logger.warning(
            "_generate_embeddings called in async context, use "
            "_generate_embeddings_async"
        )
        return None

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

            # Use async embedding method if available
            try:
                if hasattr(self.embeddings, "aembed_documents"):
                    vectors = await self.embeddings.aembed_documents(texts)
                    return vectors[0] if vectors else None
                elif hasattr(self.embeddings, "embed_documents"):
                    # Run sync embeddings in executor
                    loop = asyncio.get_running_loop()
                    vectors = await loop.run_in_executor(
                        None, self.embeddings.embed_documents, texts
                    )
                    return vectors[0] if vectors else None
                else:
                    return None
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            return None

    async def _handle_search_async(self, op: SearchOp) -> list[SearchItem]:
        """Handle search operation using search strategy pattern asynchronously."""
        try:
            # For now, use the key pattern search as fallback
            return await self._key_pattern_search_async(op)
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

    async def _vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Perform vector similarity search using Valkey Search."""
        try:
            # Generate query embedding
            query_vector = None
            if self.embeddings and op.query:
                try:
                    # Handle different types of embedding functions
                    if callable(self.embeddings):
                        # Try calling it directly (works for mock functions and
                        # simple callables)
                        vectors = self.embeddings([op.query])
                        query_vector = vectors[0] if vectors else None
                    elif hasattr(self.embeddings, "embed_documents"):
                        # This is a real embedding function, use sync method
                        vectors = self.embeddings.embed_documents([op.query])
                        query_vector = vectors[0] if vectors else None
                    elif hasattr(self.embeddings, "aembed_documents"):
                        # This is a real embedding function, use async method
                        vectors = await self.embeddings.aembed_documents([op.query])
                        query_vector = vectors[0] if vectors else None
                    else:
                        logger.warning(
                            "Embeddings is not callable, skipping vector search"
                        )
                        return []
                except Exception as e:
                    logger.error(f"Error generating query embedding: {e}")
                    return []

            if not query_vector:
                return []

            # Build search query
            index_name = self.collection_name

            # Create namespace filter
            namespace_filter = ""
            if op.namespace_prefix:
                namespace_prefix = "/".join(op.namespace_prefix)
                namespace_filter = f"@namespace:{{{namespace_prefix}*}}"

            # Create additional filters
            filter_parts = []
            if namespace_filter:
                filter_parts.append(namespace_filter)

            if op.filter:
                for key, value in op.filter.items():
                    # Escape special characters in filter values
                    escaped_value = str(value).replace(":", "\\:")
                    filter_parts.append(f"@{key}:{{{escaped_value}}}")

            # Combine filters
            query_filter = " ".join(filter_parts) if filter_parts else "*"

            # Build vector search query using the recommended approach
            # Execute search - convert vector to bytes for Valkey
            import struct

            from valkey.commands.search.query import Query

            vector_bytes = b"".join(struct.pack("<f", x) for x in query_vector)

            # Create query using the recommended .paging() method
            query = (
                Query(
                    f"({query_filter})=>[KNN {op.limit + op.offset} @vector $BLOB "
                    f"AS score]"
                )
                .sort_by("score")
                .return_fields("id", "score")
                .paging(op.offset, op.limit)
                .dialect(2)
            )

            search_params: dict[str, str | int | float | bytes] = {"BLOB": vector_bytes}

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.ft(index_name).search(query, search_params),
                )

                items = []
                # Check if results has docs attribute and process results
                docs = getattr(results, "docs", None)
                if docs:
                    # Skip offset items and process results
                    for _i, doc in enumerate(docs):
                        try:
                            # Parse document data
                            doc_data = doc.__dict__

                            # Extract namespace and key from document ID
                            doc_id = doc_data.get("id", "")
                            if doc_id.startswith("langgraph:"):
                                key_path = doc_id[10:]  # Remove 'langgraph:' prefix
                                namespace, item_key = self._parse_key(key_path)
                            else:
                                continue

                            # Get the actual document content
                            full_key = self._build_key(namespace, item_key)
                            value_data = await asyncio.get_event_loop().run_in_executor(
                                None, self.client.get, full_key
                            )
                            if not value_data:
                                continue

                            value_data = await self._handle_response_t_async(value_data)
                            if value_data is None:
                                continue

                            # Parse document using DocumentProcessor
                            try:
                                parsed_data = orjson.loads(value_data)
                                value = parsed_data.get("value", {})
                                created_at = datetime.fromisoformat(
                                    parsed_data.get(
                                        "created_at", datetime.now().isoformat()
                                    )
                                )
                                updated_at = datetime.fromisoformat(
                                    parsed_data.get(
                                        "updated_at", datetime.now().isoformat()
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error parsing document data: {e}")
                                continue

                            # Apply additional filters
                            if not self._apply_filter(value, op.filter):
                                continue

                            # Get similarity score from search results
                            score = float(doc_data.get("score", 0.0))

                            item = SearchItem(
                                namespace=namespace,
                                key=item_key,
                                value=value,
                                created_at=created_at,
                                updated_at=updated_at,
                                score=score,
                            )
                            items.append(item)

                        except Exception as e:
                            logger.error(f"Error processing search result: {e}")
                            continue

                # Refresh TTL if configured
                if op.refresh_ttl and self.ttl_config:
                    await self._refresh_ttl_for_items_async(items)

                return items

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def _search_with_hash_async(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        filter_dict: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[tuple[tuple[str, ...], str, float]]:
        """Efficient search using hash fields when vector search is unavailable."""

        # Build scan pattern for namespace
        pattern = f"langgraph:{'/'.join(namespace)}/*" if namespace else "langgraph:*"

        # Use SCAN for efficient iteration
        cursor = 0
        results = []
        seen_keys = set()

        def scan_with_cursor(c):
            return self.client.scan(c, match=pattern, count=1000)

        while True:
            scan_result = await asyncio.get_event_loop().run_in_executor(
                None, scan_with_cursor, cursor
            )
            # No need to handle ResponseT here - run_in_executor already resolves it
            if scan_result is None:
                break
            cursor, keys = scan_result
            # Convert keys to strings directly
            parsed_keys = []
            if hasattr(keys, "__iter__") and not isinstance(keys, (str, bytes)):
                for key in keys:
                    if isinstance(key, bytes):
                        parsed_keys.append(key.decode("utf-8"))
                    elif isinstance(key, str):
                        parsed_keys.append(key)
                    else:
                        parsed_keys.append(str(key))
            keys = parsed_keys

            for key in keys:
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                try:
                    # Get the value
                    value = await asyncio.get_event_loop().run_in_executor(
                        None, self.client.get, key
                    )
                    if value:
                        # Handle ResponseT type for value
                        value = await self._handle_response_t_async(value)
                        if value is None:
                            continue
                        doc_data = orjson.loads(value)
                        # Apply filters efficiently
                        if filter_dict and not self._apply_filter(
                            doc_data.get("value", {}), filter_dict
                        ):
                            continue

                        # Calculate score
                        score = self._calculate_simple_score(query, doc_data)
                        if score > 0:
                            namespace, key = self._parse_key(key, "langgraph:")
                            results.append((namespace, key, score))

                except Exception as e:
                    logger.debug(f"Error processing key {key}: {e}")
                    continue

            if cursor == 0:
                break

        # Sort results by score descending
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

        # Apply pagination
        start_idx = offset
        end_idx = start_idx + limit if limit else len(sorted_results)

        logger.debug(
            f"Hash search: total results={len(sorted_results)}, "
            f"offset={offset}, limit={limit}, start_idx={start_idx}, "
            f"end_idx={end_idx}"
        )

        return sorted_results[start_idx:end_idx]

    async def _convert_to_search_items_async(
        self, results: list[tuple[tuple[str, ...], str, float]]
    ) -> list[SearchItem]:
        """Convert hash search results to SearchItem objects."""
        items = []
        for namespace, key, score in results:
            try:
                # Get full document data
                full_key = self._build_key(namespace, key)
                value_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.get, full_key
                )
                if value_data:
                    value_data = await self._handle_response_t_async(value_data)
                    if value_data:
                        # Parse document using DocumentProcessor
                        try:
                            parsed_data = orjson.loads(value_data)
                            value = parsed_data.get("value", {})
                            created_at = datetime.fromisoformat(
                                parsed_data.get(
                                    "created_at", datetime.now().isoformat()
                                )
                            )
                            updated_at = datetime.fromisoformat(
                                parsed_data.get(
                                    "updated_at", datetime.now().isoformat()
                                )
                            )
                            items.append(
                                SearchItem(
                                    namespace=namespace,
                                    key=key,
                                    value=value,
                                    created_at=created_at,
                                    updated_at=updated_at,
                                    score=score,
                                )
                            )
                        except Exception as e:
                            logger.error(f"Error parsing document data: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Error converting result {namespace}/{key}: {e}")
                continue
        return items

    async def _key_pattern_search_async(self, op: SearchOp) -> list[SearchItem]:
        """Fallback search using key pattern matching."""
        items = []

        try:
            # Build pattern with langgraph: prefix
            if op.namespace_prefix:
                namespace_path = "/".join(op.namespace_prefix)
                pattern = f"langgraph:{namespace_path}/*"
            else:
                pattern = "langgraph:*"

            # Use SCAN for better performance with large datasets
            cursor = 0
            all_keys = []

            while True:
                # Execute scan and properly await the result
                scan_result = await self._execute_client_method(
                    "scan", cursor, match=pattern, count=1000
                )
                if scan_result is None:
                    break

                # Handle ResponseT type for scan result
                scan_result = await self._handle_response_t_async(scan_result)
                if scan_result is None:
                    break

                cursor, keys = scan_result
                # Convert keys to strings directly
                parsed_keys = []
                if hasattr(keys, "__iter__") and not isinstance(keys, (str, bytes)):
                    for key in keys:
                        if isinstance(key, bytes):
                            parsed_keys.append(key.decode("utf-8"))
                        elif isinstance(key, str):
                            parsed_keys.append(key)
                        else:
                            parsed_keys.append(str(key))
                keys = parsed_keys
                all_keys.extend(keys)
                if cursor == 0:
                    break

            # Filter keys by namespace prefix if specified
            if op.namespace_prefix:
                namespace_path = "/".join(op.namespace_prefix)
                filtered_keys = []
                for key in all_keys:
                    if key.startswith("langgraph:"):
                        key_path = key[10:]  # Remove "langgraph:" prefix
                        if key_path.startswith(namespace_path + "/"):
                            filtered_keys.append(key)
                all_keys = filtered_keys

            # Sort keys for consistent ordering
            all_keys.sort()

            # Process all keys first to calculate scores, then apply pagination
            scored_items = []

            for key in all_keys:  # Fixed: use all_keys instead of limited_keys
                try:
                    # Use HGETALL since we store data as hash fields
                    hash_data = await self._execute_client_method("hgetall", key)
                    # Ensure hash_data is properly typed as dict
                    if not hash_data or not isinstance(hash_data, dict):
                        continue

                    # Parse key - remove langgraph: prefix
                    if key.startswith("langgraph:"):
                        key_path = key[10:]  # Remove "langgraph:" prefix
                    else:
                        key_path = key

                    # Parse namespace and key
                    namespace, item_key = self._parse_key(key_path)

                    # Use DocumentProcessor to convert hash fields back to document
                    # format
                    document = DocumentProcessor.convert_hash_to_document(hash_data)
                    if document is None:
                        continue

                    # Parse the JSON-encoded value using DocumentProcessor
                    parsed_value = DocumentProcessor.parse_document_value(document)
                    if parsed_value is None:
                        continue

                    # Parse timestamps using DocumentProcessor
                    created_at, updated_at = DocumentProcessor.parse_timestamps(
                        document
                    )

                    # Apply filter using base class method
                    if not self._apply_filter(parsed_value, op.filter):
                        continue

                    # Calculate score using base class method
                    score = self._calculate_simple_score(op.query, parsed_value)

                    scored_items.append(
                        (
                            score,
                            SearchItem(
                                namespace=namespace,
                                key=item_key,
                                value=parsed_value,
                                created_at=created_at,
                                updated_at=updated_at,
                                score=score,
                            ),
                        )
                    )

                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
                    continue

            # Sort by score descending
            scored_items.sort(key=lambda x: x[0], reverse=True)

            # Apply pagination after scoring and sorting
            start_idx = op.offset
            end_idx = start_idx + op.limit
            paginated_items = scored_items[start_idx:end_idx]

            # Debug logging for pagination
            logger.debug(
                f"Key pattern search: total items={len(scored_items)}, "
                f"offset={op.offset}, limit={op.limit}, start_idx={start_idx}, "
                f"end_idx={end_idx}, paginated_count={len(paginated_items)}"
            )
            if paginated_items:
                keys = [item[1].key for item in paginated_items]
                logger.debug(f"Paginated keys: {keys}")

            # Extract SearchItem objects
            items = [item for _, item in paginated_items]

            # Refresh TTL if configured
            if op.refresh_ttl and self.ttl_config:
                await self._refresh_ttl_for_items_async(items)

        except Exception as e:
            logger.error(f"Error in key pattern search: {e}")

        return items

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
