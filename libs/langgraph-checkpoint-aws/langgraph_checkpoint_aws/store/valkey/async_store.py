"""Async Valkey store implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
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
        Basic usage with ValkeyIndexConfig:

        ```python
        from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore

        # Using connection string with ValkeyIndexConfig
        with AsyncValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            index={
                "collection_name": "my_documents",
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
                "fields": ["text"],
                "timezone": "UTC",
                "index_type": "hnsw"
            },
            ttl={"default_ttl": 60.0},  # 1 hour TTL
            pool_size=10  # Connection pool size
        ) as store:
            # Use store...

        # Advanced HNSW configuration
        with AsyncValkeyStore.from_conn_string(
            "valkey://localhost:6379",
            index={
                "collection_name": "embeddings_store",
                "dims": 768,
                "embed": "openai:text-embedding-3-small",
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

        # Using connection pool with FLAT index
        pool = ConnectionPool(
            "valkey://localhost:6379",
            min_size=5,
            max_size=20,
            timeout=30
        )
        with AsyncValkeyStore.from_pool(
            pool,
            index={
                "collection_name": "exact_search_store",
                "dims": 384,
                "embed": "openai:text-embedding-3-small",
                "index_type": "flat",
                "algorithm": "FLAT",
                "distance_metric": "L2"
            }
        ) as store:
            # Use store with exact search...

        # Direct initialization with ValkeyIndexConfig
        store = AsyncValkeyStore(
            Valkey("valkey://localhost:6379"),
            index={
                "collection_name": "langgraph_store_idx",
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
                "fields": ["text"]
            }
        )
        ```

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput` will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.
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
            self._search_available = True
            return True
        except Exception as e:
            logger.debug(f"Valkey Search not available: {e}")
            self._search_available = False
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
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: ValkeyIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        pool_size: int | None = None,
        pool_timeout: float | None = None,
    ) -> Generator[AsyncValkeyStore, None, None]:
        """Create an AsyncValkeyStore from a connection string."""
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
            asyncio.run(aset_client_info(client))
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
    ) -> Generator[AsyncValkeyStore, None, None]:
        """Create an AsyncValkeyStore from an existing connection pool."""
        try:
            client = Valkey.from_pool(connection_pool=pool)
            # Set client info for library identification
            asyncio.run(aset_client_info(client))
            store = cls(client, index=index, ttl=ttl)
            yield store
        finally:
            # Pool cleanup handled by owner
            pass

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations synchronously."""
        return asyncio.run(self.abatch(ops))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute operations asynchronously.

        Supports:
        - Get: Fetch items by key
        - Put: Store or update items
        - Search: Vector similarity and filtered search
        - List: Namespace exploration
        """
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                result = await self._handle_get(op)
                results.append(result)
            elif isinstance(op, PutOp):
                await self._handle_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                result = await self._handle_search(op)
                results.append(result)
            elif isinstance(op, ListNamespacesOp):
                result = await self._handle_list(op)
                results.append(result)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def _handle_get(self, op: GetOp) -> Item | None:
        """Handle get operation."""
        try:
            key = self._build_key(op.namespace, op.key)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get, key
            )
            if not result:
                return None

            # Handle ResponseT type using async method
            result = await self._handle_response_t_async(result)
            if result is None:
                return None

            # Parse the stored document using base class method
            value, created_at, updated_at = self._parse_document(result)

            if op.refresh_ttl and self.ttl_config:
                # Refresh TTL if configured
                ttl = self.ttl_config.get("default_ttl")
                if ttl:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.client.touch, key
                    )

            return Item(
                value=value,
                key=op.key,
                namespace=op.namespace,
                created_at=created_at,
                updated_at=updated_at,
            )
        except Exception as e:
            logger.error(f"Error in get operation: {e}")
            return None

    async def _handle_put(self, op: PutOp) -> None:
        """Handle put operation."""
        # Use base class validation
        self._validate_put_operation(op.namespace, op.value)

        key = self._build_key(op.namespace, op.key)

        if op.value is None:
            # Handle deletion
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client.delete, key
                )
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}")
            return

        # Generate embeddings if indexing is enabled
        vector = None
        if self.embeddings and op.index is not False:
            try:
                fields = op.index or self.index_fields
                if fields:
                    texts = []
                    for field in fields:
                        field_value = get_text_at_path(op.value, field)
                        if isinstance(field_value, list):
                            texts.extend(str(v) for v in field_value)
                        elif field_value:
                            texts.append(str(field_value))

                    if texts:
                        vectors = await self.embeddings.aembed_documents(texts)
                        # Use first vector if multiple were generated
                        vector = vectors[0] if vectors else None
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")

        # Create document using base class method
        value = self._create_document(op.value, vector)

        try:
            # Use standard SET command with optional TTL
            if op.ttl is not None:
                ttl_seconds = int(op.ttl * 60)  # Convert minutes to seconds
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.set(key, value, ex=ttl_seconds)
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.set(key, value)
                )
        except Exception as e:
            logger.error(f"Error in put operation: {e}")
            raise

    async def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle search operation with vector search and fallback to key pattern matching."""
        items = []
        logger.debug(
            f"Starting search with query='{op.query}', limit={op.limit}, offset={op.offset}"
        )

        try:
            # Try vector search first if we have embeddings, query, and search is available
            if (
                self.embeddings
                and op.query
                and self.dims
                and await self._is_search_available_async()
                and self.index
            ):
                logger.debug("Attempting vector search")
                items = await self._vector_search(op)
                if items:  # If vector search succeeded, return results
                    logger.debug(f"Vector search returned {len(items)} items")
                    return items
                else:
                    logger.debug("Vector search returned no items")

            # Try hash-based search if vector search failed or wasn't attempted
            if not items:
                try:
                    logger.debug("Attempting hash-based search")
                    hash_results = await self._search_with_hash_async(
                        op.namespace_prefix, op.query, op.filter, op.limit, op.offset
                    )
                    items = await self._convert_to_search_items_async(hash_results)
                    if items:
                        logger.debug(f"Hash-based search returned {len(items)} items")
                        return items
                    else:
                        logger.debug("Hash-based search returned no items")
                except Exception as e:
                    logger.debug(
                        f"Hash-based search failed: {e}, falling back to key pattern"
                    )

            # Final fallback to key pattern matching
            if not items:
                logger.debug("Falling back to key pattern search")
                items = await self._key_pattern_search_async(op)
                logger.debug(f"Key pattern search returned {len(items)} items")

        except Exception as e:
            logger.error(f"Error in search operation: {e}")

        return items

    async def _vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Perform vector similarity search using Valkey Search."""
        try:
            # Generate query embedding
            query_vector = None
            if self.embeddings and op.query:
                try:
                    # Handle different types of embedding functions
                    if callable(self.embeddings):
                        # Try calling it directly (works for mock functions and simple callables)
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
            index_name = "langgraph_store_idx"

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
                    f"({query_filter})=>[KNN {op.limit + op.offset} @vector $BLOB AS score]"
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
                    for _i, doc in enumerate(docs[op.offset :]):
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

                            # Parse document
                            value, created_at, updated_at = self._parse_document(
                                value_data
                            )

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

        while True:
            scan_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.scan(cursor, match=pattern, count=1000)
            )
            # Handle ResponseT type for scan result
            scan_result = await self._handle_response_t_async(scan_result)
            if scan_result is None:
                break
            cursor, keys = scan_result
            keys = await self._safe_parse_keys_async(keys)

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
            f"Hash search: total results={len(sorted_results)}, offset={offset}, limit={limit}, start_idx={start_idx}, end_idx={end_idx}"
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
                        value, created_at, updated_at = self._parse_document(value_data)
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
                scan_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda c=cursor: self.client.scan(c, match=pattern, count=1000),
                )
                # Handle ResponseT type for scan result
                scan_result = await self._handle_response_t_async(scan_result)
                if scan_result is None:
                    break
                cursor, keys = scan_result
                keys = await self._safe_parse_keys_async(keys)
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
                    value_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.client.get, key
                    )
                    if value_data:
                        # Handle ResponseT type for value_data
                        value_data = await self._handle_response_t_async(value_data)
                        if value_data is None:
                            continue

                        # Parse key - remove langgraph: prefix
                        if key.startswith("langgraph:"):
                            key_path = key[10:]  # Remove "langgraph:" prefix
                        else:
                            key_path = key

                        # Parse namespace and key
                        namespace, item_key = self._parse_key(key_path)

                        # Parse document using base class method
                        value, created_at, updated_at = self._parse_document(value_data)

                        # Apply filter using base class method
                        if not self._apply_filter(value, op.filter):
                            continue

                        # Calculate score using base class method
                        score = self._calculate_simple_score(op.query, value)

                        scored_items.append(
                            (
                                score,
                                SearchItem(
                                    namespace=namespace,
                                    key=item_key,
                                    value=value,
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
                f"Key pattern search: total items={len(scored_items)}, offset={op.offset}, limit={op.limit}, start_idx={start_idx}, end_idx={end_idx}, paginated_count={len(paginated_items)}"
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
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.client.expire, item_key, int(ttl_seconds * 60)
                    )
                except Exception as e:
                    logger.error(f"Error refreshing TTL for {item_key}: {e}")

    async def _handle_list(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
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

            # Scan for keys matching patterns
            all_keys = []
            for pattern in patterns:
                try:
                    keys_response = await asyncio.get_event_loop().run_in_executor(
                        None, self.client.keys, pattern
                    )
                    # Handle ResponseT type using base class method
                    keys = self._safe_parse_keys(keys_response)
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
            namespaces = self._extract_namespaces_from_keys(cleaned_keys, op.max_depth)

            # Convert to sorted list and apply pagination
            namespace_list = sorted(list(namespaces))

            # Apply offset and limit
            start_idx = op.offset or 0
            end_idx = start_idx + (op.limit or len(namespace_list))

            return namespace_list[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return []

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
        return await self._handle_get(op)

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
        await self._handle_put(op)

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete an item from the store."""
        op = PutOp(namespace=namespace, key=key, value=None)
        await self._handle_put(op)

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
        return await self._handle_search(op)

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
        return await self._handle_list(op)

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
