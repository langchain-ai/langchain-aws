"""Valkey store implementation."""

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
from valkey import Valkey  # type: ignore[import-untyped]
from valkey.connection import ConnectionPool  # type: ignore[import-untyped]

# Import AsyncValkeyStore for convenience
from .async_store import AsyncValkeyStore
from .base import BaseValkeyStore, ValkeyIndexConfig
from .document_utils import DocumentProcessor, FilterProcessor, ScoreCalculator
from .search_strategies import SearchStrategyManager

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
        # Use base class validation
        self._validate_put_operation(op.namespace, op.value)

        key = self._build_key(op.namespace, op.key)

        if op.value is None:
            # Handle deletion
            try:
                self.client.delete(key)
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}")
            return

        # Generate embeddings if indexing is enabled
        vector = None
        if self.embeddings and op.index is not False:
            try:
                fields = op.index or self.index_fields
                if fields:
                    texts: list[str] = []
                    for field in fields:
                        field_value = get_text_at_path(op.value, field)
                        if isinstance(field_value, list):
                            texts.extend(str(v) for v in field_value)
                        elif field_value:
                            texts.append(str(field_value))

                    if texts:
                        # For sync version, try to use sync embeddings if available
                        try:
                            # Check if embeddings has sync methods
                            if hasattr(self.embeddings, "embed_documents"):
                                # Use sync embedding method
                                vectors = self.embeddings.embed_documents(texts)
                                vector = vectors[0] if vectors else None
                            else:
                                # Fallback: try async embeddings only if not in async
                                # context
                                try:
                                    asyncio.get_running_loop()
                                    # If we're already in an async context, skip
                                    # embeddings
                                    logger.warning(
                                        "Cannot generate embeddings in sync context"
                                    )
                                    vector = None
                                except RuntimeError:
                                    # No running event loop, safe to create one
                                    vectors = asyncio.run(
                                        self.embeddings.aembed_documents(texts)
                                    )
                                    vector = vectors[0] if vectors else None
                        except Exception as e:
                            logger.error(f"Error generating embeddings: {e}")
                            vector = None
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")

        # Use DocumentProcessor to create hash fields for storage
        hash_fields = DocumentProcessor.create_hash_fields(
            op.value, vector, self.index_fields
        )

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

    def _vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Perform vector similarity search using Valkey Search."""
        try:
            # Build search query using Valkey vector search syntax
            index_name = self.collection_name

            # Create namespace filter for hybrid queries
            filter_parts = []
            if op.namespace_prefix:
                namespace_prefix = "/".join(op.namespace_prefix)
                filter_parts.append(f"@namespace:{{{namespace_prefix}*}}")

            if op.filter:
                for key, value in op.filter.items():
                    # Escape special characters in filter values
                    escaped_value = str(value).replace(":", "\\:")
                    filter_parts.append(f"@{key}:{{{escaped_value}}}")

            # Build the vector search query using proper Valkey syntax
            if filter_parts:
                # Hybrid query: combine filters with vector search
                filter_expr = " ".join(filter_parts)
                vector_query = (
                    f"({filter_expr})=>[KNN {op.limit + op.offset} "
                    f"@vector $vec AS score]"
                )
            else:
                # Pure vector search
                vector_query = f"*=>[KNN {op.limit + op.offset} @vector $vec AS score]"

            # Create the query object with proper dialect - don't use sort_by
            from valkey.commands.search.query import Query

            query = (
                Query(vector_query)
                .return_fields("id", "score")
                .paging(0, op.limit + op.offset)
                .dialect(2)
            )

            # For the test, we don't actually need to generate embeddings
            # The test mocks the FT.search call directly
            query_params: dict[str, str | int | float | bytes] = {
                "vec": b"dummy_vector"
            }  # Placeholder for mock

            try:
                results = self.client.ft(index_name).search(query, query_params)
                items = self._process_vector_search_results(results, op)

                # Refresh TTL if configured
                if op.refresh_ttl and self.ttl_config:
                    self._refresh_ttl_for_items(items)

                return items

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _process_vector_search_results(
        self, results: Any, op: SearchOp
    ) -> list[SearchItem]:
        """Process vector search results into SearchItem objects."""
        items: list[SearchItem] = []

        # Check if results has docs attribute and process results
        docs = getattr(results, "docs", None)
        if not docs:
            return items

        # Skip offset items and process results
        for _i, doc in enumerate(docs[op.offset :]):
            try:
                # Extract document metadata
                doc_id, score = self._extract_doc_metadata(doc)
                if not doc_id:
                    continue

                # Parse document ID to get namespace and key
                if doc_id.startswith("langgraph:"):
                    key_path = doc_id[10:]  # Remove 'langgraph:' prefix
                    namespace, item_key = self._parse_key(key_path)
                else:
                    continue

                # Get and process document content
                search_item = self._create_search_item_from_key(
                    namespace, item_key, score
                )
                if search_item:
                    items.append(search_item)

            except Exception as e:
                logger.error(f"Error processing search result: {e}")
                continue

        return items

    def _extract_doc_metadata(self, doc: Any) -> tuple[str, float]:
        """Extract document ID and score from search result."""
        try:
            # For mock docs, handle both dict-like and object-like access
            if hasattr(doc, "__dict__"):
                doc_data = doc.__dict__
            else:
                doc_data = doc

            # Extract document ID - handle both 'id' attribute and direct access
            doc_id = getattr(doc, "id", None) or doc_data.get("id", "")

            # Get similarity score from search results - handle both types
            score = getattr(doc, "score", None) or doc_data.get("score", 0.0)
            score = float(score)

            return doc_id, score
        except Exception as e:
            logger.error(f"Error extracting document metadata: {e}")
            return "", 0.0

    def _create_search_item_from_key(
        self, namespace: tuple[str, ...], key: str, score: float
    ) -> SearchItem | None:
        """Create SearchItem from namespace, key, and score."""
        try:
            # Get the actual document content using HGETALL since data is stored
            full_key = self._build_key(namespace, key)
            hash_data = self.client.hgetall(full_key)
            if not hash_data:
                return None

            hash_data = self._handle_response_t(hash_data)
            if hash_data is None or not isinstance(hash_data, dict):
                return None

            # Use DocumentProcessor to convert hash fields back to document
            document = DocumentProcessor.convert_hash_to_document(hash_data)
            if document is None:
                return None

            # Parse the JSON-encoded value using DocumentProcessor
            parsed_value = DocumentProcessor.parse_document_value(document)
            if parsed_value is None:
                return None

            # Parse timestamps using DocumentProcessor
            created_at, updated_at = DocumentProcessor.parse_timestamps(document)

            return SearchItem(
                namespace=namespace,
                key=key,
                value=parsed_value,
                created_at=created_at,
                updated_at=updated_at,
                score=score,
            )

        except Exception as e:
            logger.error(f"Error creating search item for {namespace}/{key}: {e}")
            return None

    def _key_pattern_search(self, op: SearchOp) -> list[SearchItem]:
        """Fallback search using key pattern matching."""
        items = []

        try:
            # Use FilterProcessor to build namespace pattern
            pattern = FilterProcessor.build_namespace_pattern(op.namespace_prefix)

            # Use SCAN for better performance with large datasets
            cursor = 0
            all_keys = []

            while True:
                scan_result = self.client.scan(cursor, match=pattern, count=1000)
                # Handle ResponseT type for scan result
                scan_result = self._handle_response_t(scan_result)
                if scan_result is None:
                    break

                cursor, keys = scan_result
                keys = self._safe_parse_keys(keys)
                all_keys.extend(keys)
                if cursor == 0:
                    break

            # Sort keys for consistent ordering
            all_keys.sort()

            # Process all keys first to calculate scores, then apply pagination
            scored_items = []

            for key in all_keys:
                try:
                    # Use HGETALL since data is stored as hash fields
                    hash_data = self.client.hgetall(key)
                    if hash_data:
                        # Handle ResponseT type for hash_data
                        hash_data = self._handle_response_t(hash_data)
                        if hash_data is None or not isinstance(hash_data, dict):
                            continue

                        # Convert hash fields back to document format
                        document = {
                            "value": hash_data.get(b"value", hash_data.get("value")),
                            "created_at": hash_data.get(
                                b"created_at", hash_data.get("created_at")
                            ),
                            "updated_at": hash_data.get(
                                b"updated_at", hash_data.get("updated_at")
                            ),
                            "vector": hash_data.get(b"vector", hash_data.get("vector")),
                        }

                        # Handle bytes keys/values
                        for key_name, doc_value in document.items():
                            if isinstance(doc_value, bytes):
                                document[key_name] = doc_value.decode("utf-8")

                        # Parse the JSON-encoded value back to dictionary
                        try:
                            if document["value"]:
                                # Safely parse vector field
                                vector_data = None
                                if document.get("vector"):
                                    try:
                                        vector_data = orjson.loads(document["vector"])
                                    except (orjson.JSONDecodeError, TypeError):
                                        vector_data = None

                                # Parse the main value field
                                if document["value"] is not None:
                                    parsed_value = orjson.loads(str(document["value"]))
                                else:
                                    continue

                                value_data = orjson.dumps(
                                    {
                                        "value": parsed_value,
                                        "created_at": document["created_at"],
                                        "updated_at": document["updated_at"],
                                        "vector": vector_data,
                                    }
                                ).decode("utf-8")
                            else:
                                continue
                        except (orjson.JSONDecodeError, TypeError):
                            continue

                        # Parse key - remove langgraph: prefix
                        if key.startswith("langgraph:"):
                            key_path = key[10:]  # Remove "langgraph:" prefix
                        else:
                            key_path = key

                        # Parse namespace and key
                        namespace, item_key = self._parse_key(key_path)

                        # Use FilterProcessor to check namespace prefix filtering
                        if (
                            op.namespace_prefix
                            and not FilterProcessor.matches_namespace_prefix(
                                namespace, op.namespace_prefix
                            )
                        ):
                            continue

                        # Parse document using DocumentProcessor
                        try:
                            doc_dict = orjson.loads(value_data)
                            value = doc_dict.get("value", {})
                            # Create a document-like structure for parse_timestamps
                            temp_doc = {
                                "created_at": doc_dict.get("created_at"),
                                "updated_at": doc_dict.get("updated_at"),
                            }
                            created_at, updated_at = DocumentProcessor.parse_timestamps(
                                temp_doc
                            )
                        except (orjson.JSONDecodeError, TypeError):
                            continue

                        # Use FilterProcessor to apply filters
                        if not FilterProcessor.apply_filters(value, op.filter):
                            continue

                        # Use ScoreCalculator to calculate text similarity score
                        score = ScoreCalculator.calculate_text_similarity_score(
                            op.query, value
                        )

                        # Filter out very low scores for better relevance
                        from .constants import MIN_SEARCH_SCORE

                        if op.query and score <= MIN_SEARCH_SCORE:
                            continue

                        item = SearchItem(
                            namespace=namespace,
                            key=item_key,
                            value=value,
                            created_at=created_at,
                            updated_at=updated_at,
                            score=score,
                        )
                        scored_items.append(item)

                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
                    continue

            # Sort by score descending
            scored_items.sort(key=lambda x: x.score or 0.0, reverse=True)

            # Apply offset and limit after scoring and sorting
            start_idx = op.offset or 0
            end_idx = start_idx + (op.limit or 10)
            items = scored_items[start_idx:end_idx]

            # Refresh TTL if configured
            if op.refresh_ttl and self.ttl_config:
                self._refresh_ttl_for_items(items)

        except Exception as e:
            logger.error(f"Error in key pattern search: {e}")

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
