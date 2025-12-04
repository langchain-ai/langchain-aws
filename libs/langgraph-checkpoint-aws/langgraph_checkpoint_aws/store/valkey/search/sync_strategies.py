"""Synchronous search strategies for ValkeyStore.

This module contains all synchronous search strategy implementations.
"""

from __future__ import annotations

import logging
import struct
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langgraph.store.base import SearchItem, SearchOp

if TYPE_CHECKING:
    from valkey.commands.search.query import Query  # type: ignore[import-untyped]
else:
    try:
        from valkey.commands.search.query import Query  # type: ignore[import-untyped]
    except ImportError:
        Query = None  # type: ignore[assignment, misc]

from ..exceptions import SearchIndexError
from .adapters import (
    DocumentAdapter,
    EmbeddingAdapter,
    HashSearchAdapter,
    SyncClientAdapter,
)
from .helpers import BaseSearchHelper
from .protocols import ValkeyClientProtocol

logger = logging.getLogger(__name__)


# ============================================================================
# Base Search Strategy (Sync)
# ============================================================================


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, client: ValkeyClientProtocol, store_instance: Any) -> None:
        """Initialize search strategy.

        Args:
            client: Valkey client instance
            store_instance: Reference to the store instance for accessing methods
        """
        self.client = client
        self.store = store_instance

    @abstractmethod
    def search(self, op: SearchOp) -> list[SearchItem]:
        """Execute search using this strategy.

        Args:
            op: Search operation parameters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this search strategy is available.

        Returns:
            True if strategy can be used, False otherwise
        """
        pass

    def _refresh_ttl_for_items(self, items: list[SearchItem]) -> None:
        """Refresh TTL for search result items."""
        if not self.store.ttl_config:
            return

        ttl_seconds = self.store.ttl_config.get("default_ttl")
        if ttl_seconds:
            for item in items:
                item_key = self.store._build_key(item.namespace, item.key)
                try:
                    self.client.expire(item_key, int(ttl_seconds * 60))
                except Exception as e:
                    logger.error(f"Error refreshing TTL for {item_key}: {e}")


# ============================================================================
# Vector Search Strategy
# ============================================================================


class VectorSearchStrategy(SearchStrategy):
    """Vector similarity search strategy using Valkey Search."""

    def is_available(self) -> bool:
        """Check if vector search is available."""
        return (
            self.store.embeddings is not None
            and self.store.dims is not None
            and self.store._is_search_available()
            and self.store.index is not None
        )

    def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform vector similarity search."""
        if not op.query:
            return []

        try:
            return self._execute_vector_search(op)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchIndexError(
                f"Vector search failed: {e}",
                index_name=self.store.collection_name,
                index_operation="search",
            ) from e

    def _execute_vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute the vector search query."""
        # Build search query using Valkey vector search syntax
        index_name = self.store.collection_name

        # Use BaseSearchHelper to build query and filters
        vector_query, _ = BaseSearchHelper.build_vector_query(
            namespace_prefix=op.namespace_prefix,
            filter_dict=op.filter,
            limit=op.limit,
            offset=op.offset,
        )

        # Create the query object with proper dialect

        query = (
            Query(vector_query)
            .return_fields("id", "score")
            .paging(0, op.limit + op.offset)
            .dialect(2)
        )

        # Generate query embedding for vector search
        try:
            if not self.store.embeddings:
                raise SearchIndexError(
                    "Embeddings not configured for vector search",
                    index_name=index_name,
                    index_operation="embedding_generation",
                )

            if not op.query:
                raise SearchIndexError(
                    "Query is required for vector search",
                    index_name=index_name,
                    index_operation="embedding_generation",
                )

            # Use EmbeddingAdapter for unified embedding generation
            adapter = EmbeddingAdapter(self.store.embeddings)
            query_vector = adapter.embed_query_sync(op.query, index_name=index_name)

            # Pack vector to binary bytes for FT.SEARCH
            vec_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)
            query_params: dict[str, str | int | float | bytes] = {"vec": vec_bytes}
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

        try:
            results = self.client.ft(index_name).search(query, query_params)
            items = self._process_vector_search_results(results, op)

            # Refresh TTL if configured
            if op.refresh_ttl:
                self._refresh_ttl_for_items(items)

            return items

        except Exception as e:
            logger.error(f"Vector search execution failed: {e}")
            raise SearchIndexError(
                f"Vector search execution failed: {e}",
                index_name=index_name,
                index_operation="query_execution",
            ) from e

    def _process_vector_search_results(
        self, results: Any, op: SearchOp
    ) -> list[SearchItem]:
        """Process vector search results into SearchItem objects."""
        items: list[SearchItem] = []

        # Check if results has docs attribute and process results
        docs = getattr(results, "docs", None)
        if not docs:
            return items

        # Create sync client adapter
        client_adapter = SyncClientAdapter(self.client, self.store)

        # Skip offset items and process results
        for _i, doc in enumerate(docs[op.offset :]):
            try:
                # Use DocumentAdapter for consistent parsing
                item = DocumentAdapter.parse_vector_search_doc_sync(
                    doc, client_adapter, self.store, apply_filters=op.filter
                )
                if item:
                    items.append(item)

            except Exception as e:
                logger.error(f"Error processing search result: {e}")
                continue

        return items


# ============================================================================
# Hash Search Strategy
# ============================================================================


class HashSearchStrategy(SearchStrategy):
    """Hash-based search strategy for when vector search is unavailable."""

    def is_available(self) -> bool:
        """Hash search is always available."""
        return True

    def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform hash-based search."""
        try:
            # Use HashSearchAdapter for consistent implementation
            namespace = op.namespace_prefix or ()
            hash_results = HashSearchAdapter.scan_and_score_keys_sync(
                client=self.client,
                store=self.store,
                namespace=namespace,
                query=op.query,
                filter_dict=op.filter,
                limit=op.limit,
                offset=op.offset,
            )

            # Convert to SearchItems
            items = HashSearchAdapter.convert_hash_results_to_items_sync(
                client=self.client, store=self.store, results=hash_results
            )

            if items:
                # Refresh TTL if configured
                if op.refresh_ttl:
                    self._refresh_ttl_for_items(items)

                return items

            return []
        except Exception as e:
            logger.debug(f"Hash-based search failed: {e}")
            return []


# ============================================================================
# Key Pattern Search Strategy
# ============================================================================


class KeyPatternSearchStrategy(SearchStrategy):
    """Fallback search strategy using key pattern matching."""

    def is_available(self) -> bool:
        """Key pattern search is always available as final fallback."""
        return True

    def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform key pattern search as final fallback."""
        items = []

        try:
            # Use BaseSearchHelper to build namespace pattern
            pattern = BaseSearchHelper.build_key_pattern(op.namespace_prefix)

            # Use SCAN for better performance with large datasets
            cursor = 0
            all_keys = []

            while True:
                scan_result = self.client.scan(cursor, match=pattern, count=1000)
                # Handle ResponseT type for scan result
                scan_result = self.store._handle_response_t(scan_result)
                if scan_result is None:
                    break

                cursor, keys = scan_result
                keys = self.store._safe_parse_keys(keys)
                all_keys.extend(keys)
                if cursor == 0:
                    break

            # Sort keys for consistent ordering
            all_keys.sort()

            # Create sync client adapter
            client_adapter = SyncClientAdapter(self.client, self.store)

            # Process all keys first to calculate scores, then apply pagination
            scored_items = []

            for key in all_keys:
                try:
                    # Use DocumentAdapter for consistent parsing
                    item = DocumentAdapter.parse_scan_key_sync(
                        key=key,
                        client_adapter=client_adapter,
                        store=self.store,
                        query=op.query,
                        apply_filters=op.filter,
                        namespace_prefix=op.namespace_prefix,
                    )
                    if item:
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
            if op.refresh_ttl:
                self._refresh_ttl_for_items(items)

        except Exception as e:
            logger.error(f"Error in key pattern search: {e}")

        return items


# ============================================================================
# Search Strategy Manager
# ============================================================================


class SearchStrategyManager:
    """Manages and coordinates different search strategies."""

    def __init__(self, client: ValkeyClientProtocol, store_instance: Any) -> None:
        """Initialize search strategy manager.

        Args:
            client: Valkey client instance
            store_instance: Reference to the store instance
        """
        self.strategies = [
            VectorSearchStrategy(client, store_instance),
            HashSearchStrategy(client, store_instance),
            KeyPatternSearchStrategy(client, store_instance),
        ]

    def search(self, op: SearchOp) -> list[SearchItem]:
        """Execute search using the first available strategy.

        Args:
            op: Search operation parameters

        Returns:
            List of search results
        """
        for strategy in self.strategies:
            if strategy.is_available():
                try:
                    # For vector search, only use if we have a query
                    if isinstance(strategy, VectorSearchStrategy) and not op.query:
                        continue

                    results = strategy.search(op)
                    if results:  # If strategy succeeded and returned results
                        return results
                except Exception as e:
                    logger.debug(f"Strategy {type(strategy).__name__} failed: {e}")
                    # Continue to next strategy
                    continue

        # If all strategies failed or returned no results, return empty list
        return []


__all__ = [
    "SearchStrategy",
    "VectorSearchStrategy",
    "HashSearchStrategy",
    "KeyPatternSearchStrategy",
    "SearchStrategyManager",
]
