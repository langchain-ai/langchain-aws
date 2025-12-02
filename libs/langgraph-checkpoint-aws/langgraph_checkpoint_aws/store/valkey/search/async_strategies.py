"""Asynchronous search strategies for ValkeyStore.

This module contains all asynchronous search strategy implementations.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from abc import ABC, abstractmethod
from typing import Any

from langgraph.store.base import SearchItem, SearchOp

# Note: Query import is done inline where needed to avoid circular imports
from ..exceptions import SearchIndexError
from .adapters import (
    AsyncClientAdapter,
    DocumentAdapter,
    EmbeddingAdapter,
)
from .helpers import BaseSearchHelper

logger = logging.getLogger(__name__)


class AsyncSearchStrategy(ABC):
    """Abstract base class for async search strategies."""

    def __init__(self, store_instance: Any) -> None:
        """Initialize async search strategy.

        Args:
            store_instance: Reference to AsyncValkeyStore instance
        """
        self.store = store_instance

    @abstractmethod
    async def search(self, op: SearchOp) -> list[SearchItem]:
        """Execute search using this strategy.

        Args:
            op: Search operation parameters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this search strategy is available.

        Returns:
            True if strategy can be used, False otherwise
        """
        pass

    async def _refresh_ttl_for_items(
        self, items: list[SearchItem], op: SearchOp
    ) -> None:
        """Refresh TTL for search result items.

        Args:
            items: Search items to refresh TTL for
            op: Search operation with refresh_ttl flag
        """
        if op.refresh_ttl and self.store.ttl_config:
            await self.store._refresh_ttl_for_items_async(items)


# ============================================================================
# Async Vector Search Strategy
# ============================================================================


class AsyncVectorSearchStrategy(AsyncSearchStrategy):
    """Async vector similarity search strategy using Valkey Search."""

    async def is_available(self) -> bool:
        """Check if async vector search is available."""
        return (
            self.store.embeddings is not None
            and self.store.dims is not None
            and self.store._is_search_available()
            and self.store.index is not None
        )

    async def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform async vector similarity search."""
        if not op.query:
            return []

        try:
            return await self._execute_vector_search(op)
        except Exception as e:
            logger.error(f"Async vector search failed: {e}")
            raise SearchIndexError(
                f"Async vector search failed: {e}",
                index_name=self.store.collection_name,
                index_operation="search",
            ) from e

    async def _execute_vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute the async vector search query."""
        # Generate query embedding using EmbeddingAdapter
        index_name = self.store.collection_name

        if not self.store.embeddings or not op.query:
            return []

        try:
            adapter = EmbeddingAdapter(self.store.embeddings)
            query_vector = await adapter.embed_query_async(
                op.query, index_name=index_name
            )
            if not query_vector:
                return []
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

        # Build query using BaseSearchHelper
        vector_query, _ = BaseSearchHelper.build_vector_query(
            namespace_prefix=op.namespace_prefix,
            filter_dict=op.filter,
            limit=op.limit,
            offset=op.offset,
        )

        # Convert vector to bytes for Valkey
        vector_bytes = b"".join(struct.pack("<f", x) for x in query_vector)

        # Create query
        from valkey.commands.search.query import Query

        query = (
            Query(vector_query)
            .return_fields("id", "score")
            .paging(op.offset, op.limit)
            .dialect(2)
        )

        search_params: dict[str, str | int | float | bytes] = {"BLOB": vector_bytes}

        # Execute FT.SEARCH in executor (client is sync)
        index_name = self.store.collection_name
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.store.client.ft(index_name).search(query, search_params),
        )

        # Process results using DocumentAdapter
        items = await self._process_results(results, op)

        # Refresh TTL if needed
        await self._refresh_ttl_for_items(items, op)

        return items

    async def _process_results(self, results: Any, op: SearchOp) -> list[SearchItem]:
        """Process vector search results using DocumentAdapter."""
        items: list[SearchItem] = []
        docs = getattr(results, "docs", None)
        if not docs:
            return items

        # Create async client adapter
        adapter = AsyncClientAdapter(self.store.client, self.store)

        # Process each doc using shared DocumentAdapter
        for doc in docs:
            item = await DocumentAdapter.parse_vector_search_doc_async(
                doc, adapter, self.store, apply_filters=op.filter
            )
            if item:
                items.append(item)

        return items


# ============================================================================
# Async Hash Search Strategy
# ============================================================================


class AsyncHashSearchStrategy(AsyncSearchStrategy):
    """Async hash field search strategy."""

    async def is_available(self) -> bool:
        """Check if hash search is available."""
        return self.store._is_search_available() and self.store.index is not None

    async def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform async hash field search."""
        try:
            namespace = op.namespace_prefix or ()
            results = await self.store._search_with_hash_async(
                namespace=namespace,
                query=op.query,
                filter_dict=op.filter,
                limit=op.limit,
                offset=op.offset,
            )

            # Convert results to SearchItems (store method handles this)
            items = await self.store._convert_hash_results_to_search_items_async(
                results
            )

            # Refresh TTL if needed
            await self._refresh_ttl_for_items(items, op)

            return items
        except Exception as e:
            logger.debug(f"Hash search not available: {e}")
            return []


class AsyncKeyPatternSearchStrategy(AsyncSearchStrategy):
    """Async fallback search strategy using key pattern matching and SCAN."""

    async def is_available(self) -> bool:
        """Key pattern search is always available as final fallback."""
        return True

    async def search(self, op: SearchOp) -> list[SearchItem]:
        """Perform async key pattern search using SCAN and DocumentAdapter."""
        try:
            # Build pattern using BaseSearchHelper
            pattern = BaseSearchHelper.build_key_pattern(op.namespace_prefix)

            # Get keys using AsyncClientAdapter
            client_adapter = AsyncClientAdapter(self.store.client, self.store)
            all_keys = await client_adapter.scan_keys(pattern)

            # Sort for consistent ordering
            all_keys.sort()

            # Process keys using DocumentAdapter
            scored_items = []
            for key in all_keys:
                item = await DocumentAdapter.parse_scan_key_async(
                    key=key,
                    client_adapter=client_adapter,
                    store=self.store,
                    query=op.query,
                    apply_filters=op.filter,
                    namespace_prefix=op.namespace_prefix,
                )
                if item:
                    scored_items.append(item)

            # Sort by score descending
            scored_items.sort(key=lambda x: x.score or 0.0, reverse=True)

            # Apply pagination
            start_idx = op.offset or 0
            end_idx = start_idx + (op.limit or 10)
            items = scored_items[start_idx:end_idx]

            # Refresh TTL if needed
            await self._refresh_ttl_for_items(items, op)

            return items

        except Exception as e:
            logger.error(f"Async key pattern search failed: {e}")
            return []


# ============================================================================
# Async Search Strategy Manager
# ============================================================================


class AsyncSearchStrategyManager:
    """Manages async search strategies with fallback chain for AsyncValkeyStore."""

    def __init__(self, store_instance: Any) -> None:
        """Initialize async search strategy manager.

        Args:
            store_instance: Instance of AsyncValkeyStore
        """
        self.store = store_instance
        self.strategies = [
            AsyncVectorSearchStrategy(store_instance),
            AsyncHashSearchStrategy(store_instance),
            AsyncKeyPatternSearchStrategy(store_instance),
        ]

    async def search(self, op: SearchOp) -> list[SearchItem]:
        """Execute search using async strategy chain.

        Strategy chain (in order):
        1. Vector search (if embeddings + query provided)
        2. Hash field search (if index exists)
        3. Key pattern search (fallback)

        Args:
            op: Search operation parameters

        Returns:
            List of matching SearchItem objects
        """
        # Try each strategy in order until one succeeds
        for strategy in self.strategies:
            if await strategy.is_available():
                try:
                    # For vector search, only use if we have a query
                    if isinstance(strategy, AsyncVectorSearchStrategy) and not op.query:
                        continue

                    results = await strategy.search(op)
                    if results:  # If strategy succeeded and returned results
                        return results
                except Exception as e:
                    logger.debug(f"Strategy {type(strategy).__name__} failed: {e}")
                    # Continue to next strategy
                    continue

        # If all strategies failed or returned no results
        return []


__all__ = [
    "AsyncSearchStrategy",
    "AsyncVectorSearchStrategy",
    "AsyncHashSearchStrategy",
    "AsyncKeyPatternSearchStrategy",
    "AsyncSearchStrategyManager",
]
