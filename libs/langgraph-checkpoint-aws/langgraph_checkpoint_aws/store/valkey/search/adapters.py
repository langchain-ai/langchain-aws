"""Adapter classes for ValkeyStore search operations.

This module provides adapter classes that abstract sync/async operations
and provide unified interfaces for embedding generation, client operations,
and document processing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.store.base import SearchItem

from ..constants import MIN_SEARCH_SCORE
from ..document_utils import DocumentProcessor, FilterProcessor
from ..exceptions import SearchIndexError
from .helpers import BaseSearchHelper

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Adapter
# ============================================================================


class EmbeddingAdapter:
    """Unified adapter for sync/async embedding generation.

    Provides a consistent interface for generating embeddings from various
    embedding objects (LangChain embeddings, callables, functions).

    Note: ValkeyStore/AsyncValkeyStore ensure embeddings are valid via
    ensure_embeddings() before creating this adapter. This adapter focuses
    on routing to the correct method and providing helpful error messages
    when sync/async mismatch occurs.
    """

    def __init__(self, embeddings: Any) -> None:
        """Initialize embedding adapter.

        Args:
            embeddings: Embeddings object (may have sync/async methods or be callable)
        """
        self.embeddings = embeddings

    def embed_query_sync(
        self, query: str, index_name: str | None = None
    ) -> list[float]:
        """Generate embedding synchronously.

        Tries methods in order: embed_query → embed_documents → callable.
        Provides helpful error if only async methods available.

        Args:
            query: Text query to embed
            index_name: Optional index name for error messages

        Returns:
            Embedding vector

        Raises:
            SearchIndexError: If sync embedding not available
        """
        # Try sync methods in order
        if hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(query)
        elif hasattr(self.embeddings, "embed_documents"):
            vectors = self.embeddings.embed_documents([query])
            return vectors[0] if vectors else []
        elif callable(self.embeddings):
            vectors = self.embeddings([query])
            return vectors[0] if vectors else []

        # Only async methods available - provide helpful error
        if hasattr(self.embeddings, "aembed_query") or hasattr(
            self.embeddings, "aembed_documents"
        ):
            raise SearchIndexError(
                "Cannot generate embeddings: embeddings object only has async methods. "
                "Use AsyncValkeyStore for async embedding generation.",
                index_name=index_name or "unknown",
                index_operation="embedding_generation",
            )

        raise SearchIndexError(
            "No embedding method available",
            index_name=index_name or "unknown",
            index_operation="embedding_generation",
        )

    async def embed_query_async(
        self, query: str, index_name: str | None = None
    ) -> list[float] | None:
        """Generate embedding asynchronously.

        Tries methods in order: aembed_query → aembed_documents → callable.
        Provides helpful error if only sync methods available.

        Args:
            query: Text query to embed
            index_name: Optional index name for error messages

        Returns:
            Embedding vector or None if generation fails
        """
        try:
            # Try async methods in order
            if hasattr(self.embeddings, "aembed_query"):
                return await self.embeddings.aembed_query(query)
            elif hasattr(self.embeddings, "aembed_documents"):
                vectors = await self.embeddings.aembed_documents([query])
                return vectors[0] if vectors else None
            elif callable(self.embeddings):
                # Run callable in executor to avoid blocking event loop
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.embeddings([query])[0]
                )

            # Only sync methods available - provide helpful error
            if hasattr(self.embeddings, "embed_query") or hasattr(
                self.embeddings, "embed_documents"
            ):
                raise SearchIndexError(
                    "Cannot generate embeddings: embeddings object only has "
                    "sync methods. Use ValkeyStore for sync embedding generation "
                    "to avoid blocking the event loop.",
                    index_name=index_name or "unknown",
                    index_operation="embedding_generation",
                )

            logger.warning("No embedding method available")
            return None

        except Exception as e:
            # Re-raise SearchIndexError to provide clear guidance
            if isinstance(e, SearchIndexError):
                raise
            logger.error(f"Error generating query embedding: {e}")
            return None


# ============================================================================
# Client Adapters
# ============================================================================


class ClientAdapter:
    """Base adapter for Valkey client operations.

    This adapter provides a common interface for both sync and async
    client operations, allowing strategies to work with either type
    of client without knowing the implementation details.
    """

    def __init__(self, client: Any, store: Any):
        """Initialize client adapter.

        Args:
            client: Valkey client instance (sync or async)
            store: Store instance (for accessing configuration)
        """
        self.client = client
        self.store = store


class SyncClientAdapter(ClientAdapter):
    """Adapter for synchronous Valkey client operations."""

    def get_document(self, key: str) -> dict[str, Any] | None:
        """Get a document by key.

        Args:
            key: Document key

        Returns:
            Document data or None if not found
        """
        try:
            result = self.client.hgetall(key)
            if not result:
                return None
            return self.store._handle_response_t(result)
        except Exception as e:
            logger.debug(f"Error getting document {key}: {e}")
            return None

    def scan_keys(self, pattern: str, batch_size: int = 1000) -> list[str]:
        """Scan for keys matching pattern.

        Args:
            pattern: Key pattern to match
            batch_size: Batch size for SCAN operation

        Returns:
            List of matching keys
        """
        keys = []
        cursor = 0

        try:
            while True:
                cursor, batch = self.client.scan(
                    cursor, match=pattern, count=batch_size
                )
                if batch:
                    # Use store's safe parse method
                    parsed = self.store._safe_parse_keys(batch)
                    keys.extend(parsed)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Error scanning keys with pattern {pattern}: {e}")

        return keys


class AsyncClientAdapter(ClientAdapter):
    """Adapter for asynchronous Valkey client operations."""

    async def get_document(self, key: str) -> dict[str, Any] | None:
        """Get a document by key asynchronously.

        Args:
            key: Document key

        Returns:
            Document data or None if not found
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.hgetall, key
            )
            if not result:
                return None
            return await self.store._handle_response_t_async(result)
        except Exception as e:
            logger.debug(f"Error getting document {key}: {e}")
            return None

    async def scan_keys(self, pattern: str, batch_size: int = 1000) -> list[str]:
        """Scan for keys matching pattern asynchronously.

        Args:
            pattern: Key pattern to match
            batch_size: Batch size for SCAN operation

        Returns:
            List of matching keys
        """
        keys = []
        cursor = 0

        try:
            while True:
                cursor, batch = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda c: self.client.scan(c, match=pattern, count=batch_size),
                    cursor,
                )

                if batch:
                    # Parse keys manually since async store doesn't have
                    # _safe_parse_keys helper method
                    parsed = []
                    if hasattr(batch, "__iter__") and not isinstance(
                        batch, (str, bytes)
                    ):
                        for key in batch:
                            if isinstance(key, bytes):
                                parsed.append(key.decode("utf-8"))
                            elif isinstance(key, str):
                                parsed.append(key)
                            else:
                                parsed.append(str(key))
                    keys.extend(parsed)

                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Error scanning keys with pattern {pattern}: {e}")

        return keys


# ============================================================================
# Document Adapter
# ============================================================================


class DocumentAdapter:
    """Adapter for document parsing, filtering, and scoring operations.

    This adapter provides shared logic for processing search results,
    applicable to both sync and async search strategies.
    """

    @staticmethod
    def parse_vector_search_doc_sync(
        doc: Any,
        client_adapter: SyncClientAdapter,
        store: Any,
        apply_filters: dict[str, Any] | None = None,
    ) -> SearchItem | None:
        """Parse a vector search result document (sync version).

        Args:
            doc: Document from FT.SEARCH results
            client_adapter: Client adapter for fetching full document
            store: Store instance
            apply_filters: Optional filters to apply

        Returns:
            SearchItem or None if parsing fails
        """
        try:
            # Extract document ID and score
            doc_id = doc.id
            score = float(doc.score) if hasattr(doc, "score") else 1.0

            # Parse namespace and key from doc_id
            parsed = BaseSearchHelper.parse_doc_id(doc_id)
            if not parsed:
                return None

            namespace, key = parsed

            # Get full document data
            full_key = store._build_key(namespace, key)
            doc_data = client_adapter.get_document(full_key)

            if not doc_data or not isinstance(doc_data, dict):
                return None

            # Apply filters if provided
            if apply_filters:
                # Extract value from hash fields
                value_json = doc_data.get("value")
                if not value_json:
                    return None

                try:
                    value_data = DocumentProcessor.parse_document_value(
                        {"value": value_json}
                    )
                    if not FilterProcessor.apply_filters(value_data, apply_filters):
                        return None
                except Exception:
                    return None

            # Convert hash fields to document
            document = DocumentProcessor.convert_hash_to_document(doc_data)
            if document is None:
                return None

            # Parse value
            parsed_value = DocumentProcessor.parse_document_value(document)
            if parsed_value is None:
                return None

            # Parse timestamps
            created_at, updated_at = DocumentProcessor.parse_timestamps(document)

            # Normalize score to 0-1 range
            normalized_score = max(0.0, min(1.0, score))

            return SearchItem(
                namespace=namespace,
                key=key,
                value=parsed_value,
                created_at=created_at,
                updated_at=updated_at,
                score=normalized_score,
            )

        except Exception as e:
            logger.debug(f"Error parsing vector search doc: {e}")
            return None

    @staticmethod
    async def parse_vector_search_doc_async(
        doc: Any,
        client_adapter: AsyncClientAdapter,
        store: Any,
        apply_filters: dict[str, Any] | None = None,
    ) -> SearchItem | None:
        """Parse a vector search result document (async version).

        Args:
            doc: Document from FT.SEARCH results
            client_adapter: Async client adapter for fetching full document
            store: Store instance
            apply_filters: Optional filters to apply

        Returns:
            SearchItem or None if parsing fails
        """
        try:
            # Extract document ID and score
            doc_id = doc.id
            score = float(doc.score) if hasattr(doc, "score") else 1.0

            # Parse namespace and key from doc_id
            parsed = BaseSearchHelper.parse_doc_id(doc_id)
            if not parsed:
                return None

            namespace, key = parsed

            # Get full document data
            full_key = store._build_key(namespace, key)
            doc_data = await client_adapter.get_document(full_key)

            if not doc_data or not isinstance(doc_data, dict):
                return None

            # Apply filters if provided
            if apply_filters:
                value_json = doc_data.get("value")
                if not value_json:
                    return None

                try:
                    value_data = DocumentProcessor.parse_document_value(
                        {"value": value_json}
                    )
                    if not FilterProcessor.apply_filters(value_data, apply_filters):
                        return None
                except Exception:
                    return None

            # Convert hash fields to document
            document = DocumentProcessor.convert_hash_to_document(doc_data)
            if document is None:
                return None

            # Parse value
            parsed_value = DocumentProcessor.parse_document_value(document)
            if parsed_value is None:
                return None

            # Parse timestamps
            created_at, updated_at = DocumentProcessor.parse_timestamps(document)

            # Normalize score
            normalized_score = max(0.0, min(1.0, score))

            return SearchItem(
                namespace=namespace,
                key=key,
                value=parsed_value,
                created_at=created_at,
                updated_at=updated_at,
                score=normalized_score,
            )

        except Exception as e:
            logger.debug(f"Error parsing vector search doc: {e}")
            return None

    @staticmethod
    def parse_scan_key_sync(
        key: str,
        client_adapter: SyncClientAdapter,
        store: Any,
        query: str | None = None,
        apply_filters: dict[str, Any] | None = None,
        namespace_prefix: tuple[str, ...] | None = None,
    ) -> SearchItem | None:
        """Parse a key from SCAN results (sync version).

        Args:
            key: Key from SCAN results
            client_adapter: Client adapter for fetching document
            store: Store instance
            query: Optional query for text scoring
            apply_filters: Optional filters to apply
            namespace_prefix: Expected namespace prefix

        Returns:
            SearchItem or None if parsing fails or filters don't match
        """
        try:
            # Parse namespace and key
            parsed = BaseSearchHelper.parse_doc_id(key)
            if not parsed:
                return None

            namespace, item_key = parsed

            # Check namespace prefix
            if namespace_prefix:
                if not namespace[: len(namespace_prefix)] == namespace_prefix:
                    return None

            # Get document data
            doc_data = client_adapter.get_document(key)
            if not doc_data or not isinstance(doc_data, dict):
                return None

            # Convert hash to document
            document = DocumentProcessor.convert_hash_to_document(doc_data)
            if document is None:
                return None

            # Parse value
            parsed_value = DocumentProcessor.parse_document_value(document)
            if parsed_value is None:
                return None

            # Apply filters
            if apply_filters:
                if not FilterProcessor.apply_filters(parsed_value, apply_filters):
                    return None

            # Calculate score
            score = BaseSearchHelper.calculate_text_score(
                query, {"value": parsed_value}
            )
            if score < MIN_SEARCH_SCORE:
                return None

            # Parse timestamps
            created_at, updated_at = DocumentProcessor.parse_timestamps(document)

            return SearchItem(
                namespace=namespace,
                key=item_key,
                value=parsed_value,
                created_at=created_at,
                updated_at=updated_at,
                score=score,
            )

        except Exception as e:
            logger.debug(f"Error parsing scan key {key}: {e}")
            return None

    @staticmethod
    async def parse_scan_key_async(
        key: str,
        client_adapter: AsyncClientAdapter,
        store: Any,
        query: str | None = None,
        apply_filters: dict[str, Any] | None = None,
        namespace_prefix: tuple[str, ...] | None = None,
    ) -> SearchItem | None:
        """Parse a key from SCAN results (async version).

        Args:
            key: Key from SCAN results
            client_adapter: Async client adapter for fetching document
            store: Store instance
            query: Optional query for text scoring
            apply_filters: Optional filters to apply
            namespace_prefix: Expected namespace prefix

        Returns:
            SearchItem or None if parsing fails or filters don't match
        """
        try:
            # Parse namespace and key
            parsed = BaseSearchHelper.parse_doc_id(key)
            if not parsed:
                return None

            namespace, item_key = parsed

            # Check namespace prefix
            if namespace_prefix:
                if not namespace[: len(namespace_prefix)] == namespace_prefix:
                    return None

            # Get document data
            doc_data = await client_adapter.get_document(key)
            if not doc_data or not isinstance(doc_data, dict):
                return None

            # Convert hash to document
            document = DocumentProcessor.convert_hash_to_document(doc_data)
            if document is None:
                return None

            # Parse value
            parsed_value = DocumentProcessor.parse_document_value(document)
            if parsed_value is None:
                return None

            # Apply filters
            if apply_filters:
                if not FilterProcessor.apply_filters(parsed_value, apply_filters):
                    return None

            # Calculate score
            score = BaseSearchHelper.calculate_text_score(
                query, {"value": parsed_value}
            )
            if score < MIN_SEARCH_SCORE:
                return None

            # Parse timestamps
            created_at, updated_at = DocumentProcessor.parse_timestamps(document)

            return SearchItem(
                namespace=namespace,
                key=item_key,
                value=parsed_value,
                created_at=created_at,
                updated_at=updated_at,
                score=score,
            )

        except Exception as e:
            logger.debug(f"Error parsing scan key {key}: {e}")
            return None


# ============================================================================
# Hash Search Adapter
# ============================================================================


class HashSearchAdapter:
    """Adapter for hash-based search operations shared between sync and async.

    This adapter provides hash search functionality when vector search is unavailable,
    using SCAN operations with text-based scoring.
    """

    @staticmethod
    def scan_and_score_keys_sync(
        client: Any,
        store: Any,
        namespace: tuple[str, ...],
        query: str | None = None,
        filter_dict: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[tuple[tuple[str, ...], str, float]]:
        """Scan keys and calculate scores synchronously.

        Args:
            client: Valkey client instance
            store: Store instance
            namespace: Namespace prefix to search in
            query: Optional text query for scoring
            filter_dict: Optional filters to apply
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of (namespace, key, score) tuples
        """
        from ..constants import LANGGRAPH_KEY_PREFIX
        from ..document_utils import FilterProcessor, ScoreCalculator

        # Build scan pattern
        if namespace:
            pattern = f"{LANGGRAPH_KEY_PREFIX}:{'/'.join(namespace)}/*"
        else:
            pattern = f"{LANGGRAPH_KEY_PREFIX}:*"

        # SCAN for keys
        cursor = 0
        results = []
        seen_keys = set()

        while True:
            scan_result = client.scan(cursor, match=pattern, count=1000)
            scan_result = store._handle_response_t(scan_result)
            if scan_result is None:
                break

            cursor, keys = scan_result
            keys = store._safe_parse_keys(keys)

            for key in keys:
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                try:
                    # Parse namespace and key
                    parsed_namespace, item_key = store._parse_key(
                        key, f"{LANGGRAPH_KEY_PREFIX}:"
                    )

                    # Check namespace prefix
                    if (
                        namespace
                        and not parsed_namespace[: len(namespace)] == namespace
                    ):
                        continue

                    # Get document value
                    value = client.get(key)
                    value = store._handle_response_t(value)
                    if not value:
                        continue

                    import orjson

                    doc_data = orjson.loads(value)

                    # Apply filters
                    if filter_dict and not FilterProcessor.apply_filters(
                        doc_data.get("value", {}), filter_dict
                    ):
                        continue

                    # Calculate score
                    score = ScoreCalculator.calculate_text_similarity_score(
                        query, doc_data
                    )
                    if score > MIN_SEARCH_SCORE:
                        results.append((parsed_namespace, item_key, score))

                except Exception as e:
                    logger.debug(f"Error processing key {key}: {e}")
                    continue

            if cursor == 0:
                break

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

        # Apply pagination
        start_idx = offset
        end_idx = start_idx + limit if limit else len(sorted_results)
        return sorted_results[start_idx:end_idx]

    @staticmethod
    async def scan_and_score_keys_async(
        client: Any,
        store: Any,
        namespace: tuple[str, ...],
        query: str | None = None,
        filter_dict: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[tuple[tuple[str, ...], str, float]]:
        """Scan keys and calculate scores asynchronously.

        Args:
            client: Valkey client instance
            store: Store instance
            namespace: Namespace prefix to search in
            query: Optional text query for scoring
            filter_dict: Optional filters to apply
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of (namespace, key, score) tuples
        """
        from ..constants import LANGGRAPH_KEY_PREFIX
        from ..document_utils import FilterProcessor, ScoreCalculator

        # Build scan pattern
        if namespace:
            pattern = f"{LANGGRAPH_KEY_PREFIX}:{'/'.join(namespace)}/*"
        else:
            pattern = f"{LANGGRAPH_KEY_PREFIX}:*"

        # SCAN for keys
        cursor = 0
        results = []
        seen_keys = set()

        while True:
            scan_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda c: client.scan(c, match=pattern, count=1000), cursor
            )
            if scan_result is None:
                break

            cursor, keys = scan_result

            # Parse keys to strings
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
                    # Get document value
                    value = await asyncio.get_event_loop().run_in_executor(
                        None, client.get, key
                    )
                    if not value:
                        continue

                    value = await store._handle_response_t_async(value)
                    if not value:
                        continue

                    import orjson

                    doc_data = orjson.loads(value)

                    # Apply filters
                    if filter_dict and not FilterProcessor.apply_filters(
                        doc_data.get("value", {}), filter_dict
                    ):
                        continue

                    # Calculate score
                    score = ScoreCalculator.calculate_text_similarity_score(
                        query, doc_data
                    )
                    if score > MIN_SEARCH_SCORE:
                        namespace_parsed, key_parsed = store._parse_key(
                            key, f"{LANGGRAPH_KEY_PREFIX}:"
                        )
                        results.append((namespace_parsed, key_parsed, score))

                except Exception as e:
                    logger.debug(f"Error processing key {key}: {e}")
                    continue

            if cursor == 0:
                break

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

        # Apply pagination
        start_idx = offset
        end_idx = start_idx + limit if limit else len(sorted_results)
        return sorted_results[start_idx:end_idx]

    @staticmethod
    def convert_hash_results_to_items_sync(
        client: Any,
        store: Any,
        results: list[tuple[tuple[str, ...], str, float]],
    ) -> list[SearchItem]:
        """Convert hash search results to SearchItem objects synchronously.

        Args:
            client: Valkey client instance
            store: Store instance
            results: List of (namespace, key, score) tuples

        Returns:
            List of SearchItem objects
        """

        items = []
        for namespace, key, score in results:
            try:
                # Get full document data using GET (hash search stores as JSON)
                full_key = store._build_key(namespace, key)
                value_data = client.get(full_key)
                if not value_data:
                    continue

                value_data = store._handle_response_t(value_data)
                if not value_data:
                    continue

                # Parse JSON document
                from datetime import datetime

                import orjson

                parsed_data = orjson.loads(value_data)
                parsed_value = parsed_data.get("value", {})

                # Parse timestamps
                created_at = datetime.fromisoformat(
                    parsed_data.get("created_at", datetime.now().isoformat())
                )
                updated_at = datetime.fromisoformat(
                    parsed_data.get("updated_at", datetime.now().isoformat())
                )

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

    @staticmethod
    async def convert_hash_results_to_items_async(
        client: Any,
        store: Any,
        results: list[tuple[tuple[str, ...], str, float]],
    ) -> list[SearchItem]:
        """Convert hash search results to SearchItem objects asynchronously.

        Args:
            client: Valkey client instance
            store: Store instance
            results: List of (namespace, key, score) tuples

        Returns:
            List of SearchItem objects
        """
        items = []
        for namespace, key, score in results:
            try:
                # Get full document data
                full_key = store._build_key(namespace, key)
                value_data = await asyncio.get_event_loop().run_in_executor(
                    None, client.get, full_key
                )
                if not value_data:
                    continue

                value_data = await store._handle_response_t_async(value_data)
                if not value_data:
                    continue

                # Parse document
                from datetime import datetime

                import orjson

                parsed_data = orjson.loads(value_data)
                value = parsed_data.get("value", {})
                created_at = datetime.fromisoformat(
                    parsed_data.get("created_at", datetime.now().isoformat())
                )
                updated_at = datetime.fromisoformat(
                    parsed_data.get("updated_at", datetime.now().isoformat())
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
                logger.debug(f"Error converting result {namespace}/{key}: {e}")
                continue
        return items


__all__ = [
    "EmbeddingAdapter",
    "ClientAdapter",
    "SyncClientAdapter",
    "AsyncClientAdapter",
    "DocumentAdapter",
    "HashSearchAdapter",
]
