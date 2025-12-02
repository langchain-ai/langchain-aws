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

    This adapter provides a consistent interface for generating embeddings
    regardless of whether the underlying embeddings object supports sync,
    async, or callable methods. It detects available methods on initialization
    and provides appropriate error handling for each mode.
    """

    def __init__(self, embeddings: Any) -> None:
        """Initialize embedding adapter.

        Args:
            embeddings: Embeddings object (may have sync/async methods or be callable)
        """
        self.embeddings = embeddings
        self._cached_capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> dict[str, bool]:
        """Detect what embedding methods are available.

        Returns:
            Dictionary mapping capability names to availability
        """
        return {
            "sync_query": hasattr(self.embeddings, "embed_query"),
            "sync_documents": hasattr(self.embeddings, "embed_documents"),
            "async_query": hasattr(self.embeddings, "aembed_query"),
            "async_documents": hasattr(self.embeddings, "aembed_documents"),
            "callable": callable(self.embeddings),
        }

    def can_embed_sync(self) -> bool:
        """Check if sync embedding is available.

        Returns:
            True if sync embedding methods are available
        """
        caps = self._cached_capabilities
        return caps["sync_query"] or caps["sync_documents"] or caps["callable"]

    def can_embed_async(self) -> bool:
        """Check if async embedding is available.

        Returns:
            True if async embedding methods are available
        """
        caps = self._cached_capabilities
        return caps["async_query"] or caps["async_documents"] or caps["callable"]

    def embed_query_sync(
        self, query: str, index_name: str | None = None
    ) -> list[float]:
        """Generate embedding synchronously.

        Args:
            query: Text query to embed
            index_name: Optional index name for error messages

        Returns:
            Embedding vector

        Raises:
            SearchIndexError: If sync embedding not available
        """
        caps = self._cached_capabilities

        # Try sync methods in order of preference
        if caps["sync_query"]:
            return self.embeddings.embed_query(query)
        elif caps["sync_documents"]:
            vectors = self.embeddings.embed_documents([query])
            return vectors[0] if vectors else []
        elif caps["callable"]:
            # For mock functions
            vectors = self.embeddings([query])
            return vectors[0] if vectors else []

        # If only async methods available, provide helpful error
        if caps["async_query"] or caps["async_documents"]:
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

        Args:
            query: Text query to embed
            index_name: Optional index name for error messages

        Returns:
            Embedding vector or None if generation fails
        """
        caps = self._cached_capabilities

        try:
            # Try async methods in order of preference
            if caps["async_query"]:
                return await self.embeddings.aembed_query(query)
            elif caps["async_documents"]:
                vectors = await self.embeddings.aembed_documents([query])
                return vectors[0] if vectors else None
            elif caps["callable"]:
                # For mock functions - run in executor
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.embeddings([query])[0]
                )

            # If only sync methods available, provide helpful error
            if caps["sync_query"] or caps["sync_documents"]:
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
            normalized_score = max(0.0, min(1.0, 1.0 - score))

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
            normalized_score = max(0.0, min(1.0, 1.0 - score))

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


__all__ = [
    "EmbeddingAdapter",
    "ClientAdapter",
    "SyncClientAdapter",
    "AsyncClientAdapter",
    "DocumentAdapter",
]
