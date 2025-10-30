"""Document processing utilities for Valkey store."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import orjson

from .constants import (
    HASH_FIELD_CREATED_AT,
    HASH_FIELD_UPDATED_AT,
    HASH_FIELD_VALUE,
    HASH_FIELD_VECTOR,
    HIGH_SCORE_THRESHOLD,
    LOW_SCORE_THRESHOLD,
    MEDIUM_SCORE_THRESHOLD,
    MIN_SEARCH_SCORE,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document conversion and parsing operations."""

    @staticmethod
    def convert_hash_to_document(hash_data: dict[str, Any]) -> dict[str, Any] | None:
        """Convert hash fields back to document format.

        Args:
            hash_data: Raw hash data from Valkey

        Returns:
            Parsed document or None if parsing fails
        """
        if not hash_data:
            return None

        try:
            # Extract and decode hash fields - handle both bytes and string keys
            def get_field_value(data: dict[Any, Any], field_name: str) -> Any:
                """Get field value handling both string and bytes keys."""
                # Try string key first
                if field_name in data:
                    return data[field_name]
                # Try bytes key
                bytes_key = field_name.encode("utf-8")
                if bytes_key in data:
                    return data[bytes_key]
                return None

            document = {
                HASH_FIELD_VALUE: get_field_value(hash_data, "value"),
                HASH_FIELD_CREATED_AT: get_field_value(hash_data, "created_at"),
                HASH_FIELD_UPDATED_AT: get_field_value(hash_data, "updated_at"),
                HASH_FIELD_VECTOR: get_field_value(hash_data, "vector"),
            }

            # Handle bytes keys/values
            for key_name, value in document.items():
                if isinstance(value, bytes):
                    document[key_name] = value.decode("utf-8")

            return document
        except Exception as e:
            logger.error(f"Error converting hash to document: {e}")
            return None

    @staticmethod
    def parse_document_value(document: dict[str, Any]) -> Any:
        """Parse the JSON-encoded value from document.

        Args:
            document: Document with encoded value field

        Returns:
            Parsed value or None if parsing fails
        """
        try:
            if document.get(HASH_FIELD_VALUE):
                return orjson.loads(document[HASH_FIELD_VALUE])
            return None
        except (orjson.JSONDecodeError, TypeError) as e:
            logger.debug(f"Error parsing document value: {e}")
            return document.get(HASH_FIELD_VALUE)

    @staticmethod
    def parse_timestamps(document: dict[str, Any]) -> tuple[datetime, datetime]:
        """Parse created_at and updated_at timestamps from document.

        Args:
            document: Document with timestamp fields

        Returns:
            Tuple of (created_at, updated_at) datetimes
        """
        now = datetime.now()

        try:
            created_at = (
                datetime.fromisoformat(document[HASH_FIELD_CREATED_AT])
                if document.get(HASH_FIELD_CREATED_AT)
                else now
            )
        except (ValueError, TypeError):
            created_at = now

        try:
            updated_at = (
                datetime.fromisoformat(document[HASH_FIELD_UPDATED_AT])
                if document.get(HASH_FIELD_UPDATED_AT)
                else now
            )
        except (ValueError, TypeError):
            updated_at = now

        return created_at, updated_at

    @staticmethod
    def create_hash_fields(
        value: dict[str, Any],
        vector: list[float] | None = None,
        index_fields: list[str] | None = None,
    ) -> dict[str, str]:
        """Create hash fields for storage.

        Args:
            value: Document value to store
            vector: Optional vector for search
            index_fields: Fields to index for search

        Returns:
            Dictionary of hash fields ready for storage
        """
        now = datetime.now()
        hash_fields = {
            HASH_FIELD_VALUE: orjson.dumps(value).decode("utf-8"),
            HASH_FIELD_CREATED_AT: now.isoformat(),
            HASH_FIELD_UPDATED_AT: now.isoformat(),
        }

        if vector is not None:
            hash_fields[HASH_FIELD_VECTOR] = orjson.dumps(vector).decode("utf-8")

        # Add searchable fields based on index configuration
        if index_fields:
            for field in index_fields:
                if field != "$":  # Skip root field
                    field_value = value.get(field)
                    if field_value is not None:
                        # Convert lists and complex types to TAG-compatible strings
                        if isinstance(field_value, list):
                            hash_fields[field] = ",".join(
                                str(item) for item in field_value
                            )
                        else:
                            hash_fields[field] = str(field_value)

        return hash_fields


class ScoreCalculator:
    """Handles score calculation for search results."""

    @staticmethod
    def calculate_text_similarity_score(
        query: str | None, value: dict[str, Any]
    ) -> float:
        """Calculate a text-based relevance score.

        Args:
            query: Search query string
            value: Document value to score

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not query:
            return 1.0

        query_lower = query.lower()

        # Extract searchable text from value
        searchable_text = ScoreCalculator._extract_searchable_text(value)

        # Calculate score based on match quality
        query_words = query_lower.split()
        text_words = searchable_text.split()
        exact_matches = sum(1 for word in query_words if word in text_words)

        if exact_matches == len(query_words):
            return HIGH_SCORE_THRESHOLD
        elif exact_matches > 0:
            return MEDIUM_SCORE_THRESHOLD
        elif query_lower in searchable_text:
            return LOW_SCORE_THRESHOLD
        else:
            return MIN_SEARCH_SCORE

    @staticmethod
    def _extract_searchable_text(value: dict[str, Any]) -> str:
        """Extract searchable text from document value.

        Args:
            value: Document value

        Returns:
            Concatenated searchable text
        """
        # Handle nested document structure
        value_data = value
        if isinstance(value, dict) and "_hash_fields" in value:
            try:
                hash_fields = value["_hash_fields"]
                value_data = orjson.loads(hash_fields["value"])
            except Exception:
                value_data = value.get("value", value)
        elif isinstance(value, dict) and "value" in value:
            value_data = value["value"]

        # Convert value to searchable text
        if isinstance(value_data, dict):
            return " ".join(str(v) for v in value_data.values()).lower()
        else:
            return str(value_data).lower()


class FilterProcessor:
    """Handles filter operations for search queries."""

    @staticmethod
    def apply_filters(
        value: dict[str, Any], filter_dict: dict[str, Any] | None
    ) -> bool:
        """Apply filter conditions to a document value.

        Args:
            value: Document value to filter
            filter_dict: Filter conditions to apply

        Returns:
            True if value passes all filters, False otherwise
        """
        if not filter_dict:
            return True

        for filter_key, filter_value in filter_dict.items():
            if filter_key not in value or value[filter_key] != filter_value:
                return False
        return True

    @staticmethod
    def build_namespace_pattern(namespace_prefix: tuple[str, ...]) -> str:
        """Build a key pattern for namespace filtering.

        Args:
            namespace_prefix: Namespace prefix tuple

        Returns:
            Pattern string for key matching
        """
        from .constants import LANGGRAPH_KEY_PREFIX

        if namespace_prefix:
            namespace_path = "/".join(namespace_prefix)
            return f"{LANGGRAPH_KEY_PREFIX}:{namespace_path}/*"
        else:
            return f"{LANGGRAPH_KEY_PREFIX}:*"

    @staticmethod
    def matches_namespace_prefix(
        namespace: tuple[str, ...], prefix: tuple[str, ...]
    ) -> bool:
        """Check if namespace matches the given prefix.

        Args:
            namespace: Full namespace tuple
            prefix: Prefix to match against

        Returns:
            True if namespace starts with prefix, False otherwise
        """
        if len(namespace) < len(prefix):
            return False
        return namespace[: len(prefix)] == prefix
