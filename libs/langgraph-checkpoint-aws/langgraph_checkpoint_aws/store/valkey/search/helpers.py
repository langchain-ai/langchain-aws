"""Search helper utilities for ValkeyStore."""

from __future__ import annotations

import logging
from typing import Any

from ..constants import LANGGRAPH_KEY_PREFIX
from ..document_utils import ScoreCalculator

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Search Utilities
# ============================================================================


class BaseSearchHelper:
    """Base helper class providing shared utilities for search strategies."""

    @staticmethod
    def build_vector_query(
        namespace_prefix: tuple[str, ...] | None,
        filter_dict: dict[str, Any] | None,
        limit: int,
        offset: int,
    ) -> tuple[str, str]:
        """Build FT.SEARCH query string for vector search.

        Args:
            namespace_prefix: Namespace prefix to filter by
            filter_dict: Additional filters to apply
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Tuple of (base_query, tag_query)
        """
        filter_parts = []

        # Add namespace filter
        if namespace_prefix:
            namespace_prefix_str = "/".join(namespace_prefix)
            filter_parts.append(f"@namespace:{{{namespace_prefix_str}*}}")

        # Add field filters
        if filter_dict:
            for key, value in filter_dict.items():
                # Escape special characters in filter values
                escaped_value = str(value).replace(":", "\\:")
                filter_parts.append(f"@{key}:{{{escaped_value}}}")

        # Combine filters
        filter_expr = " ".join(filter_parts) if filter_parts else "*"

        # Base query is for vector search
        base_query = f"({filter_expr})=>[KNN {limit + offset} @vector $vec AS score]"

        return base_query, filter_expr

    @staticmethod
    def build_key_pattern(namespace_prefix: tuple[str, ...] | None) -> str:
        """Build Redis key pattern for SCAN operations.

        Args:
            namespace_prefix: Namespace prefix to filter by

        Returns:
            Redis key pattern string
        """
        if namespace_prefix:
            namespace_str = "/".join(namespace_prefix)
            return f"langgraph:{namespace_str}/*"
        return "langgraph:*"

    @staticmethod
    def parse_doc_id(doc_id: str) -> tuple[tuple[str, ...], str] | None:
        """Parse document ID into namespace and key.

        Args:
            doc_id: Full document ID (e.g., 'langgraph:test/docs/key1')

        Returns:
            Tuple of (namespace_tuple, key) or None if invalid
        """
        if not doc_id.startswith(f"{LANGGRAPH_KEY_PREFIX}:"):
            return None

        # Remove prefix (e.g., 'langgraph:')
        prefix_len = len(LANGGRAPH_KEY_PREFIX) + 1  # +1 for ':'
        key_path = doc_id[prefix_len:]

        parts = key_path.rsplit("/", 1)
        if len(parts) == 2:
            namespace_str, key = parts
            namespace = tuple(namespace_str.split("/"))
            return namespace, key

        return None

    @staticmethod
    def calculate_text_score(query: str | None, doc_data: dict[str, Any]) -> float:
        """Calculate text similarity score for a document.

        Args:
            query: Search query text
            doc_data: Document data

        Returns:
            Similarity score (0.0-1.0)
        """
        return ScoreCalculator.calculate_text_similarity_score(query, doc_data)


__all__ = ["BaseSearchHelper"]
