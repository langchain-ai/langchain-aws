"""Unit tests for BaseSearchHelper utility class."""

import pytest

# Skip entire module if valkey not available
pytest.importorskip("valkey")

from langgraph_checkpoint_aws.store.valkey.search import BaseSearchHelper


class TestBaseSearchHelper:
    """Test suite for BaseSearchHelper utility methods."""

    def test_build_vector_query_with_namespace_and_filters(self) -> None:
        """Test building vector query with namespace prefix and filters."""
        query, filters = BaseSearchHelper.build_vector_query(
            namespace_prefix=("test", "docs"),
            filter_dict={"category": "tech", "status": "active"},
            limit=10,
            offset=0,
        )

        # Check query format
        assert "=>[KNN 10 @vector $vec AS score]" in query
        assert "@namespace:{test/docs*}" in filters
        assert "@category:{tech}" in filters
        assert "@status:{active}" in filters

    def test_build_vector_query_no_namespace(self) -> None:
        """Test building vector query without namespace prefix."""
        query, filters = BaseSearchHelper.build_vector_query(
            namespace_prefix=None,
            filter_dict={"category": "tech"},
            limit=5,
            offset=2,
        )

        # Should use wildcard for namespace
        assert "@namespace" not in filters
        assert "@category:{tech}" in filters
        assert "=>[KNN 7 @vector $vec AS score]" in query  # limit + offset = 7

    def test_build_vector_query_no_filters(self) -> None:
        """Test building vector query with no filters."""
        query, filters = BaseSearchHelper.build_vector_query(
            namespace_prefix=("test",),
            filter_dict=None,
            limit=10,
            offset=0,
        )

        # Only namespace should be present
        assert "@namespace:{test*}" in filters
        assert "@category" not in filters

    def test_build_vector_query_escapes_special_chars(self) -> None:
        """Test that special characters in filter values are escaped."""
        query, filters = BaseSearchHelper.build_vector_query(
            namespace_prefix=None,
            filter_dict={"url": "http://example.com"},
            limit=10,
            offset=0,
        )

        # Colons should be escaped
        assert (
            "@url:{http\\://example.com}" in filters
            or "@url:{http\\\\://example.com}" in filters
        )

    def test_build_key_pattern_with_namespace(self) -> None:
        """Test building key pattern with namespace prefix."""
        pattern = BaseSearchHelper.build_key_pattern(("test", "docs"))
        assert pattern == "langgraph:test/docs/*"

    def test_build_key_pattern_without_namespace(self) -> None:
        """Test building key pattern without namespace prefix."""
        pattern = BaseSearchHelper.build_key_pattern(None)
        assert pattern == "langgraph:*"

    def test_build_key_pattern_single_level(self) -> None:
        """Test building key pattern with single-level namespace."""
        pattern = BaseSearchHelper.build_key_pattern(("test",))
        assert pattern == "langgraph:test/*"

    def test_parse_doc_id_valid(self) -> None:
        """Test parsing valid document ID."""
        result = BaseSearchHelper.parse_doc_id("langgraph:test/docs/mykey")
        assert result is not None
        namespace, key = result
        assert namespace == ("test", "docs")
        assert key == "mykey"

    def test_parse_doc_id_single_namespace(self) -> None:
        """Test parsing doc ID with single-level namespace."""
        result = BaseSearchHelper.parse_doc_id("langgraph:test/key1")
        assert result is not None
        namespace, key = result
        assert namespace == ("test",)
        assert key == "key1"

    def test_parse_doc_id_invalid_prefix(self) -> None:
        """Test parsing doc ID with invalid prefix."""
        result = BaseSearchHelper.parse_doc_id("invalid:test/docs/key")
        assert result is None

    def test_parse_doc_id_no_key(self) -> None:
        """Test parsing doc ID without a key part."""
        result = BaseSearchHelper.parse_doc_id("langgraph:test")
        # Should return None since there's no "/" to split namespace and key
        assert result is None

    def test_parse_doc_id_empty(self) -> None:
        """Test parsing empty doc ID."""
        result = BaseSearchHelper.parse_doc_id("")
        assert result is None

    def test_calculate_text_score_with_match(self) -> None:
        """Test text score calculation with exact word match."""
        doc_data = {
            "value": {
                "title": "Python Programming",
                "description": "Learn Python basics",
            }
        }
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # ScoreCalculator returns HIGH_SCORE_THRESHOLD (0.8) for exact match
        assert score == 0.8

    def test_calculate_text_score_case_insensitive(self) -> None:
        """Test that text matching is case-insensitive."""
        doc_data = {
            "value": {
                "title": "PYTHON GUIDE",
                "text": "python programming",
            }
        }
        score = BaseSearchHelper.calculate_text_score("Python", doc_data)
        # Should match both regardless of case - HIGH_SCORE_THRESHOLD
        assert score == 0.8

    def test_calculate_text_score_no_match(self) -> None:
        """Test text score calculation with no match."""
        doc_data = {
            "value": {
                "title": "JavaScript Guide",
                "text": "Learn JS",
            }
        }
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # ScoreCalculator returns MIN_SEARCH_SCORE (0.1) for no match
        assert score == 0.1

    def test_calculate_text_score_no_query(self) -> None:
        """Test text score calculation with no query."""
        doc_data = {"value": {"title": "Some Title"}}
        score = BaseSearchHelper.calculate_text_score(None, doc_data)
        # No query should return default score
        assert score == 1.0

    def test_calculate_text_score_empty_value(self) -> None:
        """Test text score calculation with empty document."""
        doc_data: dict[str, str] = {}
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # ScoreCalculator returns MIN_SEARCH_SCORE (0.1) for no match
        assert score == 0.1

    def test_calculate_text_score_non_dict_value(self) -> None:
        """Test text score calculation with non-dict value."""
        doc_data = {"value": "not a dict"}
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # Should handle gracefully and return MIN_SEARCH_SCORE (0.1)
        assert score == 0.1

    def test_calculate_text_score_non_string_field(self) -> None:
        """Test text score calculation with non-string fields."""
        doc_data = {
            "value": {
                "title": "Python Guide",
                "count": 42,  # non-string field
                "active": True,  # non-string field
            }
        }
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # ScoreCalculator handles exact word matches - HIGH_SCORE_THRESHOLD
        assert score == 0.8

    def test_calculate_text_score_partial_match(self) -> None:
        """Test that partial substring matches work."""
        doc_data = {
            "text": "This is about pythonic programming",
        }
        score = BaseSearchHelper.calculate_text_score("python", doc_data)
        # "python" is a substring of "pythonic" - LOW_SCORE_THRESHOLD
        assert score == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
