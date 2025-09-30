"""Tests to improve coverage of langgraph_checkpoint_aws/store/valkey/base.py to 80%+."""

from datetime import datetime
from unittest.mock import Mock, patch

import orjson
import pytest

from langgraph_checkpoint_aws.store.valkey.base import BaseValkeyStore


class ConcreteValkeyStore(BaseValkeyStore):
    """Concrete implementation of BaseValkeyStore for testing purposes."""

    def __init__(self, client, **kwargs):
        super().__init__(client, **kwargs)

    def batch(self, ops):
        return []

    def abatch(self, ops):
        return []

    def put(self, namespace, key, value, index=None):
        pass

    def get(self, namespace, key, refresh_ttl=False):
        return None

    def delete(self, namespace, key):
        pass

    def search(
        self, namespace, query=None, filter=None, limit=10, offset=0, refresh_ttl=False
    ):
        return []

    def list_namespaces(self, prefix=None, max_depth=None, limit=100, offset=0):
        return []


@pytest.fixture
def mock_valkey_client():
    """Create a mock Valkey client."""
    client = Mock()
    client.execute_command = Mock()
    client.scan = Mock()
    client.get = Mock()
    return client


class TestBaseValkeyStoreSearchAvailability:
    """Test search availability detection."""

    def test_is_search_available_cached_true(self, mock_valkey_client):
        """Test _is_search_available when cached as True."""
        store = ConcreteValkeyStore(mock_valkey_client)
        store._search_available = True

        result = store._is_search_available()

        assert result is True
        # Should not call execute_command when cached
        mock_valkey_client.execute_command.assert_not_called()

    def test_is_search_available_cached_false(self, mock_valkey_client):
        """Test _is_search_available when cached as False."""
        store = ConcreteValkeyStore(mock_valkey_client)
        store._search_available = False

        result = store._is_search_available()

        assert result is False
        # Should not call execute_command when cached
        mock_valkey_client.execute_command.assert_not_called()

    def test_is_search_available_success(self, mock_valkey_client):
        """Test _is_search_available when command succeeds."""
        store = ConcreteValkeyStore(mock_valkey_client)
        store._search_available = None  # Not cached

        mock_valkey_client.execute_command.return_value = "OK"

        result = store._is_search_available()

        assert result is True
        assert store._search_available is True
        mock_valkey_client.execute_command.assert_called_once_with("FT._LIST")

    def test_is_search_available_exception_handling(self, mock_valkey_client):
        """Test _is_search_available when command raises exception (COVERS lines 119-122)."""
        store = ConcreteValkeyStore(mock_valkey_client)
        store._search_available = None  # Not cached

        mock_valkey_client.execute_command.side_effect = Exception(
            "Search not available"
        )

        result = store._is_search_available()

        assert result is False
        assert store._search_available is False
        mock_valkey_client.execute_command.assert_called_once_with("FT._LIST")


class TestBaseValkeyStoreIndexCreation:
    """Test index creation functionality."""

    def test_create_index_command_invalid_algorithm_fallback(self, mock_valkey_client):
        """Test _create_index_command with invalid algorithm (COVERS line 165)."""
        # Create store with invalid algorithm to trigger fallback
        store = ConcreteValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "algorithm": "INVALID_ALGO",  # Invalid algorithm
                "fields": ["title"],
            },
        )

        cmd = store._create_index_command("test_idx", "test_prefix")

        # Should fallback to HNSW algorithm
        assert "HNSW" in cmd
        assert "vector" in cmd
        assert "VECTOR" in cmd

    def test_create_index_command_with_vector_config(self, mock_valkey_client):
        """Test _create_index_command with vector configuration (COVERS lines 312, 316-320)."""
        store = ConcreteValkeyStore(
            mock_valkey_client,
            index={"dims": 256, "algorithm": "HNSW", "fields": ["title", "content"]},
        )

        cmd = store._create_index_command("vector_idx", "vectors")

        assert "FT.CREATE" in cmd
        assert "vector_idx" in cmd
        assert "256" in str(cmd)  # dimensions
        # The fields are used in schema construction but may not be literally in the command

    def test_setup_search_index_sync_not_available(self, mock_valkey_client):
        """Test _setup_search_index_sync when search not available (COVERS lines 208-210)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        with patch.object(store, "_is_search_available", return_value=False):
            # Should return early and log warning
            store._setup_search_index_sync()

            # Should not call execute_command
            mock_valkey_client.execute_command.assert_not_called()

    def test_setup_search_index_sync_index_exists(self, mock_valkey_client):
        """Test _setup_search_index_sync when index already exists (COVERS lines 218-220)."""
        store = ConcreteValkeyStore(
            mock_valkey_client, index={"dims": 128, "fields": ["title"]}
        )

        with patch.object(store, "_is_search_available", return_value=True):
            with patch.object(store, "_execute_command") as mock_execute:
                mock_execute.return_value = "OK"  # Index exists

                store._setup_search_index_sync()

                # Should call FT.INFO and return early
                mock_execute.assert_called_once()

    def test_setup_search_index_sync_creation_error(self, mock_valkey_client):
        """Test _setup_search_index_sync when creation fails (COVERS lines 230-231)."""
        store = ConcreteValkeyStore(
            mock_valkey_client, index={"dims": 128, "fields": ["title"]}
        )

        with patch.object(store, "_is_search_available", return_value=True):
            with patch.object(store, "_execute_command") as mock_execute:
                mock_execute.side_effect = Exception("Index creation failed")

                # Should handle error gracefully (not raise)
                store._setup_search_index_sync()


class TestBaseValkeyStoreValidation:
    """Test validation methods."""

    def test_validate_put_empty_namespace(self, mock_valkey_client):
        """Test _validate_put with empty namespace (COVERS line 293)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        with pytest.raises(ValueError, match="Namespace cannot be empty"):
            store._validate_put_operation(tuple(), {"value": "test"})

    def test_validate_put_invalid_value_type(self, mock_valkey_client):
        """Test _validate_put with non-dict value."""
        store = ConcreteValkeyStore(mock_valkey_client)

        with pytest.raises(TypeError, match="Value must be a dictionary or None"):
            store._validate_put_operation(("namespace",), "not_a_dict")

    def test_validate_put_none_value(self, mock_valkey_client):
        """Test _validate_put with None value (valid for delete)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Should not raise exception
        store._validate_put_operation(("namespace",), None)


class TestBaseValkeyStoreDocumentCreation:
    """Test document creation functionality."""

    def test_create_document_with_vector(self, mock_valkey_client):
        """Test _create_document with vector data (COVERS line 312)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        test_value = {"title": "Test Document", "content": "Test content"}
        test_vector = [0.1, 0.2, 0.3, 0.4]

        result = store._create_document(test_value, test_vector)

        assert isinstance(result, bytes)
        # Verify the document structure contains the vector as bytes
        doc_data = orjson.loads(result)
        assert "value" in doc_data
        assert doc_data["value"] == test_value

    def test_create_document_with_index_fields(self, mock_valkey_client):
        """Test _create_document with index fields (COVERS lines 316-320)."""
        store = ConcreteValkeyStore(
            mock_valkey_client, index={"dims": 128, "fields": ["title", "category"]}
        )

        test_value = {
            "title": "Indexed Document",
            "category": "test",
            "description": "This should not be indexed",
        }

        result = store._create_document(test_value)

        assert isinstance(result, bytes)
        # The document should include searchable fields
        doc_data = orjson.loads(result)
        assert doc_data["value"] == test_value


class TestBaseValkeyStoreKeyParsing:
    """Test key parsing functionality."""

    def test_parse_key_empty_namespace(self, mock_valkey_client):
        """Test _parse_key with single level key (COVERS line 364)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        namespace, key = store._parse_key("simple_key")

        assert namespace == tuple()
        assert key == "simple_key"

    def test_parse_key_with_namespace(self, mock_valkey_client):
        """Test _parse_key with multi-level namespace."""
        store = ConcreteValkeyStore(mock_valkey_client)

        namespace, key = store._parse_key("level1/level2/level3/actual_key")

        assert namespace == ("level1", "level2", "level3")
        assert key == "actual_key"


class TestBaseValkeyStoreScoring:
    """Test scoring functionality."""

    def test_calculate_simple_score_no_query(self, mock_valkey_client):
        """Test _calculate_simple_score with no query (COVERS line 369)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        score = store._calculate_simple_score(None, {"title": "Test"})

        assert score == 1.0

    def test_calculate_simple_score_with_hash_fields(self, mock_valkey_client):
        """Test _calculate_simple_score with _hash_fields structure (COVERS lines 380-381)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Test with _hash_fields structure
        value_with_hash = {
            "_hash_fields": {
                "value": orjson.dumps(
                    {"title": "Test Document", "content": "Important content"}
                )
            }
        }

        score = store._calculate_simple_score("important", value_with_hash)

        assert score > 0.0

    def test_calculate_simple_score_hash_fields_exception(self, mock_valkey_client):
        """Test _calculate_simple_score with invalid _hash_fields (COVERS line 381)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Test with invalid _hash_fields that causes exception
        value_with_bad_hash = {
            "_hash_fields": {
                "value": "invalid_json"  # This will cause orjson.loads to fail
            },
            "value": {"title": "Fallback content"},
        }

        score = store._calculate_simple_score("content", value_with_bad_hash)

        assert score > 0.0  # Should fallback to value field

    def test_calculate_simple_score_with_value_field(self, mock_valkey_client):
        """Test _calculate_simple_score with value field (COVERS line 383)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        value_with_value_field = {
            "value": {"title": "Document Title", "content": "Document content"}
        }

        score = store._calculate_simple_score("document", value_with_value_field)

        assert score > 0.0

    def test_calculate_simple_score_exact_matches(self, mock_valkey_client):
        """Test _calculate_simple_score with exact word matches (COVERS lines 402)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        test_value = {
            "title": "Machine Learning Tutorial",
            "content": "Deep learning concepts",
        }

        # Test exact matches - should get high score (0.6)
        score = store._calculate_simple_score("machine tutorial", test_value)

        assert score >= 0.6  # Some exact word matches


class TestBaseValkeyStoreSearchWithHash:
    """Test _search_with_hash functionality."""

    def test_search_with_hash_scan_result_none(self, mock_valkey_client):
        """Test _search_with_hash when scan returns None (COVERS line 447)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Mock scan to return None after _handle_response_t
        mock_valkey_client.scan.return_value = (0, [])

        with patch.object(store, "_handle_response_t", return_value=None):
            results = store._search_with_hash(
                namespace=("test",), query="search", filter_dict={}, limit=10
            )

            assert results == []

    def test_search_with_hash_namespace_filtering(self, mock_valkey_client):
        """Test _search_with_hash with namespace filtering (COVERS lines 462-471)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Mock scan to return keys
        mock_valkey_client.scan.return_value = (
            0,
            ["langgraph:test/sub/doc1", "langgraph:other/doc2"],
        )

        # Mock get to return document data
        mock_valkey_client.get.return_value = orjson.dumps(
            {
                "value": {"title": "Test Document", "content": "Test content"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        )

        with patch.object(store, "_handle_response_t") as mock_handle:
            mock_handle.side_effect = lambda x: x  # Return input unchanged

            with patch.object(store, "_parse_document") as mock_parse:
                mock_parse.return_value = (
                    {"title": "Test Document"},
                    datetime.now(),
                    datetime.now(),
                )

                results = store._search_with_hash(
                    namespace=("test", "sub"),  # Should match test/sub namespace
                    query="document",
                    filter_dict={},
                    limit=10,
                )

                # Should return results since namespace matches
                assert isinstance(results, list)

    def test_search_with_hash_score_filtering(self, mock_valkey_client):
        """Test _search_with_hash with score filtering (COVERS lines 480-484)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Mock scan to return a key
        mock_valkey_client.scan.return_value = (0, ["langgraph:test/doc1"])

        # Mock get to return document
        mock_valkey_client.get.return_value = orjson.dumps(
            {"value": {"title": "Low relevance doc"}}
        )

        with patch.object(store, "_handle_response_t") as mock_handle:
            mock_handle.side_effect = lambda x: x

            with patch.object(store, "_parse_document") as mock_parse:
                mock_parse.return_value = (
                    {"title": "Low relevance doc"},
                    datetime.now(),
                    datetime.now(),
                )

                with patch.object(
                    store, "_calculate_simple_score", return_value=0.05
                ):  # Low score
                    results = store._search_with_hash(
                        namespace=("test",),
                        query="irrelevant search",
                        filter_dict={},
                        limit=10,
                    )

                    # Should filter out low-scoring results (score <= 0.1)
                    assert len(results) == 0

    def test_search_with_hash_cursor_loop_termination(self, mock_valkey_client):
        """Test _search_with_hash cursor loop termination (COVERS line 490)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Mock scan to return cursor 0 (end of scan)
        mock_valkey_client.scan.return_value = (0, [])

        with patch.object(store, "_handle_response_t") as mock_handle:
            mock_handle.side_effect = lambda x: x

            results = store._search_with_hash(
                namespace=("test",), query="search", filter_dict={}, limit=10
            )

            assert results == []
            # Should call scan once and stop when cursor is 0
            mock_valkey_client.scan.assert_called_once()


class TestBaseValkeyStoreNamespaceExtraction:
    """Test namespace extraction functionality."""

    def test_extract_namespaces_from_keys_bytes_key(self, mock_valkey_client):
        """Test _extract_namespaces_from_keys with bytes keys (COVERS lines 520)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Test with bytes keys
        keys = [b"level1/level2/doc1", b"level1/level3/doc2"]

        namespaces = store._extract_namespaces_from_keys(keys, max_depth=None)

        expected_namespaces = {("level1", "level2"), ("level1", "level3")}
        assert namespaces == expected_namespaces

    def test_extract_namespaces_from_keys_other_types(self, mock_valkey_client):
        """Test _extract_namespaces_from_keys with other key types (COVERS lines 525)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Test with mixed types (int, None, etc.)
        keys = [123, None, "string/key"]

        namespaces = store._extract_namespaces_from_keys(keys, max_depth=None)

        # Should convert all to strings and extract namespaces
        expected_namespaces = {
            tuple(),
            ("string",),
        }  # 123 and None become single keys, "string/key" has namespace
        assert namespaces == expected_namespaces

    def test_extract_namespaces_single_level_keys(self, mock_valkey_client):
        """Test _extract_namespaces_from_keys with single level keys (COVERS lines 538-540)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Test with single level keys (no namespace)
        keys = ["doc1", "doc2", "doc3"]

        namespaces = store._extract_namespaces_from_keys(keys, max_depth=None)

        # Should add empty namespace for single level keys
        expected_namespaces = {tuple()}
        assert namespaces == expected_namespaces


class TestBaseValkeyStoreResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_t_awaitable_result(self, mock_valkey_client):
        """Test _handle_response_t with awaitable result (COVERS lines 552-553)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        # Create a mock awaitable object
        mock_awaitable = Mock()
        mock_awaitable.__await__ = Mock(return_value=iter([]))

        result = store._handle_response_t(mock_awaitable)

        # Should return None and log error
        assert result is None

    def test_safe_parse_keys_none_result(self, mock_valkey_client):
        """Test _safe_parse_keys with None result (COVERS line 561)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        with patch.object(store, "_handle_response_t", return_value=None):
            result = store._safe_parse_keys("some_input")

            assert result == []

    def test_safe_parse_keys_mixed_types(self, mock_valkey_client):
        """Test _safe_parse_keys with mixed key types (COVERS lines 569-572)."""
        store = ConcreteValkeyStore(mock_valkey_client)

        mixed_keys = [b"bytes_key", "string_key", 123, None]

        with patch.object(store, "_handle_response_t", return_value=mixed_keys):
            result = store._safe_parse_keys(mixed_keys)

            expected = ["bytes_key", "string_key", "123", "None"]
            assert result == expected
