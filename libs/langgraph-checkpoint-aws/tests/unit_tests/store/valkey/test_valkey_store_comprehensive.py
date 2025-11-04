"""Comprehensive tests for ValkeyStore to improve code coverage."""

import pytest
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
)

from langgraph_checkpoint_aws import (
    ValkeyEmbeddingGenerationError,
    ValkeyIndexConfig,
    ValkeyStore,
    ValkeyValidationError,
)

# Check for optional dependencies
try:
    import orjson  # noqa: F401
    import valkey  # noqa: F401
    from valkey.exceptions import ValkeyError  # noqa: F401

    VALKEY_AVAILABLE = True
except ImportError:
    # Create dummy objects for type checking when dependencies not available
    class MockOrjson:
        @staticmethod
        def dumps(obj):  # type: ignore[misc]
            import json

            return json.dumps(obj).encode("utf-8")

    orjson = MockOrjson()  # type: ignore[assignment]
    ValkeyError = Exception  # type: ignore[assignment, misc]
    VALKEY_AVAILABLE = False

# Skip all tests if valkey dependencies are not available
pytestmark = pytest.mark.skipif(
    not VALKEY_AVAILABLE,
    reason=(
        "valkey dependencies not available. "
        "Install with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ),
)

# Import after optional dependency check
if VALKEY_AVAILABLE:
    import json
    from datetime import datetime
    from unittest.mock import MagicMock, Mock, patch


@pytest.fixture
def mock_valkey_client():
    """Create a mock sync Valkey client."""
    client = MagicMock()
    # Remove async attributes to ensure it's detected as sync
    delattr_list = ["aclose", "__aenter__", "__aexit__", "aset", "aget", "ahgetall"]
    for attr in delattr_list:
        if hasattr(client, attr):
            try:
                delattr(client, attr)
            except AttributeError:
                pass

    # Set up common mock returns
    client.ping.return_value = True
    client.hgetall.return_value = {}
    client.hset.return_value = 1
    client.delete.return_value = 1
    client.expire.return_value = True
    client.scan.return_value = (0, [])
    client.keys.return_value = []
    client.ft.return_value = Mock()
    client.ft().search.return_value = Mock(total=0, docs=[])
    return client


@pytest.fixture
def basic_index_config():
    """Basic index configuration for testing."""
    return ValkeyIndexConfig(
        collection_name="test_collection",
        dims=128,
        embed="fake_embeddings",
        fields=["title", "content"],
    )


class TestValkeyStoreInitialization:
    """Test ValkeyStore initialization and configuration."""

    def test_init_with_index_config(self, mock_valkey_client, basic_index_config):
        """Test initialization with index configuration."""
        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings"
        ) as mock_ensure:
            mock_embeddings = MagicMock()
            mock_ensure.return_value = mock_embeddings

            store = ValkeyStore(mock_valkey_client, index=basic_index_config)

            assert store.client == mock_valkey_client
            assert store.index == basic_index_config
            assert store.collection_name == "test_collection"
            assert store.dims == 128
            assert store.index_fields == ["title", "content"]
            assert store.embeddings == mock_embeddings

    def test_init_without_index_config(self, mock_valkey_client):
        """Test initialization without index configuration."""
        store = ValkeyStore(mock_valkey_client)

        assert store.client == mock_valkey_client
        assert store.index is None
        assert store.embeddings is None
        assert store.index_fields is None
        assert store.collection_name == "langgraph_store_idx"  # Default value

    def test_init_with_ttl_config(self, mock_valkey_client):
        """Test initialization with TTL configuration."""
        ttl_config = TTLConfig(default_ttl=3600, refresh_on_read=True)
        store = ValkeyStore(mock_valkey_client, ttl=ttl_config)

        assert store.ttl_config == ttl_config

    def test_init_with_invalid_embeddings(self, mock_valkey_client, basic_index_config):
        """Test initialization with invalid embeddings configuration."""
        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            side_effect=Exception("Invalid embeddings"),
        ):
            with pytest.raises(ValkeyEmbeddingGenerationError):
                ValkeyStore(mock_valkey_client, index=basic_index_config)


class TestValkeyStoreSetup:
    """Test ValkeyStore setup methods."""

    def test_setup_without_index(self, mock_valkey_client):
        """Test setup without index configuration."""
        store = ValkeyStore(mock_valkey_client)
        store.setup()  # Should not raise any errors

    def test_setup_with_index_search_unavailable(
        self, mock_valkey_client, basic_index_config
    ):
        """Test setup when search module is unavailable."""
        mock_valkey_client.execute_command.side_effect = Exception(
            "Search not available"
        )

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=MagicMock(),
        ):
            store = ValkeyStore(mock_valkey_client, index=basic_index_config)
            # Should not raise, just log warning
            store.setup()

    def test_setup_with_existing_index(self, mock_valkey_client, basic_index_config):
        """Test setup when index already exists."""
        # First call succeeds (index exists), second call should not be made
        mock_valkey_client.execute_command.return_value = {
            "index_name": "test_collection"
        }

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=MagicMock(),
        ):
            store = ValkeyStore(mock_valkey_client, index=basic_index_config)
            store.setup()

            # Should call FT.INFO to check if index exists
            mock_valkey_client.execute_command.assert_called()

    def test_setup_creates_new_index(self, mock_valkey_client, basic_index_config):
        """Test setup creates a new search index."""
        # First call fails (index doesn't exist), second call succeeds (creates index)
        mock_valkey_client.execute_command.side_effect = [
            Exception("Index not found"),  # FT.INFO fails
            "OK",  # FT.CREATE succeeds
        ]

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=MagicMock(),
        ):
            store = ValkeyStore(mock_valkey_client, index=basic_index_config)
            store.setup()

            assert mock_valkey_client.execute_command.call_count >= 2

    def test_setup_search_index_method_coverage(self, mock_valkey_client):
        """Test _setup_search_index method for coverage."""
        store = ValkeyStore(mock_valkey_client)

        with patch.object(store, "_setup_search_index_sync") as mock_setup_sync:
            result = store._setup_search_index()

            mock_setup_sync.assert_called_once()
            assert result == mock_setup_sync.return_value


class TestValkeyStoreContextManagers:
    """Test ValkeyStore context manager methods."""

    def test_from_conn_string_context_manager(self):
        """Test from_conn_string context manager."""
        with patch("valkey.Valkey.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.close = Mock()
            mock_from_url.return_value = mock_client

            with ValkeyStore.from_conn_string("redis://localhost:6379") as store:
                assert isinstance(store, ValkeyStore)
                assert store.client == mock_client

    def test_from_conn_string_with_pool_parameters(self):
        """Test from_conn_string with pool_size and pool_timeout parameters."""
        with (
            patch("valkey.connection.ConnectionPool.from_url") as mock_pool_from_url,
            patch("valkey.Valkey") as mock_valkey_class,
        ):
            mock_pool = Mock()
            mock_client = Mock()
            mock_pool_from_url.return_value = mock_pool
            mock_valkey_class.return_value = mock_client
            mock_client.close = Mock()

            # Test with pool parameters - this should trigger the pool creation path
            with ValkeyStore.from_conn_string(
                "valkey://localhost:6379", pool_size=10, pool_timeout=30.0
            ) as store:
                assert isinstance(store, ValkeyStore)
                mock_pool_from_url.assert_called_once()

    def test_from_conn_string_without_pool_parameters(self):
        """Test from_conn_string without pool parameters."""
        with patch("valkey.Valkey.from_url") as mock_from_url:
            mock_client = Mock()
            mock_client.close = Mock()
            mock_from_url.return_value = mock_client

            # Test without pool parameters - this should trigger direct client creation
            with ValkeyStore.from_conn_string("valkey://localhost:6379") as store:
                assert isinstance(store, ValkeyStore)
                mock_from_url.assert_called_once()

    def test_from_pool_context_manager(self):
        """Test from_pool context manager."""
        with patch("valkey.Valkey.from_pool") as mock_from_pool:
            mock_pool = Mock()
            mock_client = Mock()
            mock_client.close = Mock()
            mock_from_pool.return_value = mock_client

            with ValkeyStore.from_pool(mock_pool) as store:
                assert isinstance(store, ValkeyStore)
                assert store.client == mock_client
                mock_from_pool.assert_called_once()


class TestValkeyStoreBatchOperations:
    """Test ValkeyStore batch operations."""

    def test_batch_get_operation(self, mock_valkey_client):
        """Test batch with GetOp."""
        mock_valkey_client.hgetall.return_value = {
            "value": '{"title": "test"}',
            "created_at": "2023-01-01T00:00:00.000000",
            "updated_at": "2023-01-01T00:00:00.000000",
        }

        store = ValkeyStore(mock_valkey_client)
        ops = [GetOp(namespace=("test",), key="key1")]

        results = store.batch(ops)

        assert len(results) == 1
        assert isinstance(results[0], Item)
        assert results[0].key == "key1"
        assert results[0].namespace == ("test",)

    def test_batch_put_operation(self, mock_valkey_client):
        """Test batch with PutOp."""
        mock_valkey_client.hset.return_value = 1

        store = ValkeyStore(mock_valkey_client)
        ops = [PutOp(namespace=("test",), key="key1", value={"title": "test"})]

        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None  # PutOp returns None
        mock_valkey_client.hset.assert_called_once()

    def test_batch_search_operation(self, mock_valkey_client):
        """Test batch with SearchOp."""
        # Mock search results
        mock_valkey_client.scan.return_value = (0, ["langgraph:test/key1"])
        mock_valkey_client.get.return_value = (
            '{"value": {"title": "test"}, '
            '"created_at": "2023-01-01T00:00:00.000000", '
            '"updated_at": "2023-01-01T00:00:00.000000"}'
        )

        store = ValkeyStore(mock_valkey_client)
        ops = [SearchOp(namespace_prefix=("test",), query="test")]

        results = store.batch(ops)

        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_batch_list_namespaces_operation(self, mock_valkey_client):
        """Test batch with ListNamespacesOp."""
        mock_valkey_client.keys.return_value = [
            "langgraph:test/key1",
            "langgraph:test/key2",
        ]

        store = ValkeyStore(mock_valkey_client)
        ops = [ListNamespacesOp()]

        results = store.batch(ops)

        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_batch_unknown_operation(self, mock_valkey_client):
        """Test batch with unknown operation type."""
        store = ValkeyStore(mock_valkey_client)

        class UnknownOp:
            pass

        ops = [UnknownOp()]  # type: ignore

        with pytest.raises(ValueError, match="Unknown operation type"):
            store.batch(ops)  # type: ignore


class TestValkeyStoreGetOperations:
    """Test ValkeyStore get operations."""

    def test_handle_get_success(self, mock_valkey_client):
        """Test successful get operation."""
        # Mock hash data
        hash_data = {
            "value": '{"title": "test"}',
            "created_at": "2023-01-01T00:00:00.000000",
            "updated_at": "2023-01-01T00:00:00.000000",
        }
        mock_valkey_client.hgetall.return_value = hash_data

        store = ValkeyStore(mock_valkey_client)
        op = GetOp(namespace=("test",), key="key1")

        result = store._handle_get(op)

        assert isinstance(result, Item)
        assert result.key == "key1"
        assert result.namespace == ("test",)
        assert result.value == {"title": "test"}

    def test_handle_get_not_found(self, mock_valkey_client):
        """Test get operation when key not found."""
        mock_valkey_client.hgetall.return_value = {}

        store = ValkeyStore(mock_valkey_client)
        op = GetOp(namespace=("test",), key="key1")

        result = store._handle_get(op)

        assert result is None

    def test_handle_get_with_ttl_refresh(self, mock_valkey_client):
        """Test get operation with TTL refresh."""
        hash_data = {
            "value": '{"title": "test"}',
            "created_at": "2023-01-01T00:00:00.000000",
            "updated_at": "2023-01-01T00:00:00.000000",
        }
        mock_valkey_client.hgetall.return_value = hash_data
        mock_valkey_client.expire.return_value = True

        ttl_config = TTLConfig(default_ttl=60, refresh_on_read=True)
        store = ValkeyStore(mock_valkey_client, ttl=ttl_config)
        op = GetOp(namespace=("test",), key="key1", refresh_ttl=True)

        result = store._handle_get(op)

        assert result is not None
        mock_valkey_client.expire.assert_called_once()

    def test_handle_get_parse_error(self, mock_valkey_client):
        """Test get operation with document parsing error."""
        # Return invalid JSON
        hash_data = {"value": "invalid json"}
        mock_valkey_client.hgetall.return_value = hash_data

        store = ValkeyStore(mock_valkey_client)
        op = GetOp(namespace=("test",), key="key1")

        result = store._handle_get(op)

        # Implementation actually returns the invalid value as is
        assert result is not None  # ValkeyStore doesn't validate JSON during get

    def test_handle_get_with_response_t_none(self, mock_valkey_client):
        """Test _handle_get when _handle_response_t returns None."""
        store = ValkeyStore(mock_valkey_client)

        # Mock hgetall to return data, but _handle_response_t to return None
        mock_valkey_client.hgetall.return_value = {"some": "data"}

        with patch.object(store, "_handle_response_t", return_value=None):
            op = GetOp(namespace=("test",), key="key1")
            result = store._handle_get(op)

            assert result is None

    def test_handle_get_document_processor_returns_none(self, mock_valkey_client):
        """Test _handle_get when DocumentProcessor methods return None."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.hgetall.return_value = {"value": "test"}

        # Mock DocumentProcessor methods to return None - fix import path to base module
        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.DocumentProcessor"
        ) as mock_dp:
            mock_dp.convert_hash_to_document.return_value = None

            op = GetOp(namespace=("test",), key="key1")
            result = store._handle_get(op)

            assert result is None


class TestValkeyStorePutOperations:
    """Test ValkeyStore put operations."""

    def test_handle_put_success(self, mock_valkey_client):
        """Test successful put operation."""
        mock_valkey_client.hset.return_value = 1

        store = ValkeyStore(mock_valkey_client)
        op = PutOp(namespace=("test",), key="key1", value={"title": "test"})

        store._handle_put(op)

        mock_valkey_client.hset.assert_called_once()

    def test_handle_put_with_ttl(self, mock_valkey_client):
        """Test put operation with TTL."""
        mock_valkey_client.hset.return_value = 1
        mock_valkey_client.expire.return_value = True

        store = ValkeyStore(mock_valkey_client)
        op = PutOp(namespace=("test",), key="key1", value={"title": "test"}, ttl=60)

        store._handle_put(op)

        mock_valkey_client.hset.assert_called_once()
        mock_valkey_client.expire.assert_called_once()

    def test_handle_put_delete_operation(self, mock_valkey_client):
        """Test put operation with None value (delete)."""
        mock_valkey_client.delete.return_value = 1

        store = ValkeyStore(mock_valkey_client)
        op = PutOp(namespace=("test",), key="key1", value=None)

        store._handle_put(op)

        mock_valkey_client.delete.assert_called_once()

    def test_handle_put_with_embeddings_sync_method(self, mock_valkey_client):
        """Test _handle_put with embeddings that have sync embed_documents method."""
        # Create mock embeddings with sync method
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title"]

        op = PutOp(namespace=("test",), key="key1", value={"title": "test content"})

        store._handle_put(op)

        # Verify sync embedding method was called
        mock_embeddings.embed_documents.assert_called_once()
        mock_valkey_client.hset.assert_called_once()

    def test_handle_put_with_embeddings_async_method_no_loop(self, mock_valkey_client):
        """Test _handle_put with async embeddings when no event loop is running."""
        # Create mock embeddings with only async method
        mock_embeddings = Mock()
        del mock_embeddings.embed_documents  # Remove sync method

        async def mock_aembed_documents(texts):
            return [[0.1, 0.2, 0.3]]

        mock_embeddings.aembed_documents = mock_aembed_documents

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title"]

        op = PutOp(namespace=("test",), key="key1", value={"title": "test content"})

        # Mock asyncio.run to simulate successful async embedding
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = [[0.1, 0.2, 0.3]]

            store._handle_put(op)

            mock_run.assert_called_once()
            mock_valkey_client.hset.assert_called_once()

    def test_handle_put_with_embeddings_in_async_context(self, mock_valkey_client):
        """Test _handle_put with embeddings when already in async context."""
        mock_embeddings = Mock()
        del mock_embeddings.embed_documents  # Remove sync method

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title"]

        op = PutOp(namespace=("test",), key="key1", value={"title": "test content"})

        # Mock asyncio.get_running_loop to simulate being in async context
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.return_value = Mock()  # Simulate running loop

            store._handle_put(op)

            # Should skip embeddings and log warning
            mock_valkey_client.hset.assert_called_once()

    def test_handle_put_embedding_generation_error(self, mock_valkey_client):
        """Test _handle_put when embedding generation fails."""
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.side_effect = Exception("Embedding failed")

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title"]

        op = PutOp(namespace=("test",), key="key1", value={"title": "test content"})

        # Should handle embedding error gracefully
        store._handle_put(op)

        mock_valkey_client.hset.assert_called_once()

    def test_handle_put_with_list_field_values(self, mock_valkey_client):
        """Test _handle_put with list values in indexed fields."""
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["tags"]
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["tags"]

        # Value with list field
        op = PutOp(
            namespace=("test",), key="key1", value={"tags": ["tag1", "tag2", "tag3"]}
        )

        store._handle_put(op)

        # Should handle list values by extending texts
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        # The implementation converts list to string, so check for individual tags
        call_text = " ".join(call_args)
        assert "tag1" in call_text
        assert "tag2" in call_text
        assert "tag3" in call_text

    def test_handle_put_with_empty_field_values(self, mock_valkey_client):
        """Test _handle_put with empty field values."""
        mock_embeddings = Mock()

        index_config = ValkeyIndexConfig(
            collection_name="test",
            dims=3,
            embed=mock_embeddings,
            fields=["title", "content"],
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title", "content"]

        # Value with completely missing fields (no title or content)
        op = PutOp(namespace=("test",), key="key1", value={"other": "data"})

        store._handle_put(op)

        # Should not call embeddings when no valid text found
        mock_embeddings.embed_documents.assert_not_called()

    def test_handle_put_validation_error(self, mock_valkey_client):
        """Test put operation with validation error."""
        store = ValkeyStore(mock_valkey_client)

        # Empty namespace should trigger validation error
        with pytest.raises(ValkeyValidationError):
            op = PutOp(namespace=(), key="key1", value={"title": "test"})
            store._handle_put(op)

    def test_handle_put_hset_error(self, mock_valkey_client):
        """Test _handle_put when hset operation fails."""
        store = ValkeyStore(mock_valkey_client)

        # Mock hset to raise exception
        mock_valkey_client.hset.side_effect = Exception("HSET failed")

        op = PutOp(namespace=("test",), key="key1", value={"title": "test"})

        with pytest.raises(Exception, match="HSET failed"):
            store._handle_put(op)

    def test_handle_put_delete_error_handling(self, mock_valkey_client):
        """Test _handle_put delete operation error handling."""
        store = ValkeyStore(mock_valkey_client)

        # Mock delete to raise exception
        mock_valkey_client.delete.side_effect = Exception("Delete failed")

        op = PutOp(namespace=("test",), key="key1", value=None)  # Delete operation

        # Should handle delete errors gracefully (logged but not raised)
        store._handle_put(op)


class TestValkeyStoreSearchOperations:
    """Test ValkeyStore search operations."""

    def test_handle_search_key_pattern(self, mock_valkey_client):
        """Test search operation using key pattern strategy."""
        # Mock scan results
        mock_valkey_client.scan.return_value = (0, ["langgraph:test/key1"])
        mock_valkey_client.get.return_value = (
            '{"value": {"title": "test"}, '
            '"created_at": "2023-01-01T00:00:00.000000", '
            '"updated_at": "2023-01-01T00:00:00.000000"}'
        )

        store = ValkeyStore(mock_valkey_client)
        op = SearchOp(namespace_prefix=("test",), query="test")

        results = store._handle_search(op)

        assert isinstance(results, list)

    def test_handle_search_with_vector_search(self, mock_valkey_client):
        """Test search operation with vector search (when available)."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=mock_embeddings,
        ):
            # Mock search being available
            mock_valkey_client.execute_command.return_value = [
                1,  # Total results
                "langgraph:test/key1",  # Document ID
                ["score", "0.9", "value", '{"title": "test"}'],  # Document fields
            ]

            store = ValkeyStore(mock_valkey_client, index=index_config)

            # Mock search availability
            with patch.object(store, "_is_search_available", return_value=True):
                op = SearchOp(namespace_prefix=("test",), query="test")
                results = store._handle_search(op)

                assert isinstance(results, list)

    def test_vector_search_with_namespace_and_filters(self, mock_valkey_client):
        """Test _vector_search with both namespace prefix and filters."""
        mock_embeddings = Mock()

        index_config = ValkeyIndexConfig(
            collection_name="test_index",
            dims=3,
            embed=mock_embeddings,
            fields=["title"],
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings
        store.index_fields = ["title"]

        # Mock search results
        mock_result = Mock()
        mock_result.docs = []
        mock_valkey_client.ft.return_value.search.return_value = mock_result

        op = SearchOp(
            namespace_prefix=("test", "public"),
            query="test query",
            filter={"type": "document", "status": "active"},
            limit=10,
            offset=0,
        )

        results = store._vector_search(op)

        assert isinstance(results, list)
        # Verify search was called with proper query construction
        mock_valkey_client.ft.assert_called_with("test_index")

    def test_vector_search_pure_vector_no_filters(self, mock_valkey_client):
        """Test _vector_search with pure vector search (no filters)."""
        mock_embeddings = Mock()

        index_config = ValkeyIndexConfig(
            collection_name="test_index",
            dims=3,
            embed=mock_embeddings,
            fields=["title"],
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings

        mock_result = Mock()
        mock_result.docs = []
        mock_valkey_client.ft.return_value.search.return_value = mock_result

        op = SearchOp(
            namespace_prefix=(),  # No namespace filter
            query="test query",
            filter=None,  # No additional filters
            limit=10,
            offset=0,
        )

        results = store._vector_search(op)

        assert isinstance(results, list)

    def test_vector_search_error_handling(self, mock_valkey_client):
        """Test _vector_search error handling."""
        mock_embeddings = Mock()

        index_config = ValkeyIndexConfig(
            collection_name="test_index",
            dims=3,
            embed=mock_embeddings,
            fields=["title"],
        )

        store = ValkeyStore(mock_valkey_client, index=index_config)
        store.embeddings = mock_embeddings

        # Mock search to raise exception
        mock_valkey_client.ft.return_value.search.side_effect = Exception(
            "Search failed"
        )

        op = SearchOp(
            namespace_prefix=("test",), query="test query", limit=10, offset=0
        )

        results = store._vector_search(op)

        # Should return empty list on error
        assert results == []

    def test_process_vector_search_results_with_offset(self, mock_valkey_client):
        """Test _process_vector_search_results with offset."""
        store = ValkeyStore(mock_valkey_client)

        # Create mock docs
        mock_docs = []
        for i in range(5):
            doc = Mock()
            doc.id = f"langgraph:test/doc{i}"
            doc.score = 0.9 - (i * 0.1)
            doc.__dict__ = {"id": doc.id, "score": doc.score}
            mock_docs.append(doc)

        mock_results = Mock()
        mock_results.docs = mock_docs

        # Mock hgetall to return valid data
        def mock_hgetall(key):
            return {
                "value": json.dumps({"title": f"Doc {key}"}),
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        op = SearchOp(
            namespace_prefix=("test",),
            query="test",
            limit=10,
            offset=2,  # Skip first 2 results
        )

        results = store._process_vector_search_results(mock_results, op)

        # Should process docs starting from offset
        assert len(results) == 3  # 5 total - 2 offset = 3

    def test_extract_doc_metadata_dict_access(self, mock_valkey_client):
        """Test _extract_doc_metadata with dict-like document access."""
        store = ValkeyStore(mock_valkey_client)

        # Test with dict-like doc
        doc_dict = {"id": "langgraph:test/doc1", "score": 0.85}

        doc_id, score = store._extract_doc_metadata(doc_dict)

        assert doc_id == "langgraph:test/doc1"
        assert score == 0.85

    def test_extract_doc_metadata_attribute_access(self, mock_valkey_client):
        """Test _extract_doc_metadata with attribute access."""
        store = ValkeyStore(mock_valkey_client)

        # Test with object-like doc
        doc_obj = Mock()
        doc_obj.id = "langgraph:test/doc2"
        doc_obj.score = 0.75
        doc_obj.__dict__ = {"id": "langgraph:test/doc2", "score": 0.75}

        doc_id, score = store._extract_doc_metadata(doc_obj)

        assert doc_id == "langgraph:test/doc2"
        assert score == 0.75

    def test_extract_doc_metadata_error_handling(self, mock_valkey_client):
        """Test _extract_doc_metadata error handling."""
        store = ValkeyStore(mock_valkey_client)

        # Test with problematic doc
        doc_bad = Mock()
        doc_bad.id = None
        doc_bad.score = "invalid"

        doc_id, score = store._extract_doc_metadata(doc_bad)

        assert doc_id == ""
        assert score == 0.0

    def test_create_search_item_from_key_error_handling(self, mock_valkey_client):
        """Test _create_search_item_from_key error handling."""
        store = ValkeyStore(mock_valkey_client)

        # Mock hgetall to raise exception
        mock_valkey_client.hgetall.side_effect = Exception("Connection error")

        result = store._create_search_item_from_key(("test",), "doc1", 0.9)

        assert result is None

    def test_key_pattern_search_with_complex_data(self, mock_valkey_client):
        """Test _key_pattern_search with complex data scenarios."""
        store = ValkeyStore(mock_valkey_client)

        # Mock scan to return keys
        mock_valkey_client.scan.return_value = (
            0,
            ["langgraph:test/doc1", "langgraph:test/doc2"],
        )

        # Mock hgetall with complex data
        def mock_hgetall(key):
            if "doc1" in key:
                return {
                    b"value": orjson.dumps(
                        {"title": "Test Doc 1", "content": "Complex content"}
                    ).decode(),
                    b"created_at": b"2024-01-01T00:00:00",
                    b"updated_at": b"2024-01-01T00:00:00",
                    b"vector": orjson.dumps([0.1, 0.2, 0.3]).decode(),
                }
            elif "doc2" in key:
                return {
                    "value": orjson.dumps({"title": "Test Doc 2"}).decode(),
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "vector": "null",
                }
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        op = SearchOp(namespace_prefix=("test",), query="Test", limit=10, offset=0)

        results = store._key_pattern_search(op)

        assert isinstance(results, list)

    def test_key_pattern_search_with_malformed_data(self, mock_valkey_client):
        """Test _key_pattern_search with malformed data."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.scan.return_value = (0, ["langgraph:test/doc1"])

        # Mock hgetall with malformed JSON
        mock_valkey_client.hgetall.return_value = {
            "value": "invalid-json-data",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        op = SearchOp(namespace_prefix=("test",), query="test", limit=10, offset=0)

        results = store._key_pattern_search(op)

        # Should handle malformed data gracefully
        assert isinstance(results, list)

    def test_key_pattern_search_with_low_scores(self, mock_valkey_client):
        """Test _key_pattern_search filtering out low scores."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.scan.return_value = (0, ["langgraph:test/doc1"])

        mock_valkey_client.hgetall.return_value = {
            "value": orjson.dumps({"title": "Unrelated content"}).decode(),
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        op = SearchOp(
            namespace_prefix=("test",),
            # This should result in low score
            query="very specific query that won't match",
            limit=10,
            offset=0,
        )

        results = store._key_pattern_search(op)

        # Should filter out results with very low scores
        assert isinstance(results, list)

    def test_key_pattern_search_scan_continuation(self, mock_valkey_client):
        """Test _key_pattern_search with scan cursor continuation."""
        store = ValkeyStore(mock_valkey_client)

        # Mock scan to return cursor continuation
        scan_calls = [
            (100, ["langgraph:test/doc1"]),  # First call with cursor 100
            (0, ["langgraph:test/doc2"]),  # Second call with cursor 0 (end)
        ]
        mock_valkey_client.scan.side_effect = scan_calls

        mock_valkey_client.hgetall.return_value = {
            "value": orjson.dumps({"title": "Test"}).decode(),
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        op = SearchOp(namespace_prefix=("test",), query="test", limit=10, offset=0)

        results = store._key_pattern_search(op)

        # Should handle cursor continuation
        assert isinstance(results, list)
        assert mock_valkey_client.scan.call_count == 2

    def test_handle_search_error(self, mock_valkey_client):
        """Test search operation with error."""
        mock_valkey_client.scan.side_effect = Exception("Search error")

        store = ValkeyStore(mock_valkey_client)
        op = SearchOp(namespace_prefix=("test",), query="test")

        results = store._handle_search(op)

        # Should return empty list on error, not raise
        assert results == []

    def test_convert_to_search_items_handle_response_t_none(self, mock_valkey_client):
        """Test _convert_to_search_items when _handle_response_t returns None."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.hgetall.return_value = {"some": "data"}

        with patch.object(store, "_handle_response_t", return_value=None):
            results = [(("test",), "doc1", 0.9)]
            items = store._convert_to_search_items(results)

            assert len(items) == 0

    def test_convert_to_search_items_non_dict_response(self, mock_valkey_client):
        """Test _convert_to_search_items when _handle_response_t returns non-dict."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.hgetall.return_value = {"some": "data"}

        with patch.object(store, "_handle_response_t", return_value="not_a_dict"):
            results = [(("test",), "doc1", 0.9)]
            items = store._convert_to_search_items(results)

            assert len(items) == 0

    def test_key_pattern_search_handle_response_t_none(self, mock_valkey_client):
        """Test _key_pattern_search when scan _handle_response_t returns None."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.scan.return_value = (0, ["langgraph:test/doc1"])

        with patch.object(store, "_handle_response_t", return_value=None):
            op = SearchOp(namespace_prefix=("test",), query="test", limit=10, offset=0)

            results = store._key_pattern_search(op)

            # Should handle None response gracefully
            assert isinstance(results, list)


class TestValkeyStoreListOperations:
    """Test ValkeyStore list operations."""

    def test_handle_list_basic(self, mock_valkey_client):
        """Test basic list namespaces operation."""
        mock_valkey_client.keys.return_value = [
            "langgraph:test/key1",
            "langgraph:test/subtest/key2",
            "langgraph:other/key3",
        ]

        store = ValkeyStore(mock_valkey_client)
        op = ListNamespacesOp()

        results = store._handle_list(op)

        assert len(results) >= 2  # At least test and other namespaces
        assert ("test",) in results or ("other",) in results

    def test_handle_list_with_match_conditions(self, mock_valkey_client):
        """Test list namespaces with match conditions."""
        mock_valkey_client.keys.return_value = [
            "langgraph:prefix_test/key1",
            "langgraph:other_suffix/key2",
        ]

        store = ValkeyStore(mock_valkey_client)
        match_conditions = (MatchCondition(path=("prefix",), match_type="prefix"),)
        op = ListNamespacesOp(match_conditions=match_conditions)

        results = store._handle_list(op)

        # Should process the match conditions
        assert isinstance(results, list)

    def test_handle_list_with_suffix_match_condition(self, mock_valkey_client):
        """Test _handle_list with suffix match condition."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.keys.return_value = [
            "langgraph:test_suffix/doc1",
            "langgraph:other_suffix/doc2",
            "langgraph:no_match/doc3",
        ]

        match_conditions = (MatchCondition(path=("suffix",), match_type="suffix"),)
        op = ListNamespacesOp(match_conditions=match_conditions)

        results = store._handle_list(op)

        assert isinstance(results, list)

    def test_handle_list_keys_error_recovery(self, mock_valkey_client):
        """Test _handle_list error recovery when keys() fails."""
        store = ValkeyStore(mock_valkey_client)

        # Mock keys to raise exception
        mock_valkey_client.keys.side_effect = Exception("Keys operation failed")

        op = ListNamespacesOp()

        results = store._handle_list(op)

        # Should return empty list on error
        assert results == []

    def test_handle_list_multiple_patterns_with_errors(self, mock_valkey_client):
        """Test _handle_list with multiple patterns where some fail."""
        store = ValkeyStore(mock_valkey_client)

        # Mock keys to succeed for some patterns, fail for others
        def mock_keys(pattern):
            if "prefix" in pattern:
                return ["langgraph:prefix_test/doc1"]
            else:
                raise Exception("Pattern failed")

        mock_valkey_client.keys.side_effect = mock_keys

        match_conditions = (
            MatchCondition(path=("prefix",), match_type="prefix"),
            MatchCondition(path=("suffix",), match_type="suffix"),
        )
        op = ListNamespacesOp(match_conditions=match_conditions)

        results = store._handle_list(op)

        # Should handle partial failures gracefully
        assert isinstance(results, list)

    def test_handle_list_with_pagination(self, mock_valkey_client):
        """Test list namespaces with pagination."""
        mock_valkey_client.keys.return_value = [
            f"langgraph:ns{i}/key" for i in range(20)
        ]

        store = ValkeyStore(mock_valkey_client)
        op = ListNamespacesOp(limit=5, offset=2)

        results = store._handle_list(op)

        # Should respect limit
        assert len(results) <= 5

    def test_handle_list_error(self, mock_valkey_client):
        """Test list operation with error."""
        mock_valkey_client.keys.side_effect = Exception("Keys error")

        store = ValkeyStore(mock_valkey_client)
        op = ListNamespacesOp()

        results = store._handle_list(op)

        # Should return empty list on error, not raise
        assert results == []

    def test_handle_list_with_tuple_match_conditions(self, mock_valkey_client):
        """Test _handle_list with tuple match conditions."""
        store = ValkeyStore(mock_valkey_client)

        mock_valkey_client.keys.return_value = [
            "langgraph:test_prefix/doc1",
            "langgraph:other_suffix/doc2",
        ]

        match_conditions = (MatchCondition(path=("prefix",), match_type="prefix"),)
        op = ListNamespacesOp(match_conditions=match_conditions)

        results = store._handle_list(op)

        assert isinstance(results, list)


class TestValkeyStorePublicAPI:
    """Test ValkeyStore public API methods."""

    def test_get_method(self, mock_valkey_client):
        """Test public get method."""
        hash_data = {
            "value": '{"title": "test"}',
            "created_at": "2023-01-01T00:00:00.000000",
            "updated_at": "2023-01-01T00:00:00.000000",
        }
        mock_valkey_client.hgetall.return_value = hash_data

        store = ValkeyStore(mock_valkey_client)

        result = store.get(("test",), "key1")

        assert isinstance(result, Item)
        assert result.key == "key1"

    def test_put_method(self, mock_valkey_client):
        """Test public put method."""
        mock_valkey_client.hset.return_value = 1

        store = ValkeyStore(mock_valkey_client)

        store.put(("test",), "key1", {"title": "test"})

        mock_valkey_client.hset.assert_called_once()

    def test_delete_method(self, mock_valkey_client):
        """Test public delete method."""
        mock_valkey_client.delete.return_value = 1

        store = ValkeyStore(mock_valkey_client)

        store.delete(("test",), "key1")

        mock_valkey_client.delete.assert_called_once()

    def test_search_method(self, mock_valkey_client):
        """Test public search method."""
        with patch.object(ValkeyStore, "_handle_search", return_value=[]):
            store = ValkeyStore(mock_valkey_client)

            results = store.search(("test",), query="test")

            assert isinstance(results, list)

    def test_list_namespaces_method(self, mock_valkey_client):
        """Test public list_namespaces method."""
        with patch.object(ValkeyStore, "_handle_list", return_value=[]):
            store = ValkeyStore(mock_valkey_client)

            results = store.list_namespaces()

            assert isinstance(results, list)


class TestValkeyStoreTTLOperations:
    """Test ValkeyStore TTL operations."""

    def test_refresh_ttl_for_items_with_errors(self, mock_valkey_client):
        """Test _refresh_ttl_for_items with TTL errors."""
        ttl_config = TTLConfig(default_ttl=60)
        store = ValkeyStore(mock_valkey_client, ttl=ttl_config)

        # Mock expire to fail for some items
        def mock_expire(key, ttl):
            if "error" in key:
                raise Exception("TTL refresh failed")
            return True

        mock_valkey_client.expire.side_effect = mock_expire

        items = [
            SearchItem(
                namespace=("test",),
                key="doc1",
                value={"title": "Doc 1"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                score=0.9,
            ),
            SearchItem(
                namespace=("test",),
                key="error_doc",
                value={"title": "Error Doc"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                score=0.8,
            ),
        ]

        # Should handle TTL refresh errors gracefully
        store._refresh_ttl_for_items(items)

    def test_refresh_ttl_for_items_no_ttl_config(self, mock_valkey_client):
        """Test _refresh_ttl_for_items when no TTL config is set."""
        store = ValkeyStore(mock_valkey_client)  # No TTL config

        items = [
            SearchItem(
                namespace=("test",),
                key="doc1",
                value={"title": "Doc 1"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                score=0.9,
            ),
        ]

        # Should return early when no TTL config
        store._refresh_ttl_for_items(items)

        # Expire should not be called
        mock_valkey_client.expire.assert_not_called()

    def test_refresh_ttl_for_items_no_default_ttl(self, mock_valkey_client):
        """Test _refresh_ttl_for_items when TTL config has no default_ttl."""
        ttl_config = TTLConfig(refresh_on_read=True)  # No default_ttl
        store = ValkeyStore(mock_valkey_client, ttl=ttl_config)

        items = [
            SearchItem(
                namespace=("test",),
                key="doc1",
                value={"title": "Doc 1"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                score=0.9,
            ),
        ]

        # Should return early when no default_ttl
        store._refresh_ttl_for_items(items)

        # Expire should not be called
        mock_valkey_client.expire.assert_not_called()


class TestValkeyStoreErrorHandling:
    """Test ValkeyStore error handling."""

    def test_connection_error_handling(self, mock_valkey_client):
        """Test handling of connection errors."""
        mock_valkey_client.hgetall.side_effect = Exception("Connection failed")

        store = ValkeyStore(mock_valkey_client)
        op = GetOp(namespace=("test",), key="key1")

        # Should handle the error gracefully
        result = store._handle_get(op)
        assert result is None

    def test_search_index_error_in_setup(self, mock_valkey_client, basic_index_config):
        """Test search index error during setup."""
        mock_valkey_client.execute_command.side_effect = Exception(
            "Index creation failed"
        )

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=MagicMock(),
        ):
            store = ValkeyStore(mock_valkey_client, index=basic_index_config)

            # Should not raise, just log the error
            store.setup()

    def test_embedding_generation_error(self, mock_valkey_client):
        """Test embedding generation error."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.side_effect = Exception("Embedding failed")

        index_config = ValkeyIndexConfig(
            collection_name="test", dims=3, embed=mock_embeddings, fields=["title"]
        )

        with patch(
            "langgraph_checkpoint_aws.store.valkey.base.ensure_embeddings",
            return_value=mock_embeddings,
        ):
            store = ValkeyStore(mock_valkey_client, index=index_config)
            mock_valkey_client.hset.return_value = 1

            op = PutOp(namespace=("test",), key="key1", value={"title": "test"})

            # Should handle embedding error gracefully
            store._handle_put(op)

            mock_valkey_client.hset.assert_called_once()

    def test_document_creation_error(self, mock_valkey_client):
        """Test document creation error handling."""
        store = ValkeyStore(mock_valkey_client)

        # Test that document creation works normally
        op = PutOp(namespace=("test",), key="key1", value={"title": "test"})
        store._handle_put(op)  # Should complete without error
