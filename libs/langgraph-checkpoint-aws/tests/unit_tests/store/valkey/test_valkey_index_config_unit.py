"""Tests for ValkeyIndexConfig TypedDict and configuration handling."""

import pytest

# Skip entire module if valkey not available
pytest.importorskip("valkey")
pytest.importorskip("fakeredis")

# Now safe to import these
from unittest.mock import MagicMock, Mock, patch

import fakeredis

from langgraph_checkpoint_aws import AsyncValkeyStore, ValkeyIndexConfig, ValkeyStore


def mock_embed_fn(texts):
    """Mock embedding function that returns fixed vectors."""
    return [[0.1] * 128 for _ in texts]


@pytest.fixture
def fake_valkey_client():
    """Create a fake Valkey client using fakeredis."""
    return fakeredis.FakeStrictRedis(decode_responses=False)


class TestValkeyIndexConfig:
    """Test suite for ValkeyIndexConfig TypedDict."""

    def test_valkey_index_config_minimal(self):
        """Test ValkeyIndexConfig with minimal required fields."""
        config: ValkeyIndexConfig = {"dims": 128, "collection_name": "test_collection"}

        assert config["dims"] == 128
        assert config["collection_name"] == "test_collection"

    def test_valkey_index_config_with_all_fields(self):
        """Test ValkeyIndexConfig with all fields specified."""
        config: ValkeyIndexConfig = {
            "dims": 256,
            "collection_name": "custom_collection",
            "timezone": "America/New_York",
            "index_type": "flat",
            "hnsw_m": 32,
            "hnsw_ef_construction": 400,
            "hnsw_ef_runtime": 20,
        }

        assert config["dims"] == 256
        assert config["collection_name"] == "custom_collection"
        assert config["timezone"] == "America/New_York"
        assert config["index_type"] == "flat"
        assert config["hnsw_m"] == 32
        assert config["hnsw_ef_construction"] == 400
        assert config["hnsw_ef_runtime"] == 20

    def test_valkey_index_config_hnsw_defaults(self):
        """Test ValkeyIndexConfig with HNSW algorithm defaults."""
        config: ValkeyIndexConfig = {
            "dims": 128,
            "collection_name": "hnsw_collection",
            "index_type": "hnsw",
        }

        assert config["dims"] == 128
        assert config["collection_name"] == "hnsw_collection"
        assert config["index_type"] == "hnsw"

    def test_valkey_index_config_flat_algorithm(self):
        """Test ValkeyIndexConfig with FLAT algorithm."""
        config: ValkeyIndexConfig = {
            "dims": 512,
            "collection_name": "flat_collection",
            "index_type": "flat",
        }

        assert config["dims"] == 512
        assert config["collection_name"] == "flat_collection"
        assert config["index_type"] == "flat"


class TestCollectionNameConfiguration:
    """Test suite for collection_name configuration in ValkeyStore."""

    def test_uses_configured_collection_name_in_vector_search(self, fake_valkey_client):
        """Test that vector search uses the configured collection_name
        instead of hardcoded value.
        """
        # Create store with custom collection name
        custom_collection_name = "enterprise_memory_vectors"
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": custom_collection_name,
            },
        )

        # Verify the custom collection name is set
        assert store.collection_name == custom_collection_name

        # Perform search to trigger vector search (should not raise errors)
        results = store.search(
            namespace_prefix=("test",), query="test query", filter={"type": "document"}
        )

        # Should return empty results but not error
        assert isinstance(results, list)

    def test_uses_default_collection_name_when_not_configured(self, fake_valkey_client):
        """Test that default collection_name is used when not explicitly configured."""
        # Create store without custom collection name
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                # No collection_name specified
            },  # type: ignore
        )

        # Verify the default collection name is set
        assert store.collection_name == "langgraph_store_idx"

        # Perform search to trigger vector search (should not raise errors)
        results = store.search(namespace_prefix=("test",), query="test query")

        # Should return empty results but not error
        assert isinstance(results, list)


class TestValkeyStoreWithValkeyIndexConfig:
    """Test ValkeyStore initialization and configuration with ValkeyIndexConfig."""

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_init_with_valkey_index_config(
        self, mock_set_client_info, mock_valkey
    ):
        """Test ValkeyStore initialization with ValkeyIndexConfig."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        config: ValkeyIndexConfig = {
            "dims": 128,
            "collection_name": "test_store",
            "timezone": "UTC",
            "index_type": "hnsw",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef_runtime": 10,
        }

        store = ValkeyStore(client=mock_client, index=config)

        assert store.client == mock_client
        assert store.collection_name == "test_store"
        assert store.timezone == "UTC"
        assert store.index_type == "hnsw"
        assert store.hnsw_m == 16
        assert store.hnsw_ef_construction == 200
        assert store.hnsw_ef_runtime == 10

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_init_with_minimal_config(
        self, mock_set_client_info, mock_valkey
    ):
        """Test ValkeyStore initialization with minimal ValkeyIndexConfig."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        config: ValkeyIndexConfig = {"dims": 128, "collection_name": "minimal_store"}

        store = ValkeyStore(client=mock_client, index=config)

        assert store.client == mock_client
        assert store.collection_name == "minimal_store"
        # Test defaults are applied
        assert store.timezone == "UTC"
        assert store.index_type == "hnsw"
        assert store.hnsw_m == 16
        assert store.hnsw_ef_construction == 200
        assert store.hnsw_ef_runtime == 10

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_setup_search_index_hnsw(
        self, mock_set_client_info, mock_valkey
    ):
        """Test _setup_search_index_sync with HNSW algorithm configuration."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Mock FT._LIST to indicate search is available
        mock_client.execute_command = MagicMock()
        mock_client.ft = MagicMock()
        mock_client.ft.return_value.info.side_effect = Exception("Index doesn't exist")
        mock_client.ft.return_value.create_index = MagicMock()

        config: ValkeyIndexConfig = {
            "dims": 128,
            "collection_name": "hnsw_test",
            "index_type": "hnsw",
            "hnsw_m": 32,
            "hnsw_ef_construction": 400,
            "hnsw_ef_runtime": 20,
        }

        store = ValkeyStore(client=mock_client, index=config)

        # Mock _is_search_available to return True and set up necessary attributes
        store._search_available = True
        store.dims = 128

        # Create a proper mock embeddings object with the expected interface
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 128]
        store.embeddings = mock_embeddings

        # Mock the setup method to actually call _setup_search_index_sync
        store.setup = lambda: store._setup_search_index_sync()

        # Call the setup method
        # But first, we need to mock the _execute_command method properly
        def mock_execute_command(*args):
            if args[0] == "FT.INFO":
                # Simulate index doesn't exist
                raise Exception("Index doesn't exist")
            elif args[0] == "FT.CREATE":
                # Simulate successful index creation
                return "OK"
            return None

        store._execute_command = Mock(side_effect=mock_execute_command)

        store.setup()

        # Verify _execute_command was called with FT.CREATE
        expected_command = (
            "FT.CREATE",
            "hnsw_test",
            "ON",
            "HASH",
            "PREFIX",
            "1",
            "langgraph:",
            "SCHEMA",
            "namespace",
            "TAG",
            "key",
            "TAG",
            "value",
            "TAG",
            "vector",
            "VECTOR",
            "HNSW",
            "12",
            "TYPE",
            "FLOAT32",
            "DIM",
            "128",
            "DISTANCE_METRIC",
            "COSINE",
            "M",
            "32",
            "EF_CONSTRUCTION",
            "400",
            "EF_RUNTIME",
            "20",
        )
        store._execute_command.assert_any_call(*expected_command)

        # Verify the index name uses collection_name

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_setup_search_index_flat(
        self, mock_set_client_info, mock_valkey
    ):
        """Test _setup_search_index_sync with FLAT algorithm configuration."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Mock execute_command to simulate search availability and index creation
        def mock_execute_command(*args):
            if args[0] == "FT._LIST" or args[0] == "FT.LIST":
                return ["langgraph_store_idx"]
            elif args[0] == "FT.INFO":
                raise Exception("Index doesn't exist")
            # For FT.CREATE commands, just return success
            return "OK"

        mock_client.execute_command = MagicMock(side_effect=mock_execute_command)

        config: ValkeyIndexConfig = {
            "dims": 256,
            "collection_name": "flat_test",
            "index_type": "flat",
        }

        store = ValkeyStore(client=mock_client, index=config)
        store._setup_search_index_sync()

        # Verify FT.CREATE was called via _execute_command
        create_calls = [
            call
            for call in mock_client.execute_command.call_args_list
            if call[0] and call[0][0] == "FT.CREATE"
        ]
        assert len(create_calls) > 0, "Expected FT.CREATE command to be called"

        # Verify the index name uses collection_name
        create_call = create_calls[0]
        assert (
            "flat_test" in create_call[0]
        )  # Index name should include collection name

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_collection_name_in_index_name(
        self, mock_set_client_info, mock_valkey
    ):
        """Test that collection_name is used in index naming with langgraph prefix."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Mock execute_command to simulate search availability and index creation
        def mock_execute_command(*args):
            if args[0] == "FT._LIST" or args[0] == "FT.LIST":
                return ["langgraph_store_idx"]
            elif args[0] == "FT.INFO":
                raise Exception("Index doesn't exist")
            # For FT.CREATE commands, just return success
            return "OK"

        mock_client.execute_command = MagicMock(side_effect=mock_execute_command)

        config: ValkeyIndexConfig = {
            "dims": 128,
            "collection_name": "custom_collection_name",
        }

        store = ValkeyStore(client=mock_client, index=config)
        store._setup_search_index_sync()

        # Verify FT.CREATE was called via _execute_command
        create_calls = [
            call
            for call in mock_client.execute_command.call_args_list
            if call[0] and call[0][0] == "FT.CREATE"
        ]
        assert len(create_calls) > 0, "Expected FT.CREATE command to be called"

        # Verify the index name uses collection_name

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_init_with_embed_function(
        self, mock_set_client_info, mock_valkey
    ):
        """Test ValkeyStore initialization with ValkeyIndexConfig and embed function."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        def mock_embed_fn(texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        config: ValkeyIndexConfig = {
            "dims": 4,
            "collection_name": "embed_test",
            "embed": mock_embed_fn,
            "fields": ["title", "content"],
        }

        store = ValkeyStore(client=mock_client, index=config)

        assert store.client == mock_client
        assert store.collection_name == "embed_test"
        assert store.index is not None
        assert callable(store.index.get("embed"))
        assert store.index.get("fields") == ["title", "content"]


class TestAsyncValkeyStoreWithValkeyIndexConfig:
    """Test AsyncValkeyStore with ValkeyIndexConfig."""

    @patch("valkey.asyncio.Valkey")
    def test_async_valkey_store_init_with_valkey_index_config(self, mock_async_valkey):
        """Test AsyncValkeyStore initialization with ValkeyIndexConfig."""
        mock_client = MagicMock()
        mock_async_valkey.return_value = mock_client

        config: ValkeyIndexConfig = {
            "dims": 256,
            "collection_name": "async_test_store",
            "timezone": "Europe/London",
            "index_type": "flat",
        }

        store = AsyncValkeyStore(client=mock_client, index=config)

        assert store.client == mock_client
        assert store.collection_name == "async_test_store"
        assert store.timezone == "Europe/London"
        assert store.index_type == "flat"
        # Test defaults for HNSW parameters (even though using FLAT)
        assert store.hnsw_m == 16
        assert store.hnsw_ef_construction == 200
        assert store.hnsw_ef_runtime == 10


class TestValkeyIndexConfigValidation:
    """Test ValkeyIndexConfig validation and edge cases."""

    def test_valkey_index_config_index_type_options(self):
        """Test ValkeyIndexConfig with different index types."""
        index_types = ["hnsw", "flat"]

        for index_type in index_types:
            config: ValkeyIndexConfig = {
                "dims": 128,
                "collection_name": f"test_{index_type}",
                "index_type": index_type,
            }
            assert config["index_type"] == index_type

    def test_valkey_index_config_hnsw_parameter_ranges(self):
        """Test ValkeyIndexConfig with various HNSW parameter values."""
        # Test different M values
        for m in [8, 16, 32, 64]:
            config: ValkeyIndexConfig = {
                "dims": 128,
                "collection_name": "test_m",
                "hnsw_m": m,
            }
            assert config["hnsw_m"] == m

        # Test different EF_CONSTRUCTION values
        for ef_construction in [100, 200, 400, 800]:
            config: ValkeyIndexConfig = {
                "dims": 128,
                "collection_name": "test_ef_construction",
                "hnsw_ef_construction": ef_construction,
            }
            assert config["hnsw_ef_construction"] == ef_construction

        # Test different EF_RUNTIME values
        for ef_runtime in [5, 10, 20, 50]:
            config: ValkeyIndexConfig = {
                "dims": 128,
                "collection_name": "test_ef_runtime",
                "hnsw_ef_runtime": ef_runtime,
            }
            assert config["hnsw_ef_runtime"] == ef_runtime

    def test_valkey_index_config_timezone_options(self):
        """Test ValkeyIndexConfig with different timezone values."""
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]

        for timezone in timezones:
            config: ValkeyIndexConfig = {
                "dims": 128,
                "collection_name": "test_timezone",
                "timezone": timezone,
            }
            assert config["timezone"] == timezone

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_backward_compatibility_with_index_config(
        self, mock_set_client_info, mock_valkey
    ):
        """Test that ValkeyStore maintains backward compatibility with IndexConfig."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Legacy IndexConfig without ValkeyIndexConfig-specific fields
        legacy_config = {
            "dims": 128,
            "embed": lambda texts: [[0.1, 0.2] * 64 for _ in texts],
            "fields": ["title", "content"],
        }

        store = ValkeyStore(client=mock_client, index=legacy_config)  # pyright: ignore[reportArgumentType]

        # Should work without errors and apply defaults
        assert store.collection_name == "langgraph_store_idx"
        assert store.timezone == "UTC"
        assert store.index_type == "hnsw"
        assert store.hnsw_m == 16
        assert store.hnsw_ef_construction == 200
        assert store.hnsw_ef_runtime == 10


class TestValkeyIndexConfigIntegration:
    """Integration tests for ValkeyIndexConfig with actual store operations."""

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_with_custom_collection_name(
        self, mock_set_client_info, mock_valkey
    ):
        """Test ValkeyStore operations with custom collection name."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Mock successful index creation
        mock_client.execute_command = MagicMock()

        config: ValkeyIndexConfig = {
            "dims": 128,
            "collection_name": "custom_test_collection",
            "timezone": "America/Los_Angeles",
            "index_type": "hnsw",
            "hnsw_m": 24,
            "hnsw_ef_construction": 300,
        }

        store = ValkeyStore(client=mock_client, index=config)

        # Verify configuration is properly set
        assert store.collection_name == "custom_test_collection"
        assert store.timezone == "America/Los_Angeles"
        assert store.index_type == "hnsw"
        assert store.hnsw_m == 24
        assert store.hnsw_ef_construction == 300
        assert store.hnsw_ef_runtime == 10  # Default value

    @patch("valkey.Valkey")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_valkey_store_flat_index_configuration(
        self, mock_set_client_info, mock_valkey
    ):
        """Test ValkeyStore with FLAT index configuration."""
        mock_client = MagicMock()
        mock_valkey.return_value = mock_client

        # Mock execute_command to simulate search availability and index creation
        def mock_execute_command(*args):
            if args[0] == "FT._LIST" or args[0] == "FT.LIST":
                return ["langgraph_store_idx"]
            elif args[0] == "FT.INFO":
                raise Exception("Index doesn't exist")
            # For FT.CREATE commands, just return success
            return "OK"

        mock_client.execute_command = MagicMock(side_effect=mock_execute_command)

        config: ValkeyIndexConfig = {
            "dims": 512,
            "collection_name": "flat_index_test",
            "index_type": "flat",
        }

        store = ValkeyStore(client=mock_client, index=config)

        store._setup_search_index_sync()

        # Verify FT.CREATE was called via _execute_command
        create_calls = [
            call
            for call in mock_client.execute_command.call_args_list
            if call[0] and call[0][0] == "FT.CREATE"
        ]
        assert len(create_calls) > 0, "Expected FT.CREATE command to be called"

    @patch("valkey.asyncio.Valkey")
    def test_async_valkey_store_configuration(self, mock_async_valkey):
        """Test AsyncValkeyStore configuration with ValkeyIndexConfig."""
        mock_client = MagicMock()
        mock_async_valkey.return_value = mock_client

        config: ValkeyIndexConfig = {
            "dims": 384,
            "collection_name": "async_config_test",
            "timezone": "Europe/Berlin",
            "index_type": "hnsw",
            "hnsw_m": 48,
            "hnsw_ef_construction": 500,
            "hnsw_ef_runtime": 25,
        }

        store = AsyncValkeyStore(client=mock_client, index=config)

        # Verify all configuration values are properly set
        assert store.collection_name == "async_config_test"
        assert store.timezone == "Europe/Berlin"
        assert store.index_type == "hnsw"
        assert store.hnsw_m == 48
        assert store.hnsw_ef_construction == 500
        assert store.hnsw_ef_runtime == 25


class TestCollectionNameConfigurationExtended:
    """Extended tests for collection_name configuration in ValkeyStore."""

    def test_index_creation_uses_configured_collection_name(self, fake_valkey_client):
        """Test that index creation uses the configured collection_name."""
        custom_collection_name = "enterprise_memory_vectors"

        # Create store with custom collection name
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": ["user_id", "memory_type", "content"],
                "embed": mock_embed_fn,
                "collection_name": custom_collection_name,
            },
        )

        # Verify the custom collection name is set
        assert store.collection_name == custom_collection_name

        # Setup the store (this should create the index)
        store.setup()

        # Verify no errors occurred during setup
        assert store.collection_name == custom_collection_name

    def test_index_creation_includes_configured_fields(self, fake_valkey_client):
        """Test that index creation includes all configured searchable fields."""
        custom_fields = ["user_id", "memory_type", "importance", "content", "tags"]

        # Create store with custom fields
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": custom_fields,
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Verify the fields are configured
        assert store.index_fields == custom_fields

        # Setup the store (this should create the index with all fields)
        store.setup()

        # Verify no errors occurred during setup
        assert store.index_fields == custom_fields

    def test_document_creation_includes_searchable_fields(self, fake_valkey_client):
        """Test that document creation includes searchable fields as hash fields."""
        # Create store with searchable fields
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": ["user_id", "memory_type", "content", "tags"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Test document with searchable fields
        test_value = {
            "user_id": "user123",
            "memory_type": "fact",
            "content": "Test memory content",
            "tags": ["important", "work"],
            "other_field": "not indexed",
        }

        # Put the document
        store.put(("test",), "doc1", test_value)

        # Retrieve the document to verify it was stored correctly
        result = store.get(("test",), "doc1")
        assert result is not None
        assert result.value == test_value

        # Check the raw hash fields to verify searchable fields are stored
        key = store._build_key(("test",), "doc1")
        hash_fields = fake_valkey_client.hgetall(key)

        # Convert bytes keys/values to strings for easier checking
        hash_fields_str = {}
        for k, v in hash_fields.items():
            key_str = k.decode("utf-8") if isinstance(k, bytes) else k
            val_str = v.decode("utf-8") if isinstance(v, bytes) else v
            hash_fields_str[key_str] = val_str

        # Verify that searchable fields are included in hash fields
        # (without value_ prefix)
        assert "user_id" in hash_fields_str
        assert "memory_type" in hash_fields_str
        assert "content" in hash_fields_str
        assert "tags" in hash_fields_str

        # Verify the values are correct
        assert hash_fields_str["user_id"] == "user123"
        assert hash_fields_str["memory_type"] == "fact"
        assert hash_fields_str["content"] == "Test memory content"
        # List should be comma-separated
        assert hash_fields_str["tags"] == "important,work"

        # Verify non-indexed fields are not included as separate hash fields
        assert "other_field" not in hash_fields_str

    def test_list_fields_handled_correctly(self, fake_valkey_client):
        """Test that list fields are properly converted to comma-separated
        strings for TAG indexing.
        """
        # Create store with fields that might contain lists
        store = ValkeyStore(
            fake_valkey_client,
            index={
                "dims": 128,
                "fields": ["tags", "categories"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Test document with list fields
        test_value = {
            "tags": ["machine-learning", "ai", "python"],
            "categories": ["tech", "tutorial"],
            "title": "ML Guide",
        }

        # Put the document
        store.put(("test",), "doc1", test_value)

        # Retrieve the document to verify it was stored correctly
        result = store.get(("test",), "doc1")
        assert result is not None
        assert result.value == test_value

        # Check the raw hash fields to verify list conversion
        key = store._build_key(("test",), "doc1")
        hash_fields = fake_valkey_client.hgetall(key)

        # Convert bytes keys/values to strings for easier checking
        hash_fields_str = {}
        for k, v in hash_fields.items():
            key_str = k.decode("utf-8") if isinstance(k, bytes) else k
            val_str = v.decode("utf-8") if isinstance(v, bytes) else v
            hash_fields_str[key_str] = val_str

        # Verify that list fields are converted to comma-separated strings
        assert hash_fields_str["tags"] == "machine-learning,ai,python"
        assert hash_fields_str["categories"] == "tech,tutorial"
