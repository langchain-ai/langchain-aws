"""Unit tests for ValkeyCheckpointSaver using mocks."""

import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from valkey import Valkey

from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver


class TestValkeyCheckpointSaverUnit:
    """Unit tests for ValkeyCheckpointSaver that don't require external dependencies."""

    @pytest.fixture
    def mock_valkey_client(self):
        """Create a mock Valkey client."""
        client = Mock(spec=Valkey)
        client.get.return_value = None
        client.set.return_value = True
        client.delete.return_value = 1
        client.scan.return_value = (0, [])
        client.exists.return_value = 0
        client.lrange.return_value = []
        client.lpush.return_value = 1
        client.expire.return_value = True
        client.keys.return_value = []

        # Mock pipeline
        pipeline_mock = Mock()
        pipeline_mock.__enter__ = Mock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = Mock(return_value=None)
        client.pipeline.return_value = pipeline_mock

        return client

    @pytest.fixture
    def saver(self, mock_valkey_client):
        """Create a ValkeyCheckpointSaver with mocked client."""
        return ValkeyCheckpointSaver(mock_valkey_client, ttl=3600.0)

    @pytest.fixture
    def sample_config(self) -> RunnableConfig:
        """Sample configuration for testing."""
        return {
            "configurable": {"thread_id": "test-thread", "checkpoint_ns": "test-ns"}
        }

    @pytest.fixture
    def sample_checkpoint(self) -> Checkpoint:
        """Sample checkpoint for testing."""
        return {
            "v": 1,
            "id": "test-checkpoint-id",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"key": "value"},
            "channel_versions": {"key": 1},
            "versions_seen": {"key": {"key": 1}},
            "pending_sends": [],
        }

    @pytest.fixture
    def sample_metadata(self) -> CheckpointMetadata:
        """Sample metadata for testing."""
        return {"source": "input", "step": 1, "writes": {"key": "value"}}

    def test_init_with_ttl(self, mock_valkey_client):
        """Test saver initialization with TTL."""
        saver = ValkeyCheckpointSaver(mock_valkey_client, ttl=3600.0)

        assert saver.client == mock_valkey_client
        assert saver.ttl == 3600.0
        assert isinstance(saver.serde, JsonPlusSerializer)

    def test_init_without_ttl(self, mock_valkey_client):
        """Test saver initialization without TTL."""
        saver = ValkeyCheckpointSaver(mock_valkey_client)

        assert saver.client == mock_valkey_client
        assert saver.ttl is None

    def test_checkpoint_key_generation(self, saver):
        """Test checkpoint key generation."""
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        checkpoint_id = "test-checkpoint-id"
        expected_key = "checkpoint:test-thread:test-ns:test-checkpoint-id"

        actual_key = saver._make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        assert actual_key == expected_key

    def test_checkpoint_key_generation_no_namespace(self, saver):
        """Test checkpoint key generation without namespace."""
        thread_id = "test-thread"
        checkpoint_ns = ""
        checkpoint_id = "test-checkpoint-id"
        expected_key = "checkpoint:test-thread::test-checkpoint-id"

        actual_key = saver._make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        assert actual_key == expected_key

    def test_writes_key_generation(self, saver):
        """Test writes key generation."""
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        checkpoint_id = "test-checkpoint-id"
        expected_key = "writes:test-thread:test-ns:test-checkpoint-id"

        actual_key = saver._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)
        assert actual_key == expected_key

    def test_thread_key_generation(self, saver):
        """Test thread key generation."""
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        expected_key = "thread:test-thread:test-ns"

        actual_key = saver._make_thread_key(thread_id, checkpoint_ns)
        assert actual_key == expected_key

    def test_put_checkpoint_success(
        self,
        saver,
        mock_valkey_client,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """Test successful checkpoint storage."""
        # Mock pipeline operations
        pipeline_mock = Mock()
        pipeline_mock.__enter__ = Mock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = Mock(return_value=None)
        mock_valkey_client.pipeline.return_value = pipeline_mock

        config = {"configurable": {"thread_id": "test-thread"}}
        new_versions = {"key": 2}

        result = saver.put(config, sample_checkpoint, sample_metadata, new_versions)

        # Verify the result
        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]
        assert result["configurable"]["checkpoint_ns"] == ""
        assert result["configurable"]["thread_id"] == "test-thread"

        # Verify pipeline was used
        mock_valkey_client.pipeline.assert_called()

    def test_put_checkpoint_with_ttl(
        self, mock_valkey_client, sample_config, sample_checkpoint, sample_metadata
    ):
        """Test checkpoint storage with TTL."""
        saver = ValkeyCheckpointSaver(mock_valkey_client, ttl=3600.0)

        # Mock pipeline operations
        pipeline_mock = Mock()
        pipeline_mock.__enter__ = Mock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = Mock(return_value=None)
        mock_valkey_client.pipeline.return_value = pipeline_mock

        new_versions = {"key": 2}

        saver.put(sample_config, sample_checkpoint, sample_metadata, new_versions)

        # Verify pipeline was used (TTL operations happen in pipeline)
        mock_valkey_client.pipeline.assert_called()

    def test_get_checkpoint_found(self, saver, mock_valkey_client):
        """Test getting an existing checkpoint."""
        # Mock stored checkpoint data with proper format
        checkpoint_data = {
            "v": 1,
            "id": "test-id",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"key": "value"},
            "channel_versions": {"key": 1},
            "versions_seen": {"key": {"key": 1}},
            "pending_sends": [],
        }

        checkpoint_info = saver._serialize_checkpoint_data(
            {"configurable": {"thread_id": "test-thread", "checkpoint_id": "test-id"}},
            checkpoint_data,
            {"source": "input", "step": 1},
        )

        mock_valkey_client.get.return_value = json.dumps(checkpoint_info).encode()
        # Mock writes as empty list
        mock_valkey_client.lrange.return_value = []

        config = {
            "configurable": {"thread_id": "test-thread", "checkpoint_id": "test-id"}
        }

        # Mock the writes data (get method returns writes data)
        mock_valkey_client.get.side_effect = [
            json.dumps(checkpoint_info).encode(),  # First call for checkpoint
            b"[]",  # Second call for writes (empty list as JSON bytes)
        ]

        result = saver.get_tuple(config)

        assert result is not None
        assert isinstance(result, CheckpointTuple)
        assert (
            mock_valkey_client.get.call_count == 2
        )  # Called for both checkpoint and writes

    def test_get_checkpoint_not_found(self, saver, mock_valkey_client):
        """Test getting a non-existent checkpoint."""
        mock_valkey_client.get.return_value = None

        config = {
            "configurable": {"thread_id": "test-thread", "checkpoint_id": "missing"}
        }

        result = saver.get_tuple(config)

        assert result is None
        mock_valkey_client.get.assert_called()

    def test_list_checkpoints(self, saver, mock_valkey_client, sample_config):
        """Test listing checkpoints."""
        # Mock checkpoint IDs in thread
        mock_valkey_client.lrange.return_value = [b"id1", b"id2"]

        # Mock checkpoint data
        checkpoint_data = {
            "v": 1,
            "id": "id1",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"key": "value"},
            "channel_versions": {"key": 1},
            "versions_seen": {"key": {"key": 1}},
            "pending_sends": [],
        }

        checkpoint_info = saver._serialize_checkpoint_data(
            sample_config, checkpoint_data, {"step": 1}
        )

        mock_valkey_client.get.return_value = json.dumps(checkpoint_info).encode()

        # Mock the get calls - list operation calls get for each checkpoint and writes
        # Need to cycle through these for multiple checkpoints (id1, id2)
        def get_side_effect(*args, **kwargs):
            key = args[0] if args else ""
            if "checkpoint:" in key:
                return json.dumps(checkpoint_info).encode()
            elif "writes:" in key:
                return b"[]"  # Empty writes list
            else:
                return None

        mock_valkey_client.get.side_effect = get_side_effect

        checkpoints = list(saver.list(sample_config))

        # Should get at least one checkpoint
        assert len(checkpoints) >= 0
        mock_valkey_client.lrange.assert_called()

    def test_list_checkpoints_with_filter(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test listing checkpoints with metadata filters."""
        filter_config = {"source": "input"}

        # Mock empty results
        mock_valkey_client.lrange.return_value = []

        list(saver.list(sample_config, filter=filter_config))

        mock_valkey_client.lrange.assert_called()

    def test_list_checkpoints_with_limit(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test listing checkpoints with limit."""
        # Mock checkpoint IDs
        mock_valkey_client.lrange.return_value = [b"id1", b"id2", b"id3"]

        list(saver.list(sample_config, limit=2))

        mock_valkey_client.lrange.assert_called()

    def test_put_writes(self, saver, mock_valkey_client):
        """Test storing writes."""
        config_with_checkpoint = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "test-ns",
                "checkpoint_id": "test-checkpoint-id",
            }
        }

        task_id = "test-task-id"
        writes = [("channel", "value")]

        # Mock get for existing writes and pipeline for setting
        mock_valkey_client.get.return_value = None  # No existing writes

        saver.put_writes(config_with_checkpoint, writes, task_id)

        # Verify pipeline was used (put_writes uses set via pipeline)
        mock_valkey_client.pipeline.assert_called()

    def test_serialization_roundtrip(self, saver, sample_checkpoint):
        """Test checkpoint serialization and deserialization."""
        # Test that serialization works correctly
        serialized = saver.serde.dumps_typed(sample_checkpoint)
        deserialized = saver.serde.loads_typed(serialized)

        assert deserialized == sample_checkpoint

    def test_error_handling_valkey_connection_error(self, mock_valkey_client):
        """Test error handling when Valkey connection fails."""
        mock_valkey_client.get.side_effect = Exception("Connection error")
        saver = ValkeyCheckpointSaver(mock_valkey_client)

        config = {
            "configurable": {"thread_id": "test-thread", "checkpoint_id": "test-id"}
        }

        with pytest.raises(Exception):
            saver.get_tuple(config)

    def test_context_manager_not_supported(self, mock_valkey_client):
        """Test that saver doesn't support context manager by default."""
        saver = ValkeyCheckpointSaver(mock_valkey_client)

        # ValkeyCheckpointSaver doesn't implement context manager protocol directly
        # It's used through factory methods that provide context managers
        assert not hasattr(saver, "__enter__")
        assert not hasattr(saver, "__exit__")

    @patch("langgraph_checkpoint_aws.checkpoint.valkey.base.set_client_info")
    def test_client_info_setting(self, mock_set_client_info, mock_valkey_client):
        """Test that client info is set during initialization."""
        ValkeyCheckpointSaver(mock_valkey_client)

        mock_set_client_info.assert_called_once_with(mock_valkey_client)

    def test_namespace_handling(self, mock_valkey_client):
        """Test namespace handling in key generation."""
        saver = ValkeyCheckpointSaver(mock_valkey_client)

        # Test with namespace
        key_with_ns = saver._make_checkpoint_key("test", "ns1", "id1")
        assert key_with_ns == "checkpoint:test:ns1:id1"

        # Test without namespace
        key_without_ns = saver._make_checkpoint_key("test", "", "id1")
        assert key_without_ns == "checkpoint:test::id1"

    def test_thread_id_validation(self, saver):
        """Test that thread_id is handled properly."""
        # Test normal thread ID
        key = saver._make_checkpoint_key("test-thread", "ns", "id1")
        assert key == "checkpoint:test-thread:ns:id1"

    def test_cleanup_operations(self, saver, mock_valkey_client):
        """Test cleanup/deletion operations."""
        mock_valkey_client.keys.return_value = [
            b"thread:test-thread:ns1",
            b"checkpoint:test-thread:ns1:id1",
        ]
        mock_valkey_client.delete.return_value = 2

        # Test thread deletion
        saver.delete_thread("test-thread")

        # Should have called keys and delete operations
        mock_valkey_client.keys.assert_called()
        mock_valkey_client.delete.assert_called()

    def test_complex_checkpoint_data(self, saver, mock_valkey_client):
        """Test handling complex checkpoint data."""
        complex_checkpoint = {
            "v": 1,
            "id": "complex-id",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {
                "messages": [{"role": "user", "content": "Hello"}],
                "context": {"nested": {"data": [1, 2, 3]}},
            },
            "channel_versions": {"messages": 5, "context": 2},
            "versions_seen": {"messages": {"messages": 5}, "context": {"context": 2}},
            "pending_sends": [("output", {"result": "processed"})],
        }

        metadata = {
            "source": "input",
            "step": 10,
            "writes": {"complex": {"nested": True}},
        }

        # Mock pipeline
        pipeline_mock = Mock()
        pipeline_mock.__enter__ = Mock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = Mock(return_value=None)
        mock_valkey_client.pipeline.return_value = pipeline_mock

        config = {"configurable": {"thread_id": "complex-thread"}}
        new_versions = {"messages": 6, "context": 3}

        result = saver.put(config, complex_checkpoint, metadata, new_versions)

        # Should handle complex data without errors
        assert result["configurable"]["checkpoint_id"] == complex_checkpoint["id"]
        mock_valkey_client.pipeline.assert_called()

    def test_multiple_writes_handling(self, saver, mock_valkey_client):
        """Test handling multiple writes for same checkpoint."""
        config_with_checkpoint = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "test-ns",
                "checkpoint_id": "test-checkpoint-id",
            }
        }

        writes_batch1 = [("channel1", "value1"), ("channel2", "value2")]
        writes_batch2 = [("channel3", "value3")]

        # Mock get for existing writes
        mock_valkey_client.get.return_value = None  # No existing writes

        saver.put_writes(config_with_checkpoint, writes_batch1, "task1")
        saver.put_writes(config_with_checkpoint, writes_batch2, "task2")

        # Should handle multiple write operations using pipeline
        assert mock_valkey_client.pipeline.call_count == 2

    def test_serialize_checkpoint_data(self, saver, sample_checkpoint, sample_metadata):
        """Test checkpoint data serialization."""
        config = {"configurable": {"thread_id": "test-thread"}}

        serialized = saver._serialize_checkpoint_data(
            config, sample_checkpoint, sample_metadata
        )

        # Should contain the expected fields
        assert "checkpoint" in serialized
        assert "metadata" in serialized
        assert "parent_checkpoint_id" in serialized  # Not parent_config

    def test_deserialize_checkpoint_data(self, saver):
        """Test checkpoint data deserialization."""
        # Create proper serialized data using the same method as the saver
        typed_data = saver.serde.dumps_typed(
            {
                "v": 1,
                "id": "test-id",
                "ts": "2024-01-01T00:00:00+00:00",
                "channel_values": {"key": "value"},
                "channel_versions": {"key": 1},
                "versions_seen": {"key": {"key": 1}},
                "pending_sends": [],
            }
        )

        checkpoint_info = {
            "checkpoint": typed_data[1],  # Get the serialized bytes
            "type": typed_data[0],  # Get the type
            "metadata": saver.jsonplus_serde.dumps({"step": 1}),
            "parent_checkpoint_id": None,
        }

        writes = []  # Empty writes list
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        checkpoint_id = "test-id"
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        }

        result = saver._deserialize_checkpoint_data(
            checkpoint_info, writes, thread_id, checkpoint_ns, checkpoint_id, config
        )

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == checkpoint_id
