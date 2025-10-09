"""Tests for AgentCore Valkey checkpoint saver."""

import base64
import json
import time
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver
from langgraph_checkpoint_aws.agentcore.valkey.models import (
    StoredCheckpoint,
    ValkeyCheckpointerConfig,
)


class TestAgentCoreValkeySaver:
    """Test AgentCoreValkeySaver class."""

    @pytest.fixture
    def mock_valkey_client(self):
        """Create a mock Valkey client."""
        client = MagicMock()
        client.get.return_value = None
        client.set.return_value = True
        client.setex.return_value = True
        client.lpush.return_value = 1
        client.lrange.return_value = []
        client.lindex.return_value = None
        client.rpush.return_value = 1
        client.delete.return_value = 1
        client.keys.return_value = []
        client.expire.return_value = True
        # Mock for async detection
        client.aclose = None
        client.__aenter__ = None
        return client

    @pytest.fixture
    def saver(self, mock_valkey_client):
        """Create a AgentCoreValkeySaver instance with mocked client."""
        return AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

    @pytest.fixture
    def sample_config(self):
        """Sample runnable config."""
        return {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }

    @pytest.fixture
    def sample_checkpoint(self):
        """Sample checkpoint data."""
        return {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
            "channel_versions": {"messages": "1.0"},
        }

    def test_init(self, mock_valkey_client):
        """Test saver initialization."""
        saver = AgentCoreValkeySaver(mock_valkey_client)

        assert saver.client == mock_valkey_client
        assert saver.ttl is None
        assert isinstance(saver.jsonplus_serde, JsonPlusSerializer)
        assert saver.event_serializer is not None

    def test_init_with_ttl(self, mock_valkey_client):
        """Test saver initialization with TTL."""
        saver = AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

        assert saver.ttl == 3600

    def test_make_checkpoint_key(self, saver):
        """Test checkpoint key generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns="test-ns"
        )

        key = saver._make_checkpoint_key(config, "checkpoint-1")
        expected = "agentcore:checkpoint:session-1_test-ns:agent-1:test-ns:checkpoint-1"
        assert key == expected

    def test_make_writes_key(self, saver):
        """Test writes key generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns=""
        )

        key = saver._make_writes_key(config, "checkpoint-1")
        expected = "agentcore:writes:session-1:agent-1::checkpoint-1"
        assert key == expected

    def test_make_channel_key(self, saver):
        """Test channel key generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns=""
        )

        key = saver._make_channel_key(config, "messages", "checkpoint-1")
        expected = "agentcore:channel:session-1:agent-1::messages:checkpoint-1"
        assert key == expected

    def test_serialize_checkpoint(self, saver, sample_config, sample_checkpoint):
        """Test checkpoint serialization."""
        metadata = {"user": "test"}

        stored_checkpoint = saver._serialize_checkpoint(
            sample_config, sample_checkpoint, metadata
        )

        assert isinstance(stored_checkpoint, StoredCheckpoint)
        assert stored_checkpoint.checkpoint_id == "checkpoint-1"
        assert stored_checkpoint.session_id == "session-1"
        assert stored_checkpoint.actor_id == "agent-1"
        assert stored_checkpoint.checkpoint_ns == ""
        assert stored_checkpoint.parent_checkpoint_id == "checkpoint-1"
        assert stored_checkpoint.checkpoint_data is not None
        assert stored_checkpoint.metadata is not None
        assert isinstance(stored_checkpoint.created_at, float)

    def test_get_tuple_specific_checkpoint(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test getting a specific checkpoint."""
        # Create proper serialized data
        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        # Mock stored checkpoint data
        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": empty_dict_b64},
            created_at=time.time(),
        )

        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()
        mock_valkey_client.lrange.return_value = []

        result = saver.get_tuple(sample_config)

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-1"

    def test_get_tuple_latest_checkpoint(self, saver, mock_valkey_client):
        """Test getting the latest checkpoint."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        # Mock latest checkpoint ID
        mock_valkey_client.lindex.return_value = b"checkpoint-2"

        # Create proper serialized data
        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        # Mock stored checkpoint data
        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-2",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": empty_dict_b64},
            created_at=time.time(),
        )

        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()
        mock_valkey_client.lrange.return_value = []

        result = saver.get_tuple(config)

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-2"

    def test_get_tuple_not_found(self, saver, mock_valkey_client, sample_config):
        """Test getting a checkpoint that doesn't exist."""
        mock_valkey_client.get.return_value = None

        result = saver.get_tuple(sample_config)

        assert result is None

    def test_put_checkpoint(
        self, saver, mock_valkey_client, sample_config, sample_checkpoint
    ):
        """Test storing a checkpoint."""
        metadata = {"user": "test"}
        new_versions = {"messages": "1.0"}

        result_config = saver.put(
            sample_config, sample_checkpoint, metadata, new_versions
        )

        # Verify checkpoint was stored
        mock_valkey_client.setex.assert_called()  # Due to TTL
        mock_valkey_client.lpush.assert_called()  # Session list update

        # Verify returned config
        assert result_config["configurable"]["thread_id"] == "session-1"
        assert result_config["configurable"]["actor_id"] == "agent-1"
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint-1"

    def test_put_checkpoint_no_ttl(
        self, mock_valkey_client, sample_config, sample_checkpoint
    ):
        """Test storing a checkpoint without TTL."""
        saver = AgentCoreValkeySaver(mock_valkey_client)  # No TTL
        metadata = {"user": "test"}
        new_versions = {"messages": "1.0"}

        saver.put(sample_config, sample_checkpoint, metadata, new_versions)

        # Verify set was called instead of setex
        mock_valkey_client.set.assert_called()

    def test_put_writes(self, saver, mock_valkey_client, sample_config):
        """Test storing writes."""
        writes = [("messages", {"role": "assistant", "content": "Hello!"})]
        task_id = "task-1"

        saver.put_writes(sample_config, writes, task_id)

        # Verify writes were stored
        mock_valkey_client.rpush.assert_called()
        mock_valkey_client.expire.assert_called()  # Due to TTL

    def test_put_writes_no_checkpoint_id(self, saver):
        """Test put_writes without checkpoint_id raises error."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }
        writes = [("messages", {"content": "test"})]

        with pytest.raises(Exception):  # Should raise InvalidConfigError
            saver.put_writes(config, writes, "task-1")

    def test_delete_thread(self, saver, mock_valkey_client):
        """Test deleting a thread."""
        thread_id = "session-1"
        actor_id = "agent-1"

        # Mock checkpoint IDs
        mock_valkey_client.lrange.return_value = [b"checkpoint-1", b"checkpoint-2"]
        mock_valkey_client.keys.return_value = [b"channel-key-1", b"channel-key-2"]

        saver.delete_thread(thread_id, actor_id)

        # Verify delete was called
        mock_valkey_client.delete.assert_called()

    def test_get_next_version(self, saver):
        """Test version generation."""
        # Test with None (initial version)
        version = saver.get_next_version(None, None)
        assert version.startswith("00000000000000000000000000000001.")

        # Test with integer
        version = saver.get_next_version(5, None)
        assert version.startswith("00000000000000000000000000000006.")

        # Test with string version
        version = saver.get_next_version("10.123456", None)
        assert version.startswith("00000000000000000000000000000011.")

    def test_async_methods(self, saver, sample_config):
        """Test async methods delegate to sync versions."""
        # These should not raise exceptions and delegate to sync methods
        import asyncio

        async def test_async():
            result = await saver.aget_tuple(sample_config)
            # Should return same as sync version (None in this case)
            assert result is None

            # Test async list
            checkpoints = []
            async for checkpoint in saver.alist(sample_config):
                checkpoints.append(checkpoint)
            assert len(checkpoints) == 0

        asyncio.run(test_async())


class TestAgentCoreValkeySaverIntegration:
    """Integration-style tests for AgentCoreValkeySaver."""

    @pytest.fixture
    def integration_saver(self):
        """Create a saver for integration testing with a real-ish mock."""
        client = MagicMock()

        # Simulate storage
        storage = {}
        lists = {}

        def mock_get(key):
            return storage.get(key)

        def mock_set(key, value):
            storage[key] = value
            return True

        def mock_setex(key, ttl, value):
            storage[key] = value
            return True

        def mock_lpush(key, *values):
            if key not in lists:
                lists[key] = []
            for value in reversed(values):
                lists[key].insert(0, value)
            return len(lists[key])

        def mock_lrange(key, start, end):
            if key not in lists:
                return []
            lst = lists[key]
            if end == -1:
                return [
                    item.encode() if isinstance(item, str) else item
                    for item in lst[start:]
                ]
            return [
                item.encode() if isinstance(item, str) else item
                for item in lst[start : end + 1]
            ]

        def mock_lindex(key, index):
            if key not in lists or not lists[key]:
                return None
            try:
                item = lists[key][index]
                return item.encode() if isinstance(item, str) else item
            except IndexError:
                return None

        client.get.side_effect = mock_get
        client.set.side_effect = mock_set
        client.setex.side_effect = mock_setex
        client.lpush.side_effect = mock_lpush
        client.lrange.side_effect = mock_lrange
        client.lindex.side_effect = mock_lindex
        client.rpush = MagicMock(return_value=1)
        client.expire = MagicMock(return_value=True)
        client.delete = MagicMock(return_value=1)
        client.keys = MagicMock(return_value=[])

        return AgentCoreValkeySaver(client, ttl=3600)

    def test_full_checkpoint_lifecycle(self, integration_saver):
        """Test complete checkpoint lifecycle: put, get, list."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        checkpoint = {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
        }

        metadata = {"user": "test_user"}
        new_versions = {"messages": "1.0"}

        # Put checkpoint
        result_config = integration_saver.put(
            config, checkpoint, metadata, new_versions
        )

        # Verify returned config
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint-1"

        # Get checkpoint
        retrieved = integration_saver.get_tuple(result_config)

        assert retrieved is not None
        assert retrieved.config["configurable"]["checkpoint_id"] == "checkpoint-1"

        # List checkpoints
        checkpoints = list(integration_saver.list(config))
        assert len(checkpoints) == 1
        assert checkpoints[0].config["configurable"]["checkpoint_id"] == "checkpoint-1"
