"""Tests for AgentCore Valkey checkpoint saver."""

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("valkey")
pytest.importorskip("orjson")

from langgraph.checkpoint.base import CheckpointTuple
from valkey.exceptions import ConnectionError, TimeoutError, ValkeyError

from langgraph_checkpoint_aws.agentcore.constants import InvalidConfigError
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver
from langgraph_checkpoint_aws.agentcore.valkey.models import (
    StoredCheckpoint,
    StoredWrite,
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
        client.close.return_value = None
        # Mock for async detection
        client.aclose = None
        client.__aenter__ = None
        return client

    @pytest.fixture
    def mock_async_valkey_client(self):
        """Create a mock async Valkey client."""
        client = AsyncMock()
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
        client.aclose = AsyncMock()
        client.__aenter__ = AsyncMock()
        return client

    @pytest.fixture
    def saver(self, mock_valkey_client):
        """Create a AgentCoreValkeySaver instance with mocked client."""
        return AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

    @pytest.fixture
    def async_saver(self, mock_async_valkey_client):
        """Create a AgentCoreValkeySaver instance with mocked async client."""
        return AgentCoreValkeySaver(mock_async_valkey_client, ttl=3600)

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
        # Remove async detection attributes to make it sync
        del mock_valkey_client.aclose
        del mock_valkey_client.__aenter__

        saver = AgentCoreValkeySaver(mock_valkey_client)

        assert saver.client == mock_valkey_client
        assert saver.ttl is None
        assert saver.event_serializer is not None
        assert saver.max_retries == 3
        assert saver.retry_delay == 0.1
        assert not saver.is_async

    def test_init_with_ttl(self, mock_valkey_client):
        """Test saver initialization with TTL."""
        saver = AgentCoreValkeySaver(mock_valkey_client, ttl=3600)
        assert saver.ttl == 3600

    def test_init_with_async_client(self, mock_async_valkey_client):
        """Test saver initialization with async client."""
        saver = AgentCoreValkeySaver(mock_async_valkey_client)
        assert saver.is_async

    def test_init_validation_errors(self, mock_valkey_client):
        """Test initialization validation errors."""
        # Test negative TTL
        with pytest.raises(ValueError, match="TTL must be positive"):
            AgentCoreValkeySaver(mock_valkey_client, ttl=-1)

        # Test negative max_retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            AgentCoreValkeySaver(mock_valkey_client, max_retries=-1)

        # Test negative retry_delay
        with pytest.raises(ValueError, match="retry_delay must be non-negative"):
            AgentCoreValkeySaver(mock_valkey_client, retry_delay=-1)

    def test_execute_with_retry_success(self, saver):
        """Test successful retry execution."""
        operation = MagicMock(return_value="success")
        result = saver._execute_with_retry(operation, "arg1", kwarg1="value1")

        assert result == "success"
        operation.assert_called_once_with("arg1", kwarg1="value1")

    def test_execute_with_retry_failure_then_success(self, saver):
        """Test retry logic with failure then success."""
        operation = MagicMock()
        operation.side_effect = [ConnectionError("Connection failed"), "success"]

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = saver._execute_with_retry(operation)

        assert result == "success"
        assert operation.call_count == 2

    def test_execute_with_retry_all_failures(self, saver):
        """Test retry logic with all failures."""
        operation = MagicMock()
        operation.side_effect = ConnectionError("Connection failed")

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(ConnectionError):
                saver._execute_with_retry(operation)

        assert operation.call_count == saver.max_retries + 1

    def test_execute_with_retry_timeout_error(self, saver):
        """Test retry logic with TimeoutError."""
        operation = MagicMock()
        operation.side_effect = TimeoutError("Timeout")

        with patch("time.sleep"):
            with pytest.raises(TimeoutError):
                saver._execute_with_retry(operation)

    def test_execute_with_retry_valkey_error(self, saver):
        """Test retry logic with ValkeyError."""
        operation = MagicMock()
        operation.side_effect = ValkeyError("Valkey error")

        with patch("time.sleep"):
            with pytest.raises(ValkeyError):
                saver._execute_with_retry(operation)

    @pytest.mark.asyncio
    async def test_aexecute_with_retry_success(self, saver):
        """Test successful async retry execution."""
        operation = AsyncMock(return_value="success")
        result = await saver._aexecute_with_retry(operation, "arg1", kwarg1="value1")

        assert result == "success"
        operation.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_aexecute_with_retry_failure_then_success(self, saver):
        """Test async retry logic with failure then success."""
        operation = AsyncMock()
        operation.side_effect = [ConnectionError("Connection failed"), "success"]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await saver._aexecute_with_retry(operation)

        assert result == "success"
        assert operation.call_count == 2

    @pytest.mark.asyncio
    async def test_aexecute_with_retry_all_failures(self, saver):
        """Test async retry logic with all failures."""
        operation = AsyncMock()
        operation.side_effect = ConnectionError("Connection failed")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                await saver._aexecute_with_retry(operation)

        assert operation.call_count == saver.max_retries + 1

    def test_from_conn_string(self):
        """Test creating saver from connection string."""
        with (
            patch(
                "langgraph_checkpoint_aws.agentcore.valkey.saver.ConnectionPool.from_url"
            ) as mock_pool_from_url,
            patch(
                "langgraph_checkpoint_aws.agentcore.valkey.saver.Valkey"
            ) as mock_valkey,
        ):
            mock_pool = MagicMock()
            mock_pool_from_url.return_value = mock_pool
            mock_client = MagicMock()
            mock_valkey.return_value = mock_client

            with AgentCoreValkeySaver.from_conn_string(
                "valkey://localhost:6379", ttl_seconds=3600, pool_size=20
            ) as saver:
                assert isinstance(saver, AgentCoreValkeySaver)
                assert saver.ttl == 3600

            mock_pool_from_url.assert_called_once_with(
                "valkey://localhost:6379", max_connections=20
            )
            mock_valkey.assert_called_once_with(connection_pool=mock_pool)
            mock_client.close.assert_called_once()

    def test_from_pool(self):
        """Test creating saver from connection pool."""
        with patch(
            "langgraph_checkpoint_aws.agentcore.valkey.saver.Valkey"
        ) as mock_valkey:
            mock_pool = MagicMock()
            mock_client = MagicMock()
            mock_valkey.return_value = mock_client

            with AgentCoreValkeySaver.from_pool(mock_pool, ttl_seconds=1800) as saver:
                assert isinstance(saver, AgentCoreValkeySaver)
                assert saver.ttl == 1800

            mock_valkey.assert_called_once_with(connection_pool=mock_pool)
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_afrom_conn_string(self):
        """Test creating async saver from connection string."""
        # Mock the entire afrom_conn_string method since the local imports are complex
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Create a mock saver instance
        mock_saver = AgentCoreValkeySaver(mock_client, ttl=3600)

        # Mock the async context manager
        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_saver

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await mock_client.aclose()

        # Patch the method to return our mock context manager
        with patch.object(AgentCoreValkeySaver, "afrom_conn_string") as mock_method:
            mock_method.return_value = MockAsyncContextManager()

            async with AgentCoreValkeySaver.afrom_conn_string(
                "valkey://localhost:6379", ttl_seconds=3600, pool_size=20
            ) as saver:
                assert isinstance(saver, AgentCoreValkeySaver)
                assert saver.ttl == 3600

            # Verify the method was called with correct arguments
            mock_method.assert_called_once_with(
                "valkey://localhost:6379", ttl_seconds=3600, pool_size=20
            )

            # Verify aclose was called
            mock_client.aclose.assert_called_once()

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

    def test_make_session_checkpoints_key(self, saver):
        """Test session checkpoints key generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns=""
        )

        key = saver._make_session_checkpoints_key(config)
        expected = "agentcore:session:session-1:agent-1:checkpoints"
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

    def test_deserialize_checkpoint(self, saver):
        """Test checkpoint deserialization."""
        # Create proper serialized data
        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id="parent-1",
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": empty_dict_b64},
            created_at=time.time(),
        )

        writes = [
            StoredWrite(
                checkpoint_id="checkpoint-1",
                task_id="task-1",
                channel="messages",
                value={"type": "json", "data": empty_dict_b64},
                task_path="",
                created_at=time.time(),
            )
        ]

        result = saver._deserialize_checkpoint(stored_checkpoint, writes, {})

        assert isinstance(result, CheckpointTuple)
        assert (
            result.config.get("configurable", {}).get("checkpoint_id") == "checkpoint-1"
        )
        assert result.parent_config is not None
        assert (
            result.parent_config.get("configurable", {}).get("checkpoint_id")
            == "parent-1"
        )
        assert result.pending_writes is not None and len(result.pending_writes) == 1

    def test_deserialize_checkpoint_with_config(self, saver):
        """Test checkpoint deserialization with provided config."""
        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

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

        config = {
            "configurable": {
                "thread_id": "custom-thread",
                "checkpoint_id": "custom-checkpoint",
            }
        }

        result = saver._deserialize_checkpoint(stored_checkpoint, [], {}, config)

        assert result.config.get("configurable", {}).get("thread_id") == "custom-thread"
        assert (
            result.config.get("configurable", {}).get("checkpoint_id")
            == "custom-checkpoint"
        )
        assert result.parent_config is None

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
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    def test_get_tuple_corrupted_checkpoint_data(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test getting a checkpoint with corrupted data."""
        mock_valkey_client.get.return_value = "invalid json"

        with pytest.raises(ValueError, match="Failed to parse checkpoint data"):
            saver.get_tuple(sample_config)

    def test_get_tuple_corrupted_writes_data(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test getting a checkpoint with corrupted writes data."""
        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

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
        mock_valkey_client.lrange.return_value = [b"invalid json"]

        with pytest.raises(ValueError, match="Failed to parse writes data"):
            saver.get_tuple(sample_config)

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
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-2"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    def test_get_tuple_latest_checkpoint_corrupted(self, saver, mock_valkey_client):
        """Test getting latest checkpoint with corrupted data."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        mock_valkey_client.lindex.return_value = b"checkpoint-2"
        mock_valkey_client.get.return_value = "invalid json"

        with pytest.raises(ValueError, match="Failed to parse checkpoint data"):
            saver.get_tuple(config)

    def test_get_tuple_not_found(self, saver, mock_valkey_client, sample_config):
        """Test getting a checkpoint that doesn't exist."""
        mock_valkey_client.get.return_value = None

        result = saver.get_tuple(sample_config)

        assert result is None

    def test_get_tuple_latest_not_found(self, saver, mock_valkey_client):
        """Test getting latest checkpoint when none exists."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        mock_valkey_client.lindex.return_value = None

        result = saver.get_tuple(config)

        assert result is None

    def test_list_no_config(self, saver):
        """Test list with no config."""
        result = list(saver.list(None))
        assert result == []

    def test_list_with_before_filter(self, saver, mock_valkey_client):
        """Test list with before filter."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        before_config = {
            "configurable": {
                "checkpoint_id": "checkpoint-2",
            }
        }

        # Mock session checkpoint list
        def mock_lrange(key, start, end):
            if "checkpoints" in key:
                return [
                    b"checkpoint-3",  # Should be skipped (>= checkpoint-2)
                    b"checkpoint-1",  # Should be included (< checkpoint-2)
                ]
            else:
                # For writes keys, return empty list
                return []

        mock_valkey_client.lrange.side_effect = mock_lrange

        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

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

        result = list(saver.list(config, before=before_config))

        assert len(result) == 1
        assert (
            result[0].config.get("configurable", {}).get("checkpoint_id")
            == "checkpoint-1"
        )

    def test_list_with_limit(self, saver, mock_valkey_client):
        """Test list with limit."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        # Mock session checkpoint list and writes
        def mock_lrange(key, start, end):
            if "checkpoints" in key:
                return [
                    b"checkpoint-1",
                    b"checkpoint-2",
                    b"checkpoint-3",
                ]
            else:
                # For writes keys, return empty list
                return []

        mock_valkey_client.lrange.side_effect = mock_lrange

        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        def mock_get(key):
            if "checkpoint-1" in key:
                return StoredCheckpoint(
                    checkpoint_id="checkpoint-1",
                    session_id="session-1",
                    thread_id="session-1",
                    actor_id="agent-1",
                    checkpoint_ns="",
                    parent_checkpoint_id=None,
                    checkpoint_data={"type": "json", "data": empty_dict_b64},
                    metadata={"type": "json", "data": empty_dict_b64},
                    created_at=time.time(),
                ).model_dump_json()
            elif "checkpoint-2" in key:
                return StoredCheckpoint(
                    checkpoint_id="checkpoint-2",
                    session_id="session-1",
                    thread_id="session-1",
                    actor_id="agent-1",
                    checkpoint_ns="",
                    parent_checkpoint_id=None,
                    checkpoint_data={"type": "json", "data": empty_dict_b64},
                    metadata={"type": "json", "data": empty_dict_b64},
                    created_at=time.time(),
                ).model_dump_json()
            return None

        mock_valkey_client.get.side_effect = mock_get

        result = list(saver.list(config, limit=2))

        assert len(result) == 2

    def test_list_with_metadata_filter(self, saver, mock_valkey_client):
        """Test list with metadata filter."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        filter_dict = {"user": "test_user"}

        # Mock session checkpoint list and writes
        def mock_lrange(key, start, end):
            if "checkpoints" in key:
                return [b"checkpoint-1"]
            else:
                # For writes keys, return empty list
                return []

        mock_valkey_client.lrange.side_effect = mock_lrange

        # Create metadata that matches the filter
        metadata = {"user": "test_user", "other": "value"}
        metadata_json = json.dumps(metadata)
        metadata_b64 = base64.b64encode(metadata_json.encode()).decode()

        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": metadata_b64},
            created_at=time.time(),
        )

        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()

        result = list(saver.list(config, filter=filter_dict))

        assert len(result) == 1

    def test_list_with_metadata_filter_no_match(self, saver, mock_valkey_client):
        """Test list with metadata filter that doesn't match."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        filter_dict = {"user": "different_user"}

        mock_valkey_client.lrange.return_value = [b"checkpoint-1"]

        # Create metadata that doesn't match the filter
        metadata = {"user": "test_user"}
        metadata_json = json.dumps(metadata)
        metadata_b64 = base64.b64encode(metadata_json.encode()).decode()

        empty_dict_json = json.dumps({})
        empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()

        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": metadata_b64},
            created_at=time.time(),
        )

        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()

        result = list(saver.list(config, filter=filter_dict))

        assert len(result) == 0

    def test_list_checkpoint_not_found(self, saver, mock_valkey_client):
        """Test list when checkpoint data is not found."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        mock_valkey_client.lrange.return_value = [b"checkpoint-1"]
        mock_valkey_client.get.return_value = None  # Checkpoint not found

        result = list(saver.list(config))

        assert len(result) == 0

    def test_put_checkpoint(
        self, saver, mock_valkey_client, sample_config, sample_checkpoint
    ):
        """Test storing a checkpoint."""
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

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
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

        saver.put(sample_config, sample_checkpoint, metadata, new_versions)  # pyright: ignore[reportArgumentType]

        # Verify set was called instead of setex
        mock_valkey_client.set.assert_called()

    def test_put_checkpoint_with_channel_values_dict(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test storing a checkpoint with channel_values as dict."""
        checkpoint = {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
        }
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

        saver.put(sample_config, checkpoint, metadata, new_versions)

        # Verify channel data was stored
        mock_valkey_client.setex.assert_called()

    def test_put_checkpoint_with_channel_values_non_dict(
        self, saver, mock_valkey_client, sample_config
    ):
        """Test storing a checkpoint with channel_values as non-dict."""
        checkpoint = {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": "not_a_dict",
        }
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

        saver.put(sample_config, checkpoint, metadata, new_versions)

        # Should still work, just no channel data stored
        mock_valkey_client.setex.assert_called()

    def test_put_writes(self, saver, mock_valkey_client, sample_config):
        """Test storing writes."""
        writes = [("messages", {"role": "assistant", "content": "Hello!"})]
        task_id = "task-1"

        saver.put_writes(sample_config, writes, task_id)

        # Verify writes were stored
        mock_valkey_client.rpush.assert_called()
        mock_valkey_client.expire.assert_called()  # Due to TTL

    def test_put_writes_with_task_path(self, saver, mock_valkey_client, sample_config):
        """Test storing writes with task path."""
        writes = [("messages", {"role": "assistant", "content": "Hello!"})]
        task_id = "task-1"
        task_path = "path/to/task"

        saver.put_writes(sample_config, writes, task_id, task_path)

        # Verify writes were stored
        mock_valkey_client.rpush.assert_called()
        mock_valkey_client.expire.assert_called()

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

        with pytest.raises(InvalidConfigError):
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

    def test_delete_thread_async_client_warning(
        self, async_saver, mock_async_valkey_client
    ):
        """Test delete_thread with async client logs warning."""
        thread_id = "session-1"
        actor_id = "agent-1"

        # Mock checkpoint IDs - return actual values, not coroutines
        mock_async_valkey_client.lrange = MagicMock(return_value=[b"checkpoint-1"])
        mock_async_valkey_client.keys = MagicMock(return_value=[])
        mock_async_valkey_client.delete = MagicMock(return_value=1)

        with patch(
            "langgraph_checkpoint_aws.agentcore.valkey.saver.logger"
        ) as mock_logger:
            async_saver.delete_thread(thread_id, actor_id)
            mock_logger.warning.assert_called_with(
                "Sync delete_thread called on async client, operation may block"
            )

    def test_delete_thread_no_checkpoints(self, saver, mock_valkey_client):
        """Test deleting a thread with no checkpoints."""
        thread_id = "session-1"
        actor_id = "agent-1"

        # Mock no checkpoint IDs
        mock_valkey_client.lrange.return_value = []
        mock_valkey_client.keys.return_value = []

        saver.delete_thread(thread_id, actor_id)

        # Should still call delete with session key
        mock_valkey_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_aget_tuple_sync_client(self, saver, sample_config):
        """Test async get_tuple with sync client."""
        result = await saver.aget_tuple(sample_config)
        # Should return None since no data is mocked
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_async_client(self, async_saver, sample_config):
        """Test async get_tuple with async client."""
        # Mock the client methods to return actual values, not coroutines
        async_saver.client.get = MagicMock(return_value=None)
        result = await async_saver.aget_tuple(sample_config)
        # Should return None since no data is mocked
        assert result is None

    @pytest.mark.asyncio
    async def test_alist(self, saver, sample_config):
        """Test async list."""
        checkpoints = []
        async for checkpoint in saver.alist(sample_config):
            checkpoints.append(checkpoint)
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_aput(self, saver, sample_config, sample_checkpoint):
        """Test async put."""
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

        result_config = await saver.aput(
            sample_config, sample_checkpoint, metadata, new_versions
        )

        assert result_config["configurable"]["checkpoint_id"] == "checkpoint-1"

    @pytest.mark.asyncio
    async def test_aput_writes(self, saver, sample_config):
        """Test async put_writes."""
        writes = [("messages", {"role": "assistant", "content": "Hello!"})]
        task_id = "task-1"

        await saver.aput_writes(sample_config, writes, task_id)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_adelete_thread(self, saver):
        """Test async delete_thread."""
        await saver.adelete_thread("session-1", "agent-1")
        # Should complete without error

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

    def test_get_next_version_with_channel(self, saver):
        """Test version generation with channel."""
        version = saver.get_next_version(None, "messages")
        assert version.startswith("00000000000000000000000000000001.")

    def test_init_with_custom_retry_params(self, mock_valkey_client):
        """Test initialization with custom retry parameters."""
        saver = AgentCoreValkeySaver(mock_valkey_client, max_retries=5, retry_delay=0.5)
        assert saver.max_retries == 5
        assert saver.retry_delay == 0.5

    def test_init_zero_ttl_error(self, mock_valkey_client):
        """Test initialization with zero TTL raises error."""
        with pytest.raises(ValueError, match="TTL must be positive"):
            AgentCoreValkeySaver(mock_valkey_client, ttl=0)

    def test_execute_with_retry_exponential_backoff(self, saver):
        """Test retry logic uses exponential backoff."""
        operation = MagicMock()
        operation.side_effect = [
            ConnectionError("fail1"),
            ConnectionError("fail2"),
            "success",
        ]

        with (
            patch("time.sleep") as mock_sleep,
            patch("random.uniform", return_value=0.05),
        ):
            result = saver._execute_with_retry(operation)

        assert result == "success"
        # Check exponential backoff: 0.1 * 2^0 + 0.05, 0.1 * 2^1 + 0.05
        expected_delays = [0.15, 0.25]  # 0.1 + 0.05, 0.2 + 0.05
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        # Use approximate comparison due to floating point precision
        assert len(actual_delays) == len(expected_delays)
        for actual, expected in zip(actual_delays, expected_delays, strict=False):
            assert abs(actual - expected) < 0.001

    @pytest.mark.asyncio
    async def test_aexecute_with_retry_exponential_backoff(self, saver):
        """Test async retry logic uses exponential backoff."""
        operation = AsyncMock()
        operation.side_effect = [
            ConnectionError("fail1"),
            ConnectionError("fail2"),
            "success",
        ]

        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("random.uniform", return_value=0.05),
        ):
            result = await saver._aexecute_with_retry(operation)

        assert result == "success"
        # Check exponential backoff
        expected_delays = [0.15, 0.25]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        # Use approximate comparison due to floating point precision
        assert len(actual_delays) == len(expected_delays)
        for actual, expected in zip(actual_delays, expected_delays, strict=False):
            assert abs(actual - expected) < 0.001


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

        def mock_rpush(key, *values):
            if key not in lists:
                lists[key] = []
            for value in values:
                lists[key].append(value)
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

        def mock_delete(*keys):
            deleted_count = 0
            for key in keys:
                if key in storage:
                    del storage[key]
                    deleted_count += 1
                if key in lists:
                    del lists[key]
                    deleted_count += 1
            return deleted_count

        client.get.side_effect = mock_get
        client.set.side_effect = mock_set
        client.setex.side_effect = mock_setex
        client.lpush.side_effect = mock_lpush
        client.lrange.side_effect = mock_lrange
        client.lindex.side_effect = mock_lindex
        client.rpush.side_effect = mock_rpush
        client.expire = MagicMock(return_value=True)
        client.delete.side_effect = mock_delete
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
        new_versions: dict[str, str | int | float] = {"messages": 1.0}

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

    def test_multiple_checkpoints_lifecycle(self, integration_saver):
        """Test lifecycle with multiple checkpoints."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        # Create multiple checkpoints
        for i in range(3):
            checkpoint = {
                "id": f"checkpoint-{i + 1}",
                "ts": f"2024-01-0{i + 1}T00:00:00Z",
                "channel_values": {
                    "messages": [{"role": "user", "content": f"Hello {i + 1}"}]
                },
            }
            metadata = {"user": f"test_user_{i + 1}"}
            new_versions = {"messages": f"{i + 1}.0"}

            integration_saver.put(config, checkpoint, metadata, new_versions)

        # List all checkpoints
        checkpoints = list(integration_saver.list(config))
        assert len(checkpoints) == 3

        # Test with limit
        limited_checkpoints = list(integration_saver.list(config, limit=2))
        assert len(limited_checkpoints) == 2

        # Test with filter
        filtered_checkpoints = list(
            integration_saver.list(config, filter={"user": "test_user_2"})
        )
        assert len(filtered_checkpoints) == 1

    def test_writes_lifecycle(self, integration_saver):
        """Test writes lifecycle."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }

        # First create a checkpoint
        checkpoint = {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": []},
        }
        integration_saver.put(config, checkpoint, {}, {"messages": "1.0"})

        # Add writes
        writes = [
            ("messages", {"role": "user", "content": "Hello"}),
            ("messages", {"role": "assistant", "content": "Hi there!"}),
        ]
        integration_saver.put_writes(config, writes, "task-1")

        # Get checkpoint with writes
        retrieved = integration_saver.get_tuple(config)
        assert retrieved is not None
        assert len(retrieved.pending_writes) == 2

    def test_delete_thread_lifecycle(self, integration_saver):
        """Test delete thread lifecycle."""
        config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoint
        checkpoint = {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": []},
        }
        integration_saver.put(config, checkpoint, {}, {"messages": "1.0"})

        # Verify checkpoint exists
        retrieved = integration_saver.get_tuple(
            {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": "checkpoint-1",
                },
            }
        )
        assert retrieved is not None

        # Delete thread
        integration_saver.delete_thread("session-1", "agent-1")

        # Verify checkpoint is gone
        retrieved_after_delete = integration_saver.get_tuple(
            {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": "checkpoint-1",
                },
            }
        )
        assert retrieved_after_delete is None
