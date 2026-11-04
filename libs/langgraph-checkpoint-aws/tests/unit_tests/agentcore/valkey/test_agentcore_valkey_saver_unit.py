"""Tests for AgentCore Valkey checkpoint saver."""

import base64
import hashlib
import json
import time
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("valkey")
pytest.importorskip("orjson")

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from valkey.exceptions import ConnectionError, TimeoutError, ValkeyError

from langgraph_checkpoint_aws.checkpoint.agentcore.constants import InvalidConfigError
from langgraph_checkpoint_aws.checkpoint.valkey import AgentCoreValkeySaver
from langgraph_checkpoint_aws.checkpoint.valkey.agentcore.models import (
    StoredCheckpoint,
    StoredWrite,
    ValkeyCheckpointerConfig,
)


def _explicit_config(thread_id: str, actor_id: str) -> dict:
    """Build a top-level config carrying an explicit actor_id."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id,
            "checkpoint_ns": "",
        }
    }


def _subgraph_config(thread_id: str, checkpoint_ns: str) -> dict:
    """Build a derived subgraph config that omits actor_id.

    Mirrors the shape LangGraph derives during
    ``get_state(config, subgraphs=True)``: same thread_id, a non-empty
    checkpoint_ns naming the subgraph, and no actor_id.
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def _stub_stored_checkpoint(
    mock_valkey_client: MagicMock,
    *,
    checkpoint_id: str,
    thread_id: str,
    actor_id: str,
    checkpoint_ns: str,
    via_list: bool = False,
) -> None:
    """Stub a mock Valkey client so a read resolves to one stored checkpoint.

    Mirrors the fixture data built in ``test_get_tuple_latest_checkpoint``: an
    empty-dict checkpoint/metadata payload, no parent, and no writes, with the
    given ``actor_id``/``thread_id``/``checkpoint_ns`` baked into the stored
    record so the caller can assert the resolved actor_id was actually used to
    locate (and is reflected in) the returned data.

    ``get_tuple`` looks up the latest checkpoint id via ``lindex`` and has no
    writes lookup to disambiguate, so ``lrange`` can just return ``[]``
    (empty writes). ``list()`` instead reads the checkpoint-id list via
    ``lrange`` for the session key, then reads writes via ``lrange`` again
    for each checkpoint id -- ``via_list=True`` uses ``side_effect`` to return
    the id list on the first call and no writes on every call after.
    """
    empty_dict_json = json.dumps({})
    empty_dict_b64 = base64.b64encode(empty_dict_json.encode()).decode()
    stored_checkpoint = StoredCheckpoint(
        checkpoint_id=checkpoint_id,
        session_id=thread_id,
        thread_id=thread_id,
        actor_id=actor_id,
        checkpoint_ns=checkpoint_ns,
        parent_checkpoint_id=None,
        checkpoint_data={"type": "json", "data": empty_dict_b64},
        metadata={"type": "json", "data": empty_dict_b64},
        created_at=time.time(),
    )
    mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()
    if via_list:
        mock_valkey_client.lrange.side_effect = [[checkpoint_id.encode()], []]
    else:
        mock_valkey_client.lindex.return_value = checkpoint_id.encode()
        mock_valkey_client.lrange.return_value = []


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
                "langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver.ConnectionPool.from_url"
            ) as mock_pool_from_url,
            patch(
                "langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver.Valkey"
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
            "langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver.Valkey"
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
        # With checkpoint_ns set, session_id is shortened (SHA-256 hex) for AWS
        expected_session_id = hashlib.sha256(b"session-1_test-ns").hexdigest()
        expected = (
            f"agentcore:checkpoint:{expected_session_id}:agent-1:test-ns:checkpoint-1"
        )
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
            "langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver.logger"
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

    def test_sync_single_level_subgraph_inherits_actor_id_raises(
        self, saver, mock_valkey_client
    ):
        """Fixed behavior: sync get_tuple at one subgraph level.

        Property 1 (Requirements 2.1, 2.2): a top-level get_tuple call with
        an explicit actor_id records a capture for "t1", so the follow-up
        single-level subgraph get_tuple call -- same thread_id, non-empty
        checkpoint_ns, no actor_id -- now resolves the inherited "agent-1"
        instead of raising InvalidConfigError, and returns the checkpoint
        stored under that actor_id.
        """
        explicit_config = _explicit_config("t1", "agent-1")
        saver.get_tuple(explicit_config)

        subgraph_config = _subgraph_config("t1", "child1")
        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
        )
        result = saver.get_tuple(subgraph_config)
        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    def test_sync_deeply_nested_subgraph_raises_at_first_level(
        self, saver, mock_valkey_client
    ):
        """Fixed behavior: sync get_tuple across 3+ nested subgraph levels.

        Property 1 (Requirements 2.1, 2.3): after the same top-level
        capture, every one of three nested checkpoint_ns levels ("child1",
        "child1|child2", "child1|child2|child3") now resolves the inherited
        "agent-1" and returns its checkpoint, confirming the fix applies at
        every nesting level, not just the first.
        """
        explicit_config = _explicit_config("t1", "agent-1")
        saver.get_tuple(explicit_config)

        nested_namespaces = ["child1", "child1|child2", "child1|child2|child3"]
        for checkpoint_ns in nested_namespaces:
            subgraph_config = _subgraph_config("t1", checkpoint_ns)
            checkpoint_id = f"checkpoint-{checkpoint_ns}"
            _stub_stored_checkpoint(
                mock_valkey_client,
                checkpoint_id=checkpoint_id,
                thread_id="t1",
                actor_id="agent-1",
                checkpoint_ns=checkpoint_ns,
            )
            result = saver.get_tuple(subgraph_config)
            assert isinstance(result, CheckpointTuple)
            assert result.config["configurable"]["checkpoint_id"] == checkpoint_id  # pyright: ignore[reportTypedDictNotRequiredAccess]

    def test_sync_list_subgraph_inherits_actor_id_raises(
        self, saver, mock_valkey_client
    ):
        """Fixed behavior: sync list() for a derived subgraph config.

        Property 1 (Requirements 2.1, 2.2): list() is a generator, so this
        must be consumed to observe the result. Mirrors the get_tuple fixed
        behavior through the list() read path -- the inherited actor_id is
        now resolved and the subgraph's checkpoint is returned instead of
        raising InvalidConfigError.
        """
        explicit_config = _explicit_config("t1", "agent-1")
        list(saver.list(explicit_config))

        subgraph_config = _subgraph_config("t1", "child1")
        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
            via_list=True,
        )
        results = list(saver.list(subgraph_config))
        assert len(results) == 1
        assert isinstance(results[0], CheckpointTuple)
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert results[0].config["configurable"]["actor_id"] == "agent-1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    @pytest.mark.asyncio
    async def test_async_aget_tuple_subgraph_inherits_actor_id_raises(
        self, saver, mock_valkey_client
    ):
        """Fixed behavior: async aget_tuple for a derived subgraph config.

        Property 1 (Requirements 2.1, 2.4): aget_tuple now primes
        _resolve_actor_id in its own async context before dispatching via
        run_in_executor (which copies that context into the worker thread),
        so the actor_id captured by the prior explicit-actor read is visible
        to the nested subgraph read and the checkpoint is returned instead
        of raising InvalidConfigError.
        """
        explicit_config = _explicit_config("t1", "agent-1")
        await saver.aget_tuple(explicit_config)

        subgraph_config = _subgraph_config("t1", "child1")
        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
        )
        result = await saver.aget_tuple(subgraph_config)
        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    @pytest.mark.asyncio
    async def test_async_alist_subgraph_inherits_actor_id_raises(
        self, saver, mock_valkey_client
    ):
        """Fixed behavior: async alist for a derived subgraph config.

        Property 1 (Requirements 2.1, 2.4): alist dispatches the sync
        list() generator via run_in_executor after priming
        _resolve_actor_id in the calling async context, so the inherited
        actor_id is now visible to the worker thread. alist is an async
        generator, so it must be consumed (via `async for`) to observe the
        result.
        """
        explicit_config = _explicit_config("t1", "agent-1")
        async for _ in saver.alist(explicit_config):
            pass

        subgraph_config = _subgraph_config("t1", "child1")
        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
            via_list=True,
        )
        results = []
        async for item in saver.alist(subgraph_config):
            results.append(item)
        assert len(results) == 1
        assert isinstance(results[0], CheckpointTuple)
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert results[0].config["configurable"]["actor_id"] == "agent-1"  # pyright: ignore[reportTypedDictNotRequiredAccess]


class TestAgentCoreValkeySaverPreservation:
    """Property 2: Preservation - non-subgraph and unresolvable reads/writes
    fail closed.

    Observed on UNFIXED code (no ``_resolve_actor_id``/``_request_actor``
    machinery exists yet, so every one of these cases is either already
    correct behavior with no capture/fallback involved, or trivially raises
    because no fallback exists). These tests encode that observed baseline
    so they can be re-run unchanged after the fix (task 7) to confirm no
    regressions.
    """

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
        client.aclose = None
        client.__aenter__ = None
        return client

    @pytest.fixture
    def saver(self, mock_valkey_client):
        """Create a AgentCoreValkeySaver instance with mocked client."""
        return AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

    @staticmethod
    def _top_level_config_missing_actor(thread_id: str) -> dict:
        """Top-level config (empty checkpoint_ns) with no actor_id."""
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

    @staticmethod
    def _write_config(
        thread_id: str, actor_id: str | None, checkpoint_ns: str = ""
    ) -> dict:
        """Build a config shape suitable for put/put_writes, optionally
        omitting actor_id."""
        configurable = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": "checkpoint-1",
        }
        if actor_id is not None:
            configurable["actor_id"] = actor_id
        return {"configurable": configurable}

    @staticmethod
    def _sample_checkpoint() -> dict:
        return {
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": [{"role": "user", "content": "Hello"}]},
            "channel_versions": {"messages": "1.0"},
        }

    @staticmethod
    def _new_mock_valkey_client() -> MagicMock:
        """Build a fresh mock Valkey client.

        Used (instead of the function-scoped ``mock_valkey_client`` fixture)
        by standalone parametrized tests that construct their own saver
        instance per case rather than relying on function-scoped fixtures.
        """
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
        client.aclose = None
        client.__aenter__ = None
        return client

    # -- Explicit Actor Preservation -----------------------------------

    def test_get_tuple_explicit_actor_id_returns_data(self, saver, mock_valkey_client):
        """get_tuple with an explicit actor_id returns the expected
        checkpoint data -- no capture/fallback involved."""
        mock_valkey_client.lindex.return_value = b"checkpoint-1"
        empty_dict_b64 = base64.b64encode(json.dumps({}).encode()).decode()
        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="t1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": empty_dict_b64},
            created_at=time.time(),
        )
        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()
        mock_valkey_client.lrange.return_value = []

        explicit_config = _explicit_config("t1", "agent-1")
        result = saver.get_tuple(explicit_config)

        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-1"

    def test_list_explicit_actor_id_returns_data(self, saver, mock_valkey_client):
        """list() with an explicit actor_id returns the expected checkpoint
        data -- no capture/fallback involved."""
        mock_valkey_client.keys.return_value = [b"some:key:checkpoint-1"]
        empty_dict_b64 = base64.b64encode(json.dumps({}).encode()).decode()
        stored_checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="t1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id=None,
            checkpoint_data={"type": "json", "data": empty_dict_b64},
            metadata={"type": "json", "data": empty_dict_b64},
            created_at=time.time(),
        )
        mock_valkey_client.get.return_value = stored_checkpoint.model_dump_json()
        # First lrange call lists session checkpoint ids; second fetches
        # writes for that checkpoint (empty here).
        mock_valkey_client.lrange.side_effect = [[b"checkpoint-1"], []]

        explicit_config = _explicit_config("t1", "agent-1")
        results = list(saver.list(explicit_config))

        assert len(results) == 1
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint-1"

    # -- Top-Level Missing Actor Preservation ---------------------------

    def test_get_tuple_top_level_missing_actor_raises(self, saver):
        """Top-level read (empty checkpoint_ns) missing actor_id raises,
        trivially true on unfixed code since no fallback exists yet."""
        config = self._top_level_config_missing_actor("t1")
        with pytest.raises(InvalidConfigError):
            saver.get_tuple(config)

    def test_list_top_level_missing_actor_raises(self, saver):
        config = self._top_level_config_missing_actor("t1")
        with pytest.raises(InvalidConfigError):
            list(saver.list(config))

    @pytest.mark.asyncio
    async def test_aget_tuple_top_level_missing_actor_raises(self, saver):
        config = self._top_level_config_missing_actor("t1")
        with pytest.raises(InvalidConfigError):
            await saver.aget_tuple(config)

    @pytest.mark.asyncio
    async def test_alist_top_level_missing_actor_raises(self, saver):
        config = self._top_level_config_missing_actor("t1")
        with pytest.raises(InvalidConfigError):
            async for _ in saver.alist(config):
                pass

    # -- Mismatched Thread Preservation ----------------------------------

    def test_get_tuple_mismatched_thread_raises(self, saver):
        """A subgraph-shaped config for a thread_id never seen before
        raises, regardless of any other thread's prior capture."""
        explicit_config = _explicit_config("t1", "agent-1")
        saver.get_tuple(explicit_config)

        subgraph_config = _subgraph_config("t2-never-seen", "child1")
        with pytest.raises(InvalidConfigError):
            saver.get_tuple(subgraph_config)

    def test_list_mismatched_thread_raises(self, saver):
        explicit_config = _explicit_config("t1", "agent-1")
        list(saver.list(explicit_config))

        subgraph_config = _subgraph_config("t2-never-seen", "child1")
        with pytest.raises(InvalidConfigError):
            list(saver.list(subgraph_config))

    # -- Write Path Preservation ------------------------------------------

    def test_put_missing_actor_id_raises(self, saver):
        config = self._write_config("t1", None)
        checkpoint = self._sample_checkpoint()
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}
        with pytest.raises(InvalidConfigError):
            saver.put(config, checkpoint, metadata, new_versions)

    def test_put_writes_missing_actor_id_raises(self, saver):
        config = self._write_config("t1", None)
        writes = [("messages", {"role": "assistant", "content": "hi"})]
        with pytest.raises(InvalidConfigError):
            saver.put_writes(config, writes, "task-1")

    @pytest.mark.asyncio
    async def test_aput_missing_actor_id_raises(self, saver):
        config = self._write_config("t1", None)
        checkpoint = self._sample_checkpoint()
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}
        with pytest.raises(InvalidConfigError):
            await saver.aput(config, checkpoint, metadata, new_versions)

    @pytest.mark.asyncio
    async def test_aput_writes_missing_actor_id_raises(self, saver):
        config = self._write_config("t1", None)
        writes = [("messages", {"role": "assistant", "content": "hi"})]
        with pytest.raises(InvalidConfigError):
            await saver.aput_writes(config, writes, "task-1")

    def test_put_missing_actor_id_raises_immediately_after_read_capture(self, saver):
        """A write with missing actor_id raises even immediately after a
        read recorded a capture for the same thread_id -- writes never
        inherit."""
        explicit_config = _explicit_config("t1", "agent-1")
        saver.get_tuple(explicit_config)

        write_config = self._write_config("t1", None, checkpoint_ns="child1")
        checkpoint = self._sample_checkpoint()
        metadata = {"user": "test"}
        new_versions: dict[str, str | int | float] = {"messages": 1.0}
        with pytest.raises(InvalidConfigError):
            saver.put(write_config, checkpoint, metadata, new_versions)

    def test_put_writes_missing_actor_id_raises_immediately_after_read_capture(
        self, saver
    ):
        explicit_config = _explicit_config("t1", "agent-1")
        saver.get_tuple(explicit_config)

        write_config = self._write_config("t1", None, checkpoint_ns="child1")
        writes = [("messages", {"role": "assistant", "content": "hi"})]
        with pytest.raises(InvalidConfigError):
            saver.put_writes(write_config, writes, "task-1")

    # -- Parametrized: fail-closed on missing actor_id ---------------------
    @pytest.mark.parametrize(
        "thread_id,actor_id,checkpoint_ns,is_write",
        [
            # Empty checkpoint_ns (top-level read) must fail closed even
            # though a capture exists for an unrelated thread_id.
            ("t1", "agent-1", "", False),
            # Non-empty checkpoint_ns (subgraph read) still fails closed
            # when no capture was recorded for *this* thread_id.
            ("t1", "agent-1", "child1", False),
            # Deeply nested checkpoint_ns namespace behaves the same way.
            ("t1", "agent-1", "child1|child2", False),
            # Writes never inherit, regardless of checkpoint_ns.
            ("t1", "agent-1", "", True),
            ("t1", "agent-1", "child1", True),
            # Distinct thread_id/actor_id values exercise the same cases.
            ("t-two", "agent-two", "child1", False),
            ("t-two", "agent-two", "child1", True),
        ],
    )
    def test_fails_closed_without_capture_when_actor_id_missing(
        self, thread_id, actor_id, checkpoint_ns, is_write
    ):
        """For a (thread_id, actor_id, checkpoint_ns) combination, a
        read/write omitting actor_id fails closed whenever checkpoint_ns is
        empty, the operation is a write, or no prior capture exists for
        thread_id -- regardless of capture state recorded for unrelated
        threads.
        """
        saver = AgentCoreValkeySaver(self._new_mock_valkey_client(), ttl=3600)
        # Record a capture for an unrelated thread_id so we can assert it
        # never leaks into this (thread_id, checkpoint_ns) combination.
        unrelated_thread_id = thread_id + "-unrelated"
        saver.get_tuple(_explicit_config(unrelated_thread_id, actor_id))
        if is_write:
            config = self._write_config(thread_id, None, checkpoint_ns)
            checkpoint = self._sample_checkpoint()
            metadata = {"user": "test"}
            new_versions: dict[str, str | int | float] = {"messages": 1.0}
            with pytest.raises(InvalidConfigError):
                saver.put(config, checkpoint, metadata, new_versions)
        else:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            }
            with pytest.raises(InvalidConfigError):
                saver.get_tuple(config)

    # -- Parametrized: captures never leak across distinct thread_ids ------
    @pytest.mark.parametrize(
        "thread_ids",
        [
            ["t1", "t2"],
            ["t1", "t2", "t3"],
            ["t1", "t2", "t3", "t4", "t5"],
        ],
    )
    def test_captures_never_leak_across_distinct_thread_ids(self, thread_ids):
        """Interleaved explicit-actor reads across multiple distinct
        thread_ids never let an *unrelated* thread_id's subgraph read
        inherit a capture recorded for a different thread_id.
        Only the most recently captured thread_id may legitimately inherit
        its own capture on a subsequent subgraph read for that same
        thread_id -- that is correct same-thread inheritance, not a leak,
        so it is excluded from this assertion.
        """
        saver = AgentCoreValkeySaver(self._new_mock_valkey_client(), ttl=3600)
        for idx, thread_id in enumerate(thread_ids):
            saver.get_tuple(_explicit_config(thread_id, f"agent-{idx}"))
        for thread_id in thread_ids[:-1]:
            subgraph_config = _subgraph_config(thread_id, "child1")
            with pytest.raises(InvalidConfigError):
                saver.get_tuple(subgraph_config)


class TestResolveActorId:
    """Direct unit tests for ``_resolve_actor_id``, called directly rather
    than through ``get_tuple``/``list``.

    Covers the request-scoped, fail-closed inheritance semantics: explicit
    ``actor_id`` capture, subgraph inheritance from a matching capture,
    fail-closed behavior on no/mismatched capture, fail-closed behavior on
    a top-level config, and a ``None`` config.
    """

    @pytest.fixture
    def mock_valkey_client(self) -> MagicMock:
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
        client.aclose = None
        client.__aenter__ = None
        return client

    @pytest.fixture
    def saver(self, mock_valkey_client: MagicMock) -> AgentCoreValkeySaver:
        """Create an AgentCoreValkeySaver instance with a mocked client."""
        return AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

    @pytest.fixture(autouse=True)
    def _reset_request_actor(self) -> Iterator[None]:
        """Isolate the module-level ``_request_actor`` ContextVar per test.

        Without this, a capture set by one test's direct
        ``_resolve_actor_id`` call could leak into another test running in
        the same thread/context.
        """
        from langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver import (
            _request_actor,
        )

        token = _request_actor.set(None)
        try:
            yield
        finally:
            _request_actor.reset(token)

    def test_explicit_actor_id_and_thread_id_records_and_returns(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """Explicit actor_id + thread_id present records the pair into
        ``_request_actor`` and returns the actor_id."""
        from langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver import (
            _request_actor,
        )

        config = cast(RunnableConfig, _explicit_config("t1", "agent-1"))

        result = saver._resolve_actor_id(config)

        assert result == "agent-1"
        assert _request_actor.get() == ("t1", "agent-1")

    def test_subgraph_config_with_matching_capture_inherits_actor_id(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """Subgraph-shaped config (non-empty checkpoint_ns, no actor_id)
        with a matching prior capture inherits and returns the recorded
        actor_id."""
        saver._resolve_actor_id(cast(RunnableConfig, _explicit_config("t1", "agent-1")))

        result = saver._resolve_actor_id(
            cast(RunnableConfig, _subgraph_config("t1", "child1"))
        )

        assert result == "agent-1"

    def test_subgraph_config_with_no_capture_returns_none(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """Subgraph-shaped config with no prior capture at all returns
        None."""
        result = saver._resolve_actor_id(
            cast(RunnableConfig, _subgraph_config("t1", "child1"))
        )

        assert result is None

    def test_subgraph_config_with_capture_for_different_thread_returns_none(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """Subgraph-shaped config with a capture recorded for a different
        thread_id returns None instead of inheriting it."""
        saver._resolve_actor_id(cast(RunnableConfig, _explicit_config("t1", "agent-1")))

        result = saver._resolve_actor_id(
            cast(RunnableConfig, _subgraph_config("t2-other", "child1"))
        )

        assert result is None

    def test_top_level_config_with_no_actor_id_returns_none_despite_capture(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """Top-level config (empty checkpoint_ns) with no actor_id returns
        None regardless of any existing capture for the same thread_id."""
        saver._resolve_actor_id(cast(RunnableConfig, _explicit_config("t1", "agent-1")))
        top_level_config_missing_actor = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                }
            },
        )

        result = saver._resolve_actor_id(top_level_config_missing_actor)

        assert result is None

    def test_none_config_returns_none_without_raising(
        self, saver: AgentCoreValkeySaver
    ) -> None:
        """A None config returns None without raising."""
        result = saver._resolve_actor_id(None)

        assert result is None


class TestRunInExecutorContextPropagation:
    """Regression tests for the ``run_in_executor`` context-propagation fix.

    ``aget_tuple``/``alist`` prime ``_resolve_actor_id`` in their own async
    context, then dispatch the sync read to a worker thread via
    ``langchain_core.runnables.run_in_executor``, which copies the calling
    ``contextvars.Context`` into that thread. A later subgraph read running in
    the worker thread therefore sees the ``_request_actor`` capture primed in
    the async context and inherits the actor_id.

    Each test primes the capture **only in the async context** (never inside a
    worker thread) and then makes a single dispatched subgraph read. This is
    what makes the test sensitive to the regression: if the dispatch were
    reverted to a raw ``asyncio.get_running_loop().run_in_executor(...)``, the
    worker thread would run without the async context's capture, so
    ``_request_actor.get()`` would be ``None`` there and the subgraph read would
    raise ``InvalidConfigError`` instead of returning the stored checkpoint.

    Priming inside a worker thread (e.g. by first awaiting an explicit-actor
    ``aget_tuple``) would NOT be a valid regression test: the default executor
    reuses its worker thread and a raw dispatch does not reset that thread's
    context, so a capture left by the first call would leak into the second
    call's reused worker and mask the bug.
    """

    @pytest.fixture
    def mock_valkey_client(self) -> MagicMock:
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
        client.aclose = None
        client.__aenter__ = None
        return client

    @pytest.fixture
    def saver(self, mock_valkey_client: MagicMock) -> AgentCoreValkeySaver:
        """Create an AgentCoreValkeySaver instance with a mocked client."""
        return AgentCoreValkeySaver(mock_valkey_client, ttl=3600)

    @pytest.fixture(autouse=True)
    def _reset_request_actor(self) -> Iterator[None]:
        """Isolate the module-level ``_request_actor`` ContextVar per test.

        Without this, a capture set by one test could leak into another test
        running in the same thread/context.
        """
        from langgraph_checkpoint_aws.checkpoint.valkey.agentcore.saver import (
            _request_actor,
        )

        token = _request_actor.set(None)
        try:
            yield
        finally:
            _request_actor.reset(token)

    @pytest.mark.asyncio
    async def test_aget_tuple_subgraph_inherits_async_context_actor(
        self, saver: AgentCoreValkeySaver, mock_valkey_client: MagicMock
    ) -> None:
        """``aget_tuple`` copies the async context into the worker thread.

        Prime the actor capture directly in the async context, then make a
        single ``aget_tuple`` subgraph read. The worker thread running
        ``get_tuple`` resolves the actor_id only because ``run_in_executor``
        copied the async context's capture into it; a raw ``asyncio`` executor
        dispatch would start the worker without that capture and the read would
        raise ``InvalidConfigError`` instead of returning the stored checkpoint.
        """
        explicit_config = cast(RunnableConfig, _explicit_config("t1", "agent-1"))
        saver._resolve_actor_id(explicit_config)

        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
        )
        subgraph_config = cast(RunnableConfig, _subgraph_config("t1", "child1"))
        result = await saver.aget_tuple(subgraph_config)

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    @pytest.mark.asyncio
    async def test_alist_subgraph_inherits_async_context_actor(
        self, saver: AgentCoreValkeySaver, mock_valkey_client: MagicMock
    ) -> None:
        """``alist`` copies the async context into the worker thread.

        Same regression coverage as the ``aget_tuple`` test above, but through
        the ``alist`` async-generator dispatch path, which also primes
        ``_resolve_actor_id`` before handing the sync ``list()`` generator to
        ``run_in_executor``.
        """
        explicit_config = cast(RunnableConfig, _explicit_config("t1", "agent-1"))
        saver._resolve_actor_id(explicit_config)

        _stub_stored_checkpoint(
            mock_valkey_client,
            checkpoint_id="checkpoint-child1",
            thread_id="t1",
            actor_id="agent-1",
            checkpoint_ns="child1",
            via_list=True,
        )
        subgraph_config = cast(RunnableConfig, _subgraph_config("t1", "child1"))
        results = []
        async for item in saver.alist(subgraph_config):
            results.append(item)

        assert len(results) == 1
        assert isinstance(results[0], CheckpointTuple)
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint-child1"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert results[0].config["configurable"]["actor_id"] == "agent-1"  # pyright: ignore[reportTypedDictNotRequiredAccess]


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
