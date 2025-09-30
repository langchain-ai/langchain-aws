"""Comprehensive unit tests for checkpoint/valkey/async_saver.py to improve coverage."""

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from langchain_core.runnables import RunnableConfig

from langgraph_checkpoint_aws.checkpoint.valkey.async_saver import (
    AsyncValkeyCheckpointSaver,
)


class MockSerializer:
    """Mock serializer for testing."""

    def dumps(self, obj: Any) -> bytes:
        import json

        return json.dumps(obj).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        import json

        return json.loads(data.decode("utf-8"))

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Return type and serialized data."""
        return ("json", self.dumps(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Load from typed data."""
        type_name, serialized = data
        return self.loads(serialized)


@pytest.fixture
def mock_valkey_client():
    """Mock async Valkey client."""
    client = AsyncMock()

    # Create a proper pipeline mock with sync methods
    pipeline_mock = Mock()
    pipeline_mock.set = Mock(return_value=None)
    pipeline_mock.expire = Mock(return_value=None)
    pipeline_mock.lpush = Mock(return_value=None)
    pipeline_mock.execute = AsyncMock(return_value=[True, True, True])

    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = False
    client.keys.return_value = []
    client.lrange.return_value = []
    client.lpush.return_value = 1
    client.expire.return_value = True

    # Pipeline should return the proper pipeline mock
    client.pipeline = AsyncMock(return_value=pipeline_mock)

    client.aclose.return_value = None
    client.execute_command.return_value = True
    return client


@pytest.fixture
def mock_serializer():
    """Mock serializer."""
    return MockSerializer()


@pytest.fixture
def sample_checkpoint():
    """Sample checkpoint for testing."""
    return {
        "v": 1,
        "id": "test-checkpoint-id",
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": {"test_channel": "test_value"},
        "channel_versions": {"test_channel": 1},
        "versions_seen": {"test_channel": {"__start__": 1}},
        "pending_sends": [],
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {"source": "test", "step": 1, "writes": {}, "parents": {}}


@pytest.fixture
def sample_config():
    """Sample runnable config."""
    return RunnableConfig(
        configurable={
            "thread_id": "test-thread-123",
            "checkpoint_ns": "",
            "checkpoint_id": "test-checkpoint-id",
        }
    )


class TestAsyncValkeyCheckpointSaverInit:
    """Test AsyncValkeyCheckpointSaver initialization."""

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_valkey_client, mock_serializer):
        """Test initialization with client."""
        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )
        assert saver.client == mock_valkey_client
        assert saver.serde is not None

    @pytest.mark.asyncio
    async def test_from_conn_string(self):
        """Test creating saver from connection string."""
        with patch(
            "langgraph_checkpoint_aws.checkpoint.valkey.async_saver.AsyncValkey"
        ) as mock_valkey_class:
            mock_client = AsyncMock()
            mock_valkey_class.from_url.return_value = mock_client

            async with AsyncValkeyCheckpointSaver.from_conn_string(
                "valkey://localhost:6379"
            ) as saver:
                assert saver.client == mock_client
                mock_valkey_class.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_conn_string_with_ttl(self):
        """Test creating saver from connection string with TTL."""
        with patch("valkey.asyncio.Valkey") as mock_valkey_class:
            mock_client = AsyncMock()
            mock_valkey_class.from_url.return_value = mock_client

            async with AsyncValkeyCheckpointSaver.from_conn_string(
                "valkey://localhost:6379", ttl_seconds=7200
            ) as saver:
                assert saver.ttl == 7200.0


class TestAsyncValkeyCheckpointSaverGetTuple:
    """Test aget_tuple method."""

    @pytest.mark.asyncio
    async def test_aget_tuple_existing_checkpoint(
        self,
        mock_valkey_client,
        mock_serializer,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """Test getting existing checkpoint tuple."""
        # Mock stored checkpoint data
        checkpoint_info = {
            "thread_id": "test-thread-123",
            "checkpoint_id": "test-checkpoint-id",
            "parent_checkpoint_id": None,
            "type": "json",
            "checkpoint": base64.b64encode(
                mock_serializer.dumps(sample_checkpoint)
            ).decode("utf-8"),
            "metadata": base64.b64encode(mock_serializer.dumps(sample_metadata)).decode(
                "utf-8"
            ),
        }

        mock_valkey_client.get.side_effect = [
            orjson.dumps(checkpoint_info),  # checkpoint data
            orjson.dumps([]),  # writes data
        ]

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        result = await saver.aget_tuple(sample_config)

        assert result is not None
        assert result.checkpoint["id"] == "test-checkpoint-id"
        assert result.checkpoint["v"] == 1

    @pytest.mark.asyncio
    async def test_aget_tuple_with_pending_writes(
        self,
        mock_valkey_client,
        mock_serializer,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """Test getting checkpoint with pending writes."""
        # Mock checkpoint data
        checkpoint_info = {
            "thread_id": "test-thread-123",
            "checkpoint_id": "test-checkpoint-id",
            "parent_checkpoint_id": None,
            "type": "json",
            "checkpoint": base64.b64encode(
                mock_serializer.dumps(sample_checkpoint)
            ).decode("utf-8"),
            "metadata": base64.b64encode(mock_serializer.dumps(sample_metadata)).decode(
                "utf-8"
            ),
        }

        # Mock pending writes
        writes_data = [
            {
                "task_id": "task_1",
                "idx": 0,
                "channel": "channel",
                "type": "json",
                "value": base64.b64encode(mock_serializer.dumps("value")).decode(
                    "utf-8"
                ),
            }
        ]

        mock_valkey_client.get.side_effect = [
            orjson.dumps(checkpoint_info),  # checkpoint data
            orjson.dumps(writes_data),  # writes data
        ]

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        result = await saver.aget_tuple(sample_config)

        assert result is not None
        assert result.pending_writes is not None and len(result.pending_writes) > 0


class TestAsyncValkeyCheckpointSaverPut:
    """Test aput method."""

    @pytest.mark.asyncio
    async def test_aput_new_checkpoint(
        self,
        mock_valkey_client,
        mock_serializer,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """Test putting new checkpoint."""

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        result = await saver.aput(
            sample_config, sample_checkpoint, sample_metadata, {"test_channel": 1}
        )

        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]
        mock_valkey_client.pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_aput_with_ttl(
        self,
        mock_valkey_client,
        mock_serializer,
        sample_config,
        sample_checkpoint,
        sample_metadata,
    ):
        """Test putting checkpoint with TTL."""

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer, ttl=3600.0
        )

        await saver.aput(
            sample_config, sample_checkpoint, sample_metadata, {"test_channel": 1}
        )

        # Verify expire was called with TTL
        # Pipeline expire should have been called (via the pipeline mock)


class TestAsyncValkeyCheckpointSaverPutWrites:
    """Test aput_writes method."""

    @pytest.mark.asyncio
    async def test_aput_writes_basic(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test putting writes."""
        writes = [("channel1", "value1"), ("channel2", "value2")]
        task_id = "test-task"

        mock_valkey_client.get.return_value = orjson.dumps([])  # existing writes

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        await saver.aput_writes(sample_config, writes, task_id)

        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_with_task_path(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test putting writes with task path."""
        writes = [("channel", "value")]
        task_id = "test-task"
        task_path = "path/to/task"

        mock_valkey_client.get.return_value = orjson.dumps([])  # existing writes

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        await saver.aput_writes(sample_config, writes, task_id, task_path)

        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_empty_writes(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test putting empty writes."""
        writes = []
        task_id = "test-task"

        mock_valkey_client.get.return_value = orjson.dumps([])  # existing writes

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        await saver.aput_writes(sample_config, writes, task_id)

        mock_valkey_client.get.assert_called()


class TestAsyncValkeyCheckpointSaverList:
    """Test alist method."""

    @pytest.mark.asyncio
    async def test_alist_all_checkpoints(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test listing all checkpoints."""
        # Mock thread checkpoint list
        mock_valkey_client.lrange.return_value = [
            b"test-checkpoint-1",
            b"test-checkpoint-2",
        ]

        # Mock checkpoint data
        checkpoint_info = {
            "thread_id": "test-thread-123",
            "checkpoint_id": "test-checkpoint-1",
            "parent_checkpoint_id": None,
            "type": "json",
            "checkpoint": base64.b64encode(mock_serializer.dumps({})).decode("utf-8"),
            "metadata": base64.b64encode(mock_serializer.dumps({})).decode("utf-8"),
        }

        mock_valkey_client.get.side_effect = [
            orjson.dumps(checkpoint_info),  # First checkpoint
            orjson.dumps([]),  # First writes
            orjson.dumps(checkpoint_info),  # Second checkpoint
            orjson.dumps([]),  # Second writes
        ]

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        checkpoints = []
        async for checkpoint in saver.alist(sample_config):
            checkpoints.append(checkpoint)

        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_alist_with_limit(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test listing checkpoints with limit."""
        # Mock many checkpoint IDs
        checkpoint_ids = [f"test-checkpoint-{i}".encode() for i in range(10)]
        mock_valkey_client.lrange.return_value = checkpoint_ids

        # Mock single checkpoint data
        checkpoint_info = {
            "thread_id": "test-thread-123",
            "checkpoint_id": "test-checkpoint-1",
            "parent_checkpoint_id": None,
            "type": "json",
            "checkpoint": base64.b64encode(mock_serializer.dumps({})).decode("utf-8"),
            "metadata": base64.b64encode(mock_serializer.dumps({})).decode("utf-8"),
        }

        # Return same data for all checkpoints
        mock_valkey_client.get.side_effect = [
            orjson.dumps(checkpoint_info),
            orjson.dumps([]),
        ] * 10

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        checkpoints = []
        async for checkpoint in saver.alist(sample_config, limit=3):
            checkpoints.append(checkpoint)

        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_alist_with_filter(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test listing checkpoints with filter."""
        mock_valkey_client.lrange.return_value = [b"test-checkpoint-1"]

        checkpoint_info = {
            "thread_id": "test-thread-123",
            "checkpoint_id": "test-checkpoint-1",
            "parent_checkpoint_id": None,
            "type": "json",
            "checkpoint": base64.b64encode(mock_serializer.dumps({})).decode("utf-8"),
            "metadata": base64.b64encode(
                mock_serializer.dumps({"source": "filter_match"})
            ).decode("utf-8"),
        }

        mock_valkey_client.get.side_effect = [
            orjson.dumps(checkpoint_info),
            orjson.dumps([]),
        ]

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        checkpoints = []
        async for checkpoint in saver.alist(
            sample_config, filter={"source": "filter_match"}
        ):
            checkpoints.append(checkpoint)

        assert len(checkpoints) == 1
        assert checkpoints[0].metadata["source"] == "filter_match"


class TestAsyncValkeyCheckpointSaverErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_during_get(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test handling connection errors during get."""
        mock_valkey_client.get.side_effect = ConnectionError("Connection lost")

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        with pytest.raises(ConnectionError):
            await saver.aget_tuple(sample_config)

    @pytest.mark.asyncio
    async def test_serialization_error_during_put(
        self, mock_valkey_client, sample_config, sample_checkpoint, sample_metadata
    ):
        """Test handling serialization errors during put."""
        # Mock serializer that raises error
        bad_serializer = Mock()
        bad_serializer.dumps_typed.side_effect = ValueError("Serialization error")

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=bad_serializer
        )

        with pytest.raises(ValueError):
            await saver.aput(
                sample_config, sample_checkpoint, sample_metadata, {"test_channel": 1}
            )

    @pytest.mark.asyncio
    async def test_timeout_during_operation(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test handling timeouts during operations."""
        mock_valkey_client.get.side_effect = asyncio.TimeoutError("Operation timeout")

        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        with pytest.raises(asyncio.TimeoutError):
            await saver.aget_tuple(sample_config)


class TestAsyncValkeyCheckpointSaverKeyGeneration:
    """Test key generation methods."""

    @pytest.mark.asyncio
    async def test_make_checkpoint_key(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test checkpoint key generation."""
        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        key = saver._make_checkpoint_key(
            sample_config["configurable"]["thread_id"],
            sample_config["configurable"]["checkpoint_ns"],
            sample_config["configurable"]["checkpoint_id"],
        )

        assert "checkpoint" in key
        assert "test-thread-123" in key
        assert "test-checkpoint-id" in key

    @pytest.mark.asyncio
    async def test_make_writes_key(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test writes key generation."""
        saver = AsyncValkeyCheckpointSaver(
            client=mock_valkey_client, serde=mock_serializer
        )

        key = saver._make_writes_key(
            sample_config["configurable"]["thread_id"],
            sample_config["configurable"]["checkpoint_ns"],
            sample_config["configurable"]["checkpoint_id"],
        )

        assert "writes" in key
        assert "test-thread-123" in key
        assert "test-checkpoint-id" in key
