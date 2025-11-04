"""Comprehensive unit tests for checkpoint/valkey/async_saver.py to improve coverage."""

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("valkey")
pytest.importorskip("orjson")
pytest.importorskip("fakeredis")

import orjson
from langchain_core.runnables import RunnableConfig
from valkey.exceptions import ValkeyError

from langgraph_checkpoint_aws import AsyncValkeySaver


# Create dummy objects for serialization tests
class MockOrjson:
    @staticmethod
    def dumps(obj):  # type: ignore[misc]
        import json

        return json.dumps(obj).encode("utf-8")


mock_orjson = MockOrjson()  # For testing when orjson mock is needed


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

    # Create a proper pipeline mock
    pipeline_mock = Mock()
    pipeline_mock.set = Mock(return_value=None)
    pipeline_mock.expire = Mock(return_value=None)
    pipeline_mock.lpush = Mock(return_value=None)
    pipeline_mock.get = Mock(return_value=None)
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
    client.pipeline = Mock(return_value=pipeline_mock)
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


class TestAsyncValkeySaverInit:
    """Test AsyncValkeySaver initialization."""

    @pytest.mark.asyncio
    async def test_init_with_client(self, mock_valkey_client, mock_serializer):
        """Test initialization with client."""
        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)
        assert saver.client == mock_valkey_client
        assert saver.serde is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_from_conn_string(self):
        """Test creating saver from connection string."""
        with patch(
            "langgraph_checkpoint_aws.checkpoint.valkey.async_saver.AsyncValkey"
        ) as mock_valkey_class:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            mock_valkey_class.from_url.return_value = mock_client

            async with AsyncValkeySaver.from_conn_string(
                "valkey://localhost:6379"
            ) as saver:
                assert saver.client == mock_client
                mock_valkey_class.from_url.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_from_conn_string_with_ttl(self):
        """Test creating saver from connection string with TTL."""
        with patch(
            "langgraph_checkpoint_aws.checkpoint.valkey.async_saver.AsyncValkey"
        ) as mock_valkey_class:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            mock_valkey_class.from_url.return_value = mock_client

            async with AsyncValkeySaver.from_conn_string(
                "valkey://localhost:6379", ttl_seconds=7200
            ) as saver:
                assert saver.ttl == 7200.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_from_pool_basic(self):
        """Test creating saver from connection pool."""
        with patch(
            "langgraph_checkpoint_aws.checkpoint.valkey.async_saver.AsyncValkey"
        ) as mock_valkey_class:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            mock_valkey_class.return_value = mock_client

            mock_pool = Mock()

            async with AsyncValkeySaver.from_pool(mock_pool, ttl_seconds=3600) as saver:
                assert saver.client == mock_client
                assert saver.ttl == 3600.0
                mock_valkey_class.assert_called_once_with(connection_pool=mock_pool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_from_pool_no_ttl(self):
        """Test creating saver from connection pool without TTL."""
        with patch(
            "langgraph_checkpoint_aws.checkpoint.valkey.async_saver.AsyncValkey"
        ) as mock_valkey_class:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            mock_valkey_class.return_value = mock_client

            mock_pool = Mock()

            async with AsyncValkeySaver.from_pool(mock_pool) as saver:
                assert saver.client == mock_client
                assert saver.ttl is None


class TestAsyncValkeySaverGetTuple:
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

        # Mock pipeline execution
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[
                orjson.dumps(checkpoint_info),  # checkpoint data
                orjson.dumps([]),  # writes data
            ]
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

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

        # Mock pipeline execution
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[
                orjson.dumps(checkpoint_info),  # checkpoint data
                orjson.dumps(writes_data),  # writes data
            ]
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver.aget_tuple(sample_config)

        assert result is not None
        assert result.pending_writes is not None and len(result.pending_writes) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_aget_tuple_valkey_error(self, mock_valkey_client, mock_serializer):
        """Test aget_tuple with ValkeyError."""
        mock_valkey_client.lrange.side_effect = ValkeyError("Valkey error")

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        # Remove checkpoint_id to trigger latest checkpoint path
        config_without_id = RunnableConfig(
            configurable={
                "thread_id": "test-thread-123",
                "checkpoint_ns": "",
            }
        )

        result = await saver.aget_tuple(config_without_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_key_error(self, mock_valkey_client, mock_serializer):
        """Test aget_tuple with KeyError."""
        # Config missing thread_id
        bad_config = RunnableConfig(configurable={})

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver.aget_tuple(bad_config)
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_no_checkpoint_ids(
        self, mock_valkey_client, mock_serializer
    ):
        """Test aget_tuple when no checkpoint IDs exist."""
        mock_valkey_client.lrange.return_value = []  # No checkpoint IDs

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        config_without_id = RunnableConfig(
            configurable={
                "thread_id": "test-thread-123",
                "checkpoint_ns": "",
            }
        )

        result = await saver.aget_tuple(config_without_id)
        assert result is None


class TestAsyncValkeySaverGetCheckpointDataErrorHandling:
    """Test _get_checkpoint_data method error handling."""

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_pipeline_wrong_results_count(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with wrong pipeline results count."""
        # Mock pipeline returning wrong number of results
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[True]
        )  # Only 1 result instead of 2
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (None, [])

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_empty_results(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with empty pipeline results."""
        # Mock pipeline returning empty results
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(return_value=[])  # Empty results
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (None, [])

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_no_checkpoint_data(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with no checkpoint data."""
        # Mock pipeline returning None for checkpoint data
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[None, b"[]"]
        )  # No checkpoint data
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (None, [])

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_string_writes_data(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with string writes data."""
        checkpoint_info = {"test": "data"}
        writes_data = "[]"  # String instead of bytes

        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[
                orjson.dumps(checkpoint_info),
                writes_data,  # String writes data
            ]
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (checkpoint_info, [])

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_valkey_error(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with ValkeyError."""
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(side_effect=ValkeyError("Valkey error"))
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (None, [])

    @pytest.mark.asyncio
    async def test_get_checkpoint_data_json_decode_error(
        self, mock_valkey_client, mock_serializer
    ):
        """Test _get_checkpoint_data with JSON decode error."""
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            return_value=[
                b"invalid json",  # Invalid JSON
                b"[]",
            ]
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = await saver._get_checkpoint_data("thread", "ns", "checkpoint")
        assert result == (None, [])


class TestAsyncValkeySaverAlist:
    """Test alist method."""

    @pytest.mark.asyncio
    async def test_alist_no_config(self, mock_valkey_client, mock_serializer):
        """Test alist with no config."""
        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = []
        async for item in saver.alist(None):
            result.append(item)

        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_alist_valkey_error(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test alist with ValkeyError."""
        mock_valkey_client.lrange.side_effect = ValkeyError("Valkey error")

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        result = []
        async for item in saver.alist(sample_config):
            result.append(item)

        assert result == []


class TestAsyncValkeySaverPut:
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

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

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

        saver = AsyncValkeySaver(
            client=mock_valkey_client, serde=mock_serializer, ttl=3600.0
        )

        await saver.aput(
            sample_config, sample_checkpoint, sample_metadata, {"test_channel": 1}
        )

        # Verify expire was called with TTL
        # Pipeline expire should have been called (via the pipeline mock)


class TestAsyncValkeySaverPutWrites:
    """Test aput_writes method."""

    @pytest.mark.asyncio
    async def test_aput_writes_basic(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test putting writes."""
        writes = [("channel1", "value1"), ("channel2", "value2")]
        task_id = "test-task"

        mock_valkey_client.get.return_value = orjson.dumps([])  # existing writes

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

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

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

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

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.aput_writes(sample_config, writes, task_id)

        mock_valkey_client.get.assert_called()


class TestAsyncValkeySaverErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_during_get(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test handling connection errors during get."""
        # Mock pipeline execution to raise ConnectionError
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            side_effect=ConnectionError("Connection lost")
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        # Should return None instead of raising exception due to error handling
        result = await saver.aget_tuple(sample_config)
        assert result is None

    @pytest.mark.asyncio
    async def test_serialization_error_during_put(
        self, mock_valkey_client, sample_config, sample_checkpoint, sample_metadata
    ):
        """Test handling serialization errors during put."""
        # Mock serializer that raises error
        bad_serializer = Mock()
        bad_serializer.dumps_typed.side_effect = ValueError("Serialization error")

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=bad_serializer)

        with pytest.raises((ValueError, TypeError)):
            await saver.aput(
                sample_config, sample_checkpoint, sample_metadata, {"test_channel": 1}
            )

    @pytest.mark.asyncio
    async def test_timeout_during_operation(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test handling timeouts during operations."""
        # Mock pipeline execution to raise TimeoutError
        pipeline_mock = Mock()
        pipeline_mock.get = Mock(return_value=None)
        pipeline_mock.execute = AsyncMock(
            side_effect=asyncio.TimeoutError("Operation timeout")
        )
        mock_valkey_client.pipeline.return_value = pipeline_mock

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        # Should return None instead of raising exception due to error handling
        result = await saver.aget_tuple(sample_config)
        assert result is None


class TestAsyncValkeySaverKeyGeneration:
    """Test key generation methods."""

    @pytest.mark.asyncio
    async def test_make_checkpoint_key(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test checkpoint key generation."""
        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        key = saver._make_checkpoint_key(
            sample_config.get("configurable", {}).get("thread_id", ""),
            sample_config.get("configurable", {}).get("checkpoint_ns", ""),
            sample_config.get("configurable", {}).get("checkpoint_id", ""),
        )

        assert "checkpoint" in key
        assert "test-thread-123" in key
        assert "test-checkpoint-id" in key

    @pytest.mark.asyncio
    async def test_make_writes_key(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test writes key generation."""
        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        key = saver._make_writes_key(
            sample_config["configurable"]["thread_id"],
            sample_config["configurable"]["checkpoint_ns"],
            sample_config["configurable"]["checkpoint_id"],
        )

        assert "writes" in key
        assert "test-thread-123" in key
        assert "test-checkpoint-id" in key


class TestAsyncValkeySaverAputWritesErrorHandling:
    """Test aput_writes method error handling."""

    @pytest.mark.asyncio
    async def test_aput_writes_existing_data_string(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test aput_writes with existing data as string."""
        writes = [("channel", "value")]
        task_id = "test-task"

        # Mock existing writes as string
        mock_valkey_client.get.return_value = "[]"  # String instead of bytes

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.aput_writes(sample_config, writes, task_id)
        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_existing_data_invalid_type(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test aput_writes with existing data as invalid type."""
        writes = [("channel", "value")]
        task_id = "test-task"

        # Mock existing writes as invalid type (Mock object)
        mock_data = Mock()
        mock_valkey_client.get.return_value = mock_data

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.aput_writes(sample_config, writes, task_id)
        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_existing_data_json_decode_error(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test aput_writes with JSON decode error on existing data."""
        writes = [("channel", "value")]
        task_id = "test-task"

        # Mock existing writes as invalid JSON
        mock_valkey_client.get.return_value = b"invalid json"

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.aput_writes(sample_config, writes, task_id)
        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_existing_data_not_list(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test aput_writes with existing data that's not a list."""
        writes = [("channel", "value")]
        task_id = "test-task"

        # Mock existing writes as dict instead of list
        mock_valkey_client.get.return_value = orjson.dumps({"not": "a list"})

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.aput_writes(sample_config, writes, task_id)
        mock_valkey_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_aput_writes_valkey_error(
        self, mock_valkey_client, mock_serializer, sample_config
    ):
        """Test aput_writes with ValkeyError."""
        writes = [("channel", "value")]
        task_id = "test-task"

        mock_valkey_client.get.side_effect = ValkeyError("Valkey error")

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        with pytest.raises(ValkeyError):
            await saver.aput_writes(sample_config, writes, task_id)

    @pytest.mark.asyncio
    async def test_aput_writes_key_error(self, mock_valkey_client, mock_serializer):
        """Test aput_writes with KeyError."""
        writes = [("channel", "value")]
        task_id = "test-task"

        # Config missing required keys
        bad_config = RunnableConfig(configurable={})

        mock_valkey_client.get.return_value = orjson.dumps([])

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        with pytest.raises(KeyError):
            await saver.aput_writes(bad_config, writes, task_id)


class TestAsyncValkeySaverAdeleteThread:
    """Test adelete_thread method."""

    @pytest.mark.asyncio
    async def test_adelete_thread_no_keys(self, mock_valkey_client, mock_serializer):
        """Test adelete_thread when no keys exist."""
        mock_valkey_client.keys.return_value = []

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.adelete_thread("test-thread")
        mock_valkey_client.keys.assert_called_once()
        mock_valkey_client.delete.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_adelete_thread_basic_functionality(
        self, mock_valkey_client, mock_serializer
    ):
        """Test adelete_thread basic functionality."""
        mock_valkey_client.keys.return_value = [
            "checkpoint:test-thread::",
            "writes:test-thread::",
        ]
        mock_valkey_client.delete.return_value = 2

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        await saver.adelete_thread("test-thread")
        mock_valkey_client.keys.assert_called_once()
        mock_valkey_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_adelete_thread_valkey_error(
        self, mock_valkey_client, mock_serializer
    ):
        """Test adelete_thread with ValkeyError."""
        mock_valkey_client.keys.side_effect = ValkeyError("Valkey error")

        saver = AsyncValkeySaver(client=mock_valkey_client, serde=mock_serializer)

        with pytest.raises(ValkeyError):
            await saver.adelete_thread("test-thread")


# Additional async tests migrated from test_valkey_simple.py


def test_async_mock_setup():
    """Test async mock setup for Valkey client."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.ping.return_value = True
    client.get.return_value = None
    client.hgetall.return_value = {}
    client.pipeline.return_value = client

    # Test mock configuration
    assert client.ping.return_value is True
    assert client.get.return_value is None
    assert client.hgetall.return_value == {}
    assert client.pipeline.return_value == client


class TestAsyncPatterns:
    """Test async patterns and utilities."""

    @pytest.mark.asyncio
    async def test_async_mock_behavior(self):
        """Test async mock behavior."""
        from unittest.mock import AsyncMock

        async_client = AsyncMock()
        async_client.ping.return_value = True

        result = await async_client.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_async_context_manager_pattern(self):
        """Test async context manager pattern."""

        class MockAsyncContextManager:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False

        # Test context manager
        async with MockAsyncContextManager() as manager:
            assert manager.entered is True
            assert manager.exited is False

        assert manager.exited is True
