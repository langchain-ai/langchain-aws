"""
Unit tests for AgentCore Memory Checkpoint Saver.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Iterator
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.constants import TASKS

from langgraph_checkpoint_aws.checkpoint.agentcore.constants import (
    EMPTY_CHANNEL_VALUE,
    EventDecodingError,
    InvalidConfigError,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.helpers import (
    AgentCoreEventClient,
    BedrockAgentCoreClientWithRetry,
    EventProcessor,
    EventSerializer,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver

# Configure pytest to use anyio for async tests
pytestmark = pytest.mark.anyio

# Test constants for async testing
N_ASYNC_CALLS = 5
MOCK_SLEEP_DURATION = 0.1 / N_ASYNC_CALLS
OVERHEAD_DURATION = 0.1
TOTAL_EXPECTED_TIME = MOCK_SLEEP_DURATION + OVERHEAD_DURATION


@pytest.fixture
def sample_checkpoint_tuple():
    return CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "test_thread_id",
                "actor_id": "test_actor",
                "checkpoint_id": "test_checkpoint_id",
            }
        },
        checkpoint={"id": "test_checkpoint_id"},
        metadata={"source": "input", "step": 0},
    )


@pytest.fixture
def sample_checkpoint_event():
    return CheckpointEvent(
        checkpoint_id="checkpoint_123",
        checkpoint_data={
            "v": 1,
            "id": "checkpoint_123",
            "ts": "2024-01-01T00:00:00Z",
            "channel_versions": {"default": "v1", "tasks": "v2"},
            "versions_seen": {},
            "pending_sends": [],
        },
        metadata={"source": "input", "step": -1},
        parent_checkpoint_id="parent_checkpoint_id",
        thread_id="test_thread_id",
        checkpoint_ns="test_namespace",
    )


@pytest.fixture
def sample_channel_data_event():
    return ChannelDataEvent(
        channel="default",
        version="v1",
        value="test_value",
        thread_id="test_thread_id",
        checkpoint_ns="test_namespace",
    )


@pytest.fixture
def sample_writes_event():
    return WritesEvent(
        checkpoint_id="checkpoint_123",
        writes=[
            WriteItem(
                task_id="task_1",
                channel="channel_1",
                value="value_1",
                task_path="/path/1",
            ),
            WriteItem(
                task_id="task_2",
                channel=TASKS,
                value="value_2",
                task_path="/path/2",
            ),
        ],
    )


class TestAgentCoreMemorySaver:
    """Test suite for AgentCoreMemorySaver."""

    @pytest.fixture
    def mock_boto_client(self):
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.list_events = MagicMock()
        mock_client.delete_event = MagicMock()
        return mock_client

    @pytest.fixture
    def memory_id(self):
        return "test-memory-id"

    @pytest.fixture
    def saver(self, mock_boto_client, memory_id):
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_boto_client
            yield AgentCoreMemorySaver(memory_id=memory_id)

    @pytest.fixture
    def runnable_config(self):
        return RunnableConfig(
            configurable={
                "thread_id": "test_thread_id",
                "actor_id": "test_actor_id",
                "checkpoint_ns": "test_namespace",
                "checkpoint_id": "test_checkpoint_id",
            }
        )

    @pytest.fixture
    def sample_checkpoint(self):
        return Checkpoint(
            v=1,
            id="checkpoint_123",
            ts="2024-01-01T00:00:00Z",
            channel_values={
                "default": "value1",
                "tasks": ["task1", "task2"],
                "results": {"status": "completed"},
            },
            channel_versions={"default": "v1", "tasks": "v2", "results": "v1"},
            versions_seen={
                "node1": {"default": "v1", "tasks": "v2"},
                "node2": {"results": "v1"},
            },
            pending_sends=[],
        )

    @pytest.fixture
    def sample_checkpoint_metadata(self):
        return CheckpointMetadata(
            source="input",
            step=-1,
            writes={"node1": ["write1", "write2"], "node2": {"key": "value"}},
            parents={
                "namespace1": "parent_checkpoint_1",
                "namespace2": "parent_checkpoint_2",
            },
        )

    @pytest.fixture
    def mock_slow_get_tuple(self, sample_checkpoint_tuple):
        """Mock get_tuple with artificial delay for testing async concurrency."""

        def _mock_slow_get_tuple(config):  # noqa: ARG001
            time.sleep(MOCK_SLEEP_DURATION)
            return sample_checkpoint_tuple

        return _mock_slow_get_tuple

    @pytest.fixture
    def mock_slow_list(self, sample_checkpoint_tuple):
        """Mock list with artificial delay for testing async concurrency."""

        def _mock_slow_list(
            config, *, filter=None, before=None, limit=None
        ) -> Iterator[CheckpointTuple]:
            def _generator():
                time.sleep(MOCK_SLEEP_DURATION)
                yield sample_checkpoint_tuple

            return _generator()

        return _mock_slow_list

    @pytest.fixture
    def mock_slow_put(self):
        """Mock put with artificial delay for testing async concurrency."""

        def _mock_slow_put(config, checkpoint, metadata, new_versions):  # noqa: ARG001
            time.sleep(MOCK_SLEEP_DURATION)
            return config

        return _mock_slow_put

    @pytest.fixture
    def mock_slow_put_writes(self):
        """Mock put_writes with artificial delay for testing async concurrency."""

        def _mock_slow_put_writes(config, writes, task_id, task_path=""):  # noqa: ARG001
            time.sleep(MOCK_SLEEP_DURATION)
            return

        return _mock_slow_put_writes

    @pytest.fixture
    def mock_slow_delete_thread(self):
        """Mock delete_thread with artificial delay for testing async concurrency."""

        def _mock_slow_delete_thread(thread_id, actor_id=""):  # noqa: ARG001
            time.sleep(MOCK_SLEEP_DURATION)
            return

        return _mock_slow_delete_thread

    def test_init_with_default_client(self, memory_id):
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            saver = AgentCoreMemorySaver(memory_id=memory_id)

            assert saver.memory_id == memory_id
            assert isinstance(saver.serializer, EventSerializer)
            assert isinstance(saver.checkpoint_event_client, AgentCoreEventClient)
            assert isinstance(saver.processor, EventProcessor)
            mock_boto3_client.assert_called_once_with("bedrock-agentcore", config=ANY)

    def test_init_with_custom_parameters(self, memory_id):
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            saver = AgentCoreMemorySaver(
                memory_id=memory_id,
                region_name="us-west-2",
            )

            assert saver.memory_id == memory_id
            mock_boto3_client.assert_called_once_with(
                "bedrock-agentcore", region_name="us-west-2", config=ANY
            )

    def test_get_tuple_success(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint_event,
        sample_channel_data_event,
    ):
        # Remove specific checkpoint_id from config to get latest
        runnable_config["configurable"].pop("checkpoint_id", None)

        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [
                        {
                            "blob": saver.serializer.serialize_event(
                                sample_checkpoint_event
                            )
                        }
                    ],
                },
                {
                    "eventId": "event_2",
                    "payload": [
                        {
                            "blob": saver.serializer.serialize_event(
                                sample_channel_data_event
                            )
                        }
                    ],
                },
            ]
        }

        result = saver.get_tuple(runnable_config)

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint_123"
        assert result.checkpoint["id"] == "checkpoint_123"
        mock_boto_client.list_events.assert_called()

    def test_get_tuple_no_checkpoints(self, saver, mock_boto_client, runnable_config):
        # Mock empty list_events response
        mock_boto_client.list_events.return_value = {"events": []}

        result = saver.get_tuple(runnable_config)

        assert result is None
        mock_boto_client.list_events.assert_called_once()

    def test_get_tuple_with_specific_checkpoint_id(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint_event,
    ):
        # Set specific checkpoint_id
        runnable_config["configurable"]["checkpoint_id"] = "checkpoint_123"

        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [
                        {
                            "blob": saver.serializer.serialize_event(
                                sample_checkpoint_event
                            )
                        }
                    ],
                }
            ]
        }

        result = saver.get_tuple(runnable_config)

        assert isinstance(result, CheckpointTuple)
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint_123"

    def test_get_tuple_checkpoint_not_found(
        self, saver, mock_boto_client, runnable_config
    ):
        # Set specific checkpoint_id that doesn't exist
        runnable_config["configurable"]["checkpoint_id"] = "non_existent_checkpoint"

        sample_event = CheckpointEvent(
            checkpoint_id="different_checkpoint",
            checkpoint_data={},
            metadata={},
            thread_id="test_thread_id",
            checkpoint_ns="test_namespace",
        )

        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [
                        {"blob": saver.serializer.serialize_event(sample_event)}
                    ],
                }
            ]
        }

        result = saver.get_tuple(runnable_config)

        assert result is None

    def test_list_success(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint_event,
    ):
        # Remove specific checkpoint_id from config to list all
        runnable_config["configurable"].pop("checkpoint_id", None)

        checkpoint_event_1 = sample_checkpoint_event
        checkpoint_event_2 = CheckpointEvent(
            checkpoint_id="checkpoint_456",
            checkpoint_data={
                "v": 1,
                "id": "checkpoint_456",
                "ts": "2024-01-02T00:00:00Z",
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
            metadata={"source": "output", "step": 1},
            parent_checkpoint_id="checkpoint_123",
            thread_id="test_thread_id",
            checkpoint_ns="test_namespace",
        )

        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [
                        {"blob": saver.serializer.serialize_event(checkpoint_event_1)}
                    ],
                },
                {
                    "eventId": "event_2",
                    "payload": [
                        {"blob": saver.serializer.serialize_event(checkpoint_event_2)}
                    ],
                },
            ]
        }

        results = list(saver.list(runnable_config))

        assert len(results) == 2
        assert all(isinstance(r, CheckpointTuple) for r in results)
        # Should be sorted in descending order
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint_456"
        assert results[1].config["configurable"]["checkpoint_id"] == "checkpoint_123"

    def test_list_with_limit(
        self,
        saver,
        mock_boto_client,
        runnable_config,
    ):
        # Remove specific checkpoint_id from config to list all
        runnable_config["configurable"].pop("checkpoint_id", None)

        events = []
        for i in range(5):
            checkpoint_event = CheckpointEvent(
                checkpoint_id=f"checkpoint_{i}",
                checkpoint_data={
                    "v": 1,
                    "id": f"checkpoint_{i}",
                    "ts": f"2024-01-0{i + 1}T00:00:00Z",
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
                metadata={"step": i},
                thread_id="test_thread_id",
                checkpoint_ns="test_namespace",
            )
            events.append(
                {
                    "eventId": f"event_{i}",
                    "payload": [
                        {"blob": saver.serializer.serialize_event(checkpoint_event)}
                    ],
                }
            )

        mock_boto_client.list_events.return_value = {"events": events}

        results = list(saver.list(runnable_config, limit=3))

        assert len(results) == 3

    def test_list_with_before(
        self,
        saver,
        mock_boto_client,
        runnable_config,
    ):
        # Remove specific checkpoint_id from config to list all
        runnable_config["configurable"].pop("checkpoint_id", None)

        events = []
        for i in range(3):
            checkpoint_event = CheckpointEvent(
                checkpoint_id=f"checkpoint_{i}",
                checkpoint_data={
                    "v": 1,
                    "id": f"checkpoint_{i}",
                    "ts": f"2024-01-0{i + 1}T00:00:00Z",
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
                metadata={"step": i},
                thread_id="test_thread_id",
                checkpoint_ns="test_namespace",
            )
            events.append(
                {
                    "eventId": f"event_{i}",
                    "payload": [
                        {"blob": saver.serializer.serialize_event(checkpoint_event)}
                    ],
                }
            )

        mock_boto_client.list_events.return_value = {"events": events}

        before_config = RunnableConfig(configurable={"checkpoint_id": "checkpoint_2"})

        results = list(saver.list(runnable_config, before=before_config))

        assert len(results) == 2
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint_1"
        assert results[1].config["configurable"]["checkpoint_id"] == "checkpoint_0"

    def test_list_empty(self, saver, mock_boto_client, runnable_config):
        # Mock empty list_events response
        mock_boto_client.list_events.return_value = {"events": []}

        results = list(saver.list(runnable_config))

        assert len(results) == 0

    def test_put_success(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        new_versions = {"default": "v2", "tasks": "v3"}
        result = saver.put(
            runnable_config,
            sample_checkpoint,
            sample_checkpoint_metadata,
            new_versions,
        )

        assert result["configurable"]["checkpoint_id"] == "checkpoint_123"
        assert result["configurable"]["thread_id"] == "test_thread_id"
        assert result["configurable"]["checkpoint_ns"] == "test_namespace"

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["memoryId"] == saver.memory_id
        assert call_args["actorId"] == "test_actor_id"
        assert "payload" in call_args
        # Should have channel events + checkpoint event
        assert len(call_args["payload"]) == len(new_versions) + 1

    def test_put_with_empty_channel_values(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        sample_checkpoint["channel_values"] = {}

        new_versions = {"empty_channel": "v1"}
        result = saver.put(
            runnable_config,
            sample_checkpoint,
            sample_checkpoint_metadata,
            new_versions,
        )

        assert result["configurable"]["checkpoint_id"] == "checkpoint_123"
        mock_boto_client.create_event.assert_called_once()

    def test_put_writes_success(
        self,
        saver,
        mock_boto_client,
        runnable_config,
    ):
        # Set checkpoint_id in config
        runnable_config["configurable"]["checkpoint_id"] = "checkpoint_123"

        writes = [
            ("channel_1", "value_1"),
            ("channel_2", "value_2"),
            (TASKS, {"task": "data"}),
        ]

        saver.put_writes(
            runnable_config,
            writes,
            task_id="task_123",
            task_path="/test/path",
        )

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["memoryId"] == saver.memory_id
        assert call_args["actorId"] == "test_actor_id"
        assert len(call_args["payload"]) == 1

    def test_put_writes_no_checkpoint_id(
        self,
        saver,
        mock_boto_client,
    ):
        # Create config without checkpoint_id
        config = RunnableConfig(
            configurable={
                "thread_id": "test_thread_id",
                "actor_id": "test_actor_id",
            }
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            saver.put_writes(config, [("channel", "value")], "task_id")

        assert "checkpoint_id is required" in str(exc_info.value)

    def test_put_with_writes_success(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        """put_with_writes sends checkpoint + writes in a single create_event."""
        from langgraph_checkpoint_aws.checkpoint.deferred_saver import PendingWrite

        new_versions = {"default": "v2", "tasks": "v3"}
        pending_writes = [
            PendingWrite(
                config=runnable_config,
                writes=[("channel_1", "value_1"), ("channel_2", "value_2")],
                task_id="task_1",
                task_path="/path/1",
            ),
            PendingWrite(
                config=runnable_config,
                writes=[(TASKS, {"task": "data"})],
                task_id="task_2",
                task_path="/path/2",
            ),
        ]

        result = saver.put_with_writes(
            runnable_config,
            sample_checkpoint,
            sample_checkpoint_metadata,
            new_versions,
            pending_writes,
        )

        assert result["configurable"]["checkpoint_id"] == "checkpoint_123"
        assert result["configurable"]["thread_id"] == "test_thread_id"
        assert result["configurable"]["actor_id"] == "test_actor_id"
        assert result["configurable"]["checkpoint_ns"] == "test_namespace"

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["memoryId"] == saver.memory_id
        assert call_args["actorId"] == "test_actor_id"
        # 2 channels + 1 checkpoint + 2 writes events = 5 blobs
        assert len(call_args["payload"]) == 5

    def test_put_with_writes_no_pending_writes(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        """put_with_writes with empty writes list behaves like put."""
        new_versions = {"default": "v2"}

        result = saver.put_with_writes(
            runnable_config,
            sample_checkpoint,
            sample_checkpoint_metadata,
            new_versions,
            [],
        )

        assert result["configurable"]["checkpoint_id"] == "checkpoint_123"
        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        # 1 channel + 1 checkpoint = 2 blobs (no writes)
        assert len(call_args["payload"]) == 2

    def test_put_with_writes_preserves_channel_data(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint_metadata,
    ):
        """put_with_writes serializes channel values from checkpoint."""
        checkpoint = Checkpoint(
            v=1,
            id="checkpoint_456",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": ["hello", "world"], "state": 42},
            channel_versions={"messages": "v1", "state": "v1"},
            versions_seen={},
            pending_sends=[],
        )
        new_versions = {"messages": "v2", "state": "v2"}

        saver.put_with_writes(
            runnable_config,
            checkpoint,
            sample_checkpoint_metadata,
            new_versions,
            [],
        )

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        # 2 channels + 1 checkpoint = 3 blobs
        assert len(call_args["payload"]) == 3

    async def test_aput_with_writes_delegates_to_sync(
        self,
        saver,
        mock_boto_client,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        """aput_with_writes runs put_with_writes in executor."""
        from langgraph_checkpoint_aws.checkpoint.deferred_saver import PendingWrite

        new_versions = {"default": "v2"}
        pending_writes = [
            PendingWrite(
                config=runnable_config,
                writes=[("channel_1", "value_1")],
                task_id="task_1",
                task_path="",
            ),
        ]

        result = await saver.aput_with_writes(
            runnable_config,
            sample_checkpoint,
            sample_checkpoint_metadata,
            new_versions,
            pending_writes,
        )

        assert result["configurable"]["checkpoint_id"] == "checkpoint_123"
        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        # 1 channel + 1 checkpoint + 1 writes event = 3 blobs
        assert len(call_args["payload"]) == 3

    def test_delete_thread_success(
        self,
        saver,
        mock_boto_client,
    ):
        mock_boto_client.list_events.return_value = {
            "events": [
                {"eventId": "event_1"},
                {"eventId": "event_2"},
            ]
        }

        saver.delete_thread("thread_id", "actor_id")

        assert mock_boto_client.list_events.called
        assert mock_boto_client.delete_event.call_count == 2
        mock_boto_client.delete_event.assert_any_call(
            memoryId=saver.memory_id,
            sessionId="thread_id",
            eventId="event_1",
            actorId="actor_id",
        )
        mock_boto_client.delete_event.assert_any_call(
            memoryId=saver.memory_id,
            sessionId="thread_id",
            eventId="event_2",
            actorId="actor_id",
        )

    def test_delete_thread_with_pagination(
        self,
        saver,
        mock_boto_client,
    ):
        # Mock paginated list_events responses
        mock_boto_client.list_events.side_effect = [
            {
                "events": [{"eventId": "event_1"}, {"eventId": "event_2"}],
                "nextToken": "token_1",
            },
            {
                "events": [{"eventId": "event_3"}],
                "nextToken": None,
            },
        ]

        saver.delete_thread("thread_id", "actor_id")

        assert mock_boto_client.list_events.call_count == 2
        assert mock_boto_client.delete_event.call_count == 3

    def test_get_next_version(self, saver):
        # Test with None
        version = saver.get_next_version(None, None)
        assert version.startswith("00000000000000000000000000000001.")

        version = saver.get_next_version(5, None)
        assert version.startswith("00000000000000000000000000000006.")

        version = saver.get_next_version(
            "00000000000000000000000000000010.123456", None
        )
        assert version.startswith("00000000000000000000000000000011.")

    async def test_aget_tuple_calls_sync_method_with_correct_args(
        self, saver, runnable_config, mock_slow_get_tuple
    ):
        with patch.object(
            saver, "get_tuple", side_effect=mock_slow_get_tuple
        ) as mock_get:
            result = await saver.aget_tuple(runnable_config)

            # Verify sync method was called with correct arguments
            mock_get.assert_called_once_with(runnable_config)

            assert result is not None

    async def test_alist_calls_sync_method_with_correct_args(
        self, saver, runnable_config, mock_slow_list
    ):
        filter_dict = {"test": "filter"}
        before_config = {"before": "config"}
        limit_value = 10

        with patch.object(saver, "list", side_effect=mock_slow_list) as mock_list:
            # Collect all items from async iterator
            items = []
            async for item in saver.alist(
                runnable_config,
                filter=filter_dict,
                before=before_config,
                limit=limit_value,
            ):
                items.append(item)

            # Verify sync method was called with correct arguments
            mock_list.assert_called_once_with(
                runnable_config,
                filter=filter_dict,
                before=before_config,
                limit=limit_value,
            )
            assert len(items) == 1
            assert isinstance(items[0], CheckpointTuple)

    async def test_aput_calls_sync_method_with_correct_args(
        self,
        saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
        mock_slow_put,
    ):
        new_versions = {"default": "v2"}

        with patch.object(saver, "put", side_effect=mock_slow_put) as mock_put:
            result = await saver.aput(
                runnable_config,
                sample_checkpoint,
                sample_checkpoint_metadata,
                new_versions,
            )

            # Verify sync method was called with correct arguments
            mock_put.assert_called_once_with(
                runnable_config,
                sample_checkpoint,
                sample_checkpoint_metadata,
                new_versions,
            )

            assert result == runnable_config

    async def test_aput_writes_calls_sync_method_with_correct_args(
        self, saver, runnable_config, mock_slow_put_writes
    ):
        writes = [("channel", "value")]
        task_id = "test-task"
        task_path = "test-path"

        with patch.object(
            saver, "put_writes", side_effect=mock_slow_put_writes
        ) as mock_put_writes:
            result = await saver.aput_writes(
                runnable_config, writes, task_id, task_path
            )

            # Verify sync method was called with correct arguments
            mock_put_writes.assert_called_once_with(
                runnable_config, writes, task_id, task_path
            )
            assert result is None

    async def test_adelete_thread_calls_sync_method_with_correct_args(
        self, saver, runnable_config, mock_slow_delete_thread
    ):
        thread_id = runnable_config["configurable"]["thread_id"]
        actor_id = runnable_config["configurable"]["actor_id"]

        with patch.object(
            saver, "delete_thread", side_effect=mock_slow_delete_thread
        ) as mock_delete:
            result = await saver.adelete_thread(thread_id, actor_id)

            # Verify sync method was called with correct arguments
            mock_delete.assert_called_once_with(thread_id, actor_id)
            assert result is None

    async def test_concurrent_calls_aget_tuple(
        self, saver, runnable_config, mock_slow_get_tuple
    ):
        with patch.object(saver, "get_tuple", side_effect=mock_slow_get_tuple):
            await self.assert_concurrent_calls_are_faster_than_sequential(
                N_ASYNC_CALLS, saver.aget_tuple, runnable_config
            )

    async def test_concurrent_calls_adelete_thread(
        self, saver, runnable_config, mock_slow_delete_thread
    ):
        thread_id = runnable_config["configurable"]["thread_id"]
        actor_id = runnable_config["configurable"]["actor_id"]

        with patch.object(saver, "delete_thread", side_effect=mock_slow_delete_thread):
            await self.assert_concurrent_calls_are_faster_than_sequential(
                N_ASYNC_CALLS, saver.adelete_thread, thread_id, actor_id
            )

    async def test_concurrent_calls_aput_writes(
        self, saver, runnable_config, mock_slow_put_writes
    ):
        writes = [("channel", "value")]
        task_id = "test-task"
        task_path = "test-path"

        with patch.object(saver, "put_writes", side_effect=mock_slow_put_writes):
            await self.assert_concurrent_calls_are_faster_than_sequential(
                N_ASYNC_CALLS,
                saver.aput_writes,
                runnable_config,
                writes,
                task_id,
                task_path,
            )

    async def test_concurrent_calls_aput(
        self,
        saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
        mock_slow_put,
    ):
        new_versions = {"default": "v2"}

        with patch.object(saver, "put", side_effect=mock_slow_put):
            await self.assert_concurrent_calls_are_faster_than_sequential(
                N_ASYNC_CALLS,
                saver.aput,
                runnable_config,
                sample_checkpoint,
                sample_checkpoint_metadata,
                new_versions,
            )

    async def test_concurrent_calls_alist(self, saver, runnable_config, mock_slow_list):
        filter_dict = {"test": "filter"}
        before_config = {"before": "config"}
        limit_value = 10

        with patch.object(saver, "list", side_effect=mock_slow_list):

            async def consume_alist() -> list:
                """Helper coroutine to consume the async iterator."""
                items = []
                async for item in saver.alist(
                    runnable_config,
                    filter=filter_dict,
                    before=before_config,
                    limit=limit_value,
                ):
                    items.append(item)
                return items

            await self.assert_concurrent_calls_are_faster_than_sequential(
                N_ASYNC_CALLS, consume_alist
            )

    async def assert_concurrent_calls_are_faster_than_sequential(
        self, n_async_calls: int, func, *args, **kwargs
    ) -> None:
        """Helper to run n async tasks concurrently."""
        tasks = [func(*args, **kwargs) for _ in range(n_async_calls)]
        start_time = time.time()
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        assert concurrent_time < TOTAL_EXPECTED_TIME, (
            f"Concurrent execution took {concurrent_time:.2f}s, "
            f"expected < {TOTAL_EXPECTED_TIME}s"
        )


class TestCheckpointerConfig:
    """Test suite for CheckpointerConfig."""

    def test_from_runnable_config_success(self):
        config = RunnableConfig(
            configurable={
                "thread_id": "test_thread",
                "actor_id": "test_actor",
                "checkpoint_ns": "test_ns",
                "checkpoint_id": "test_checkpoint",
            }
        )

        checkpoint_config = CheckpointerConfig.from_runnable_config(config)

        assert checkpoint_config.thread_id == "test_thread"
        assert checkpoint_config.actor_id == "test_actor"
        assert checkpoint_config.checkpoint_ns == "test_ns"
        assert checkpoint_config.checkpoint_id == "test_checkpoint"
        # With checkpoint_ns set, session_id is shortened (SHA-256 hex) for AWS limit
        expected_session_id = hashlib.sha256(b"test_thread_test_ns").hexdigest()
        assert checkpoint_config.session_id == expected_session_id

    def test_from_runnable_config_no_namespace(self):
        config = RunnableConfig(
            configurable={
                "thread_id": "test_thread",
                "actor_id": "test_actor",
            }
        )

        checkpoint_config = CheckpointerConfig.from_runnable_config(config)

        assert checkpoint_config.thread_id == "test_thread"
        assert checkpoint_config.actor_id == "test_actor"
        assert checkpoint_config.checkpoint_ns == ""
        assert checkpoint_config.checkpoint_id is None
        assert checkpoint_config.session_id == "test_thread"

    def test_session_id_no_sanitization_when_checkpoint_ns_set(self):
        """full_id is thread_id + '_' + checkpoint_ns; no replace of : or |."""
        config = CheckpointerConfig(
            thread_id="t1",
            actor_id="a1",
            checkpoint_ns="ns:with|pipes",
        )
        # Hash is of raw "t1_ns:with|pipes", not sanitized
        expected = hashlib.sha256(b"t1_ns:with|pipes").hexdigest()
        assert config.session_id == expected

    def test_from_runnable_config_missing_thread_id(self):
        config = RunnableConfig(
            configurable={
                "actor_id": "test_actor",
            }
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            CheckpointerConfig.from_runnable_config(config)

        assert "thread_id" in str(exc_info.value)

    def test_from_runnable_config_missing_actor_id(self):
        config = RunnableConfig(
            configurable={
                "thread_id": "test_thread",
            }
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            CheckpointerConfig.from_runnable_config(config)

        assert "actor_id" in str(exc_info.value)


class TestEventSerializer:
    """Test suite for EventSerializer."""

    @pytest.fixture
    def serializer(self):
        return EventSerializer(JsonPlusSerializer())

    def test_serialize_deserialize_checkpoint_event(
        self, serializer, sample_checkpoint_event
    ):
        # Serialize
        serialized = serializer.serialize_event(sample_checkpoint_event)
        assert isinstance(serialized, str)

        deserialized = serializer.deserialize_event(serialized)
        assert isinstance(deserialized, CheckpointEvent)
        assert deserialized.checkpoint_id == sample_checkpoint_event.checkpoint_id
        assert deserialized.thread_id == sample_checkpoint_event.thread_id

    def test_serialize_deserialize_channel_data_event(
        self, serializer, sample_channel_data_event
    ):
        # Serialize
        serialized = serializer.serialize_event(sample_channel_data_event)
        assert isinstance(serialized, str)

        deserialized = serializer.deserialize_event(serialized)
        assert isinstance(deserialized, ChannelDataEvent)
        assert deserialized.channel == sample_channel_data_event.channel
        assert deserialized.value == sample_channel_data_event.value

    def test_serialize_deserialize_writes_event(self, serializer, sample_writes_event):
        # Serialize
        serialized = serializer.serialize_event(sample_writes_event)
        assert isinstance(serialized, str)

        deserialized = serializer.deserialize_event(serialized)
        assert isinstance(deserialized, WritesEvent)
        assert deserialized.checkpoint_id == sample_writes_event.checkpoint_id
        assert len(deserialized.writes) == len(sample_writes_event.writes)

    def test_serialize_channel_data_with_empty_value(self, serializer):
        event = ChannelDataEvent(
            channel="test",
            version="v1",
            value=EMPTY_CHANNEL_VALUE,
            thread_id="thread",
            checkpoint_ns="ns",
        )

        serialized = serializer.serialize_event(event)
        deserialized = serializer.deserialize_event(serialized)

        assert deserialized.value == EMPTY_CHANNEL_VALUE

    def test_deserialize_invalid_json(self, serializer):
        with pytest.raises(EventDecodingError) as exc_info:
            serializer.deserialize_event("invalid json {")

        assert "Failed to parse JSON" in str(exc_info.value)

    def test_deserialize_unknown_event_type(self, serializer):
        invalid_event = json.dumps({"event_type": "unknown_type", "data": "test"})

        with pytest.raises(EventDecodingError) as exc_info:
            serializer.deserialize_event(invalid_event)

        assert "Unknown event type" in str(exc_info.value)


class TestAgentCoreEventClient:
    """Test suite for AgentCoreEventClient."""

    @pytest.fixture
    def mock_boto_client(self):
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.list_events = MagicMock()
        mock_client.delete_event = MagicMock()
        return mock_client

    @pytest.fixture
    def serializer(self):
        return EventSerializer(JsonPlusSerializer())

    @pytest.fixture
    def client(self, mock_boto_client, serializer):
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_boto_client
            yield AgentCoreEventClient("test-memory-id", serializer)

    def test_store_blob_event(self, client, mock_boto_client, sample_checkpoint_event):
        client.store_blob_event(sample_checkpoint_event, "session_id", "actor_id")

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["memoryId"] == "test-memory-id"
        assert call_args["actorId"] == "actor_id"
        assert call_args["sessionId"] == "session_id"
        assert len(call_args["payload"]) == 1

    def test_store_blob_events_batch(
        self,
        client,
        mock_boto_client,
        sample_checkpoint_event,
        sample_channel_data_event,
    ):
        events = [sample_checkpoint_event, sample_channel_data_event]
        client.store_blob_events_batch(events, "session_id", "actor_id")

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert len(call_args["payload"]) == 2

    def test_store_blob_events_batch_chunks_by_item_count(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """Batches exceeding max_payload_items are split into multiple calls."""
        # Create 5 events but set max to 2 items per call
        events = [sample_channel_data_event] * 5
        client.store_blob_events_batch(
            events, "session_id", "actor_id", max_payload_items=2
        )

        # Should produce 3 API calls: [2, 2, 1]
        assert mock_boto_client.create_event.call_count == 3
        calls = mock_boto_client.create_event.call_args_list
        assert len(calls[0][1]["payload"]) == 2
        assert len(calls[1][1]["payload"]) == 2
        assert len(calls[2][1]["payload"]) == 1

    def test_store_blob_events_batch_chunks_by_byte_size(
        self,
        client,
        mock_boto_client,
        sample_checkpoint_event,
    ):
        """Batches exceeding max_payload_bytes are split into multiple calls."""
        # Serialize one event to measure its size
        serialized = client.serializer.serialize_event(sample_checkpoint_event)
        blob_size = len(serialized.encode("utf-8"))

        # Set max bytes to fit at most 2 blobs
        max_bytes = blob_size * 2 + 1
        events = [sample_checkpoint_event] * 5

        client.store_blob_events_batch(
            events, "session_id", "actor_id", max_payload_bytes=max_bytes
        )

        # Should produce 3 API calls: [2, 2, 1]
        assert mock_boto_client.create_event.call_count == 3
        calls = mock_boto_client.create_event.call_args_list
        assert len(calls[0][1]["payload"]) == 2
        assert len(calls[1][1]["payload"]) == 2
        assert len(calls[2][1]["payload"]) == 1

    def test_store_blob_events_batch_single_oversized_blob(
        self,
        client,
        mock_boto_client,
        sample_checkpoint_event,
    ):
        """A single blob that exceeds max_payload_bytes still gets its own call."""
        # Set max bytes smaller than one blob
        events = [sample_checkpoint_event]
        client.store_blob_events_batch(
            events, "session_id", "actor_id", max_payload_bytes=1
        )

        # Still sends one call (a single blob is always in its own chunk)
        mock_boto_client.create_event.assert_called_once()
        assert len(mock_boto_client.create_event.call_args[1]["payload"]) == 1

    def test_store_blob_events_batch_within_limits_single_call(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """Batch within limits sends a single API call."""
        events = [sample_channel_data_event] * 50
        client.store_blob_events_batch(events, "session_id", "actor_id")

        # Default limit is 100 items, so 50 stays within one call
        mock_boto_client.create_event.assert_called_once()
        assert len(mock_boto_client.create_event.call_args[1]["payload"]) == 50

    def test_store_blob_events_batch_empty_list(
        self,
        client,
        mock_boto_client,
    ):
        """Empty event list makes no API calls."""
        client.store_blob_events_batch([], "session_id", "actor_id")
        mock_boto_client.create_event.assert_not_called()

    def test_store_blob_events_batch_exactly_at_limit(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """Exactly 100 events fit in one call (the max)."""
        events = [sample_channel_data_event] * 100
        client.store_blob_events_batch(events, "session_id", "actor_id")

        mock_boto_client.create_event.assert_called_once()
        assert len(mock_boto_client.create_event.call_args[1]["payload"]) == 100

    def test_store_blob_events_batch_101_events_splits(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """101 events split into two calls: [100, 1]."""
        events = [sample_channel_data_event] * 101
        client.store_blob_events_batch(events, "session_id", "actor_id")

        assert mock_boto_client.create_event.call_count == 2
        calls = mock_boto_client.create_event.call_args_list
        assert len(calls[0][1]["payload"]) == 100
        assert len(calls[1][1]["payload"]) == 1

    def test_store_blob_events_batch_both_limits_hit(
        self,
        client,
        mock_boto_client,
        sample_checkpoint_event,
    ):
        """When both item count and byte size limits apply, the stricter wins."""
        serialized = client.serializer.serialize_event(sample_checkpoint_event)
        blob_size = len(serialized.encode("utf-8"))

        # max_payload_items=3 but byte size only allows 1
        events = [sample_checkpoint_event] * 4
        client.store_blob_events_batch(
            events,
            "session_id",
            "actor_id",
            max_payload_items=3,
            max_payload_bytes=blob_size + 1,
        )

        # Byte limit forces 1 per chunk = 4 calls
        assert mock_boto_client.create_event.call_count == 4
        for call_entry in mock_boto_client.create_event.call_args_list:
            assert len(call_entry[1]["payload"]) == 1

    def test_store_blob_events_batch_preserves_order(
        self,
        client,
        mock_boto_client,
        sample_checkpoint_event,
        sample_channel_data_event,
        sample_writes_event,
    ):
        """Chunked events preserve their original order across calls."""
        events = [
            sample_channel_data_event,
            sample_checkpoint_event,
            sample_writes_event,
        ]
        client.store_blob_events_batch(
            events, "session_id", "actor_id", max_payload_items=1
        )

        assert mock_boto_client.create_event.call_count == 3
        calls = mock_boto_client.create_event.call_args_list

        # Deserialize each payload blob to verify order
        blob_0 = calls[0][1]["payload"][0]["blob"]
        blob_1 = calls[1][1]["payload"][0]["blob"]
        blob_2 = calls[2][1]["payload"][0]["blob"]

        event_0 = client.serializer.deserialize_event(blob_0)
        event_1 = client.serializer.deserialize_event(blob_1)
        event_2 = client.serializer.deserialize_event(blob_2)

        assert isinstance(event_0, ChannelDataEvent)
        assert isinstance(event_1, CheckpointEvent)
        assert isinstance(event_2, WritesEvent)

    def test_store_blob_events_batch_uses_same_timestamp(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """All chunked calls use the same eventTimestamp."""
        events = [sample_channel_data_event] * 5
        client.store_blob_events_batch(
            events, "session_id", "actor_id", max_payload_items=2
        )

        assert mock_boto_client.create_event.call_count == 3
        timestamps = [
            call_entry[1]["eventTimestamp"]
            for call_entry in mock_boto_client.create_event.call_args_list
        ]
        # All timestamps should be identical
        assert timestamps[0] == timestamps[1] == timestamps[2]

    def test_store_blob_events_batch_uses_correct_session_and_actor(
        self,
        client,
        mock_boto_client,
        sample_channel_data_event,
    ):
        """All chunked calls use the same session_id, actor_id, and memory_id."""
        events = [sample_channel_data_event] * 3
        client.store_blob_events_batch(
            events, "my-session", "my-actor", max_payload_items=1
        )

        assert mock_boto_client.create_event.call_count == 3
        for call_entry in mock_boto_client.create_event.call_args_list:
            kwargs = call_entry[1]
            assert kwargs["memoryId"] == "test-memory-id"
            assert kwargs["sessionId"] == "my-session"
            assert kwargs["actorId"] == "my-actor"


class TestChunkPayload:
    """Direct unit tests for AgentCoreEventClient._chunk_payload static method."""

    def test_empty_input(self):
        result = AgentCoreEventClient._chunk_payload([], 100, 10_000_000)
        assert result == []

    def test_single_blob_within_limits(self):
        result = AgentCoreEventClient._chunk_payload(["hello"], 100, 10_000_000)
        assert result == [["hello"]]

    def test_splits_by_item_count(self):
        blobs = ["a", "b", "c", "d", "e"]
        result = AgentCoreEventClient._chunk_payload(blobs, 2, 10_000_000)
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_splits_by_byte_size(self):
        # Each blob is 10 bytes UTF-8
        blobs = ["x" * 10] * 5
        # Allow at most 25 bytes per chunk (fits 2 blobs of 10 bytes each)
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 25)
        assert result == [["x" * 10, "x" * 10], ["x" * 10, "x" * 10], ["x" * 10]]

    def test_single_blob_exceeds_byte_limit(self):
        """A blob larger than max_bytes still gets its own chunk."""
        big_blob = "x" * 1000
        result = AgentCoreEventClient._chunk_payload([big_blob], 100, 10)
        assert result == [[big_blob]]

    def test_multiple_oversized_blobs(self):
        """Each oversized blob gets its own chunk."""
        blobs = ["x" * 100, "y" * 100, "z" * 100]
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 50)
        assert result == [["x" * 100], ["y" * 100], ["z" * 100]]

    def test_mixed_sizes(self):
        """Mix of small and large blobs chunk correctly."""
        small = "s"  # 1 byte
        big = "B" * 50  # 50 bytes
        blobs = [small, small, big, small, big]
        # Chunk 1: s(1) + s(1) + B*50(50) + s(1) = 53 bytes, fits in 60
        # Adding next big(50) would be 103 > 60, so new chunk
        # Chunk 2: B*50(50) = 50 bytes
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 60)
        assert result == [[small, small, big, small], [big]]

    def test_item_limit_one(self):
        """Max 1 item per chunk gives one chunk per blob."""
        blobs = ["a", "b", "c"]
        result = AgentCoreEventClient._chunk_payload(blobs, 1, 10_000_000)
        assert result == [["a"], ["b"], ["c"]]

    def test_exact_byte_boundary(self):
        """Blob that exactly fills the byte budget starts a new chunk."""
        # Each blob is exactly 5 bytes
        blobs = ["12345", "67890", "abcde"]
        # Max 10 bytes: first two fit exactly, third starts new chunk
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 10)
        assert result == [["12345", "67890"], ["abcde"]]

    def test_unicode_byte_counting(self):
        """Byte size accounts for multi-byte UTF-8 characters."""
        # Each emoji is 4 bytes in UTF-8
        emoji_blob = "\U0001f600"  # single emoji, 4 bytes
        blobs = [emoji_blob] * 3
        # Max 8 bytes: fits 2 emoji blobs (4 + 4 = 8), third starts new chunk
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 8)
        assert result == [[emoji_blob, emoji_blob], [emoji_blob]]

    def test_preserves_blob_order(self):
        blobs = ["first", "second", "third", "fourth", "fifth"]
        result = AgentCoreEventClient._chunk_payload(blobs, 2, 10_000_000)
        flat = [blob for chunk in result for blob in chunk]
        assert flat == blobs

    def test_default_limits_no_split_for_typical_payload(self):
        """Typical small payloads stay in one chunk with default limits."""
        blobs = ["x" * 200] * 50  # 50 blobs of 200 bytes = 10 KB total
        result = AgentCoreEventClient._chunk_payload(blobs, 100, 10_000_000)
        assert len(result) == 1
        assert len(result[0]) == 50


class TestAgentCoreEventClientGetAndDelete:
    """Continuation of AgentCoreEventClient tests for get/delete operations."""

    @pytest.fixture
    def mock_boto_client(self):
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.list_events = MagicMock()
        mock_client.delete_event = MagicMock()
        return mock_client

    @pytest.fixture
    def serializer(self):
        return EventSerializer(JsonPlusSerializer())

    @pytest.fixture
    def client(self, mock_boto_client, serializer):
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_boto_client
            yield AgentCoreEventClient("test-memory-id", serializer)

    def test_get_events(
        self, client, mock_boto_client, serializer, sample_checkpoint_event
    ):
        # Mock list_events response
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [
                        {"blob": serializer.serialize_event(sample_checkpoint_event)}
                    ],
                }
            ]
        }

        events = client.get_events("session_id", "actor_id")

        assert len(events) == 1
        assert isinstance(events[0], CheckpointEvent)
        mock_boto_client.list_events.assert_called_once()

    def test_get_events_with_pagination(
        self, client, mock_boto_client, serializer, sample_checkpoint_event
    ):
        # Mock paginated responses
        mock_boto_client.list_events.side_effect = [
            {
                "events": [
                    {
                        "eventId": "event_1",
                        "payload": [
                            {
                                "blob": serializer.serialize_event(
                                    sample_checkpoint_event
                                )
                            }
                        ],
                    }
                ],
                "nextToken": "token_1",
            },
            {
                "events": [
                    {
                        "eventId": "event_2",
                        "payload": [
                            {
                                "blob": serializer.serialize_event(
                                    sample_checkpoint_event
                                )
                            }
                        ],
                    }
                ],
                "nextToken": None,
            },
        ]

        events = client.get_events("session_id", "actor_id")

        assert len(events) == 2
        assert mock_boto_client.list_events.call_count == 2

    def test_get_events_with_decoding_error(self, client, mock_boto_client, serializer):
        # Mock list_events response with invalid blob
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event_1",
                    "payload": [{"blob": "invalid json {"}],
                }
            ]
        }

        events = client.get_events("session_id", "actor_id")

        assert len(events) == 0

    def test_delete_events(self, client, mock_boto_client):
        # Mock list_events response
        mock_boto_client.list_events.return_value = {
            "events": [
                {"eventId": "event_1"},
                {"eventId": "event_2"},
            ]
        }

        client.delete_events("session_id", "actor_id")

        assert mock_boto_client.list_events.called
        assert mock_boto_client.delete_event.call_count == 2


class TestEventProcessor:
    """Test suite for EventProcessor."""

    @pytest.fixture
    def processor(self):
        return EventProcessor()

    def test_process_events(
        self,
        processor,
        sample_checkpoint_event,
        sample_channel_data_event,
        sample_writes_event,
    ):
        events = [
            sample_checkpoint_event,
            sample_channel_data_event,
            sample_writes_event,
        ]

        checkpoints, writes_by_checkpoint, channel_data = processor.process_events(
            events
        )

        assert len(checkpoints) == 1
        assert "checkpoint_123" in checkpoints
        assert len(writes_by_checkpoint["checkpoint_123"]) == 2
        assert ("default", "v1") in channel_data

    def test_process_events_empty_channel_value(
        self, processor, sample_channel_data_event
    ):
        sample_channel_data_event.value = EMPTY_CHANNEL_VALUE
        events = [sample_channel_data_event]

        checkpoints, writes_by_checkpoint, channel_data = processor.process_events(
            events
        )

        assert len(channel_data) == 0

    def test_build_checkpoint_tuple(
        self,
        processor,
        sample_checkpoint_event,
    ):
        writes = [
            WriteItem(
                task_id="task_1",
                channel="channel_1",
                value="value_1",
                task_path="/path/1",
            )
        ]
        channel_data = {("default", "v1"): "test_value"}
        config = CheckpointerConfig(
            thread_id="test_thread",
            actor_id="test_actor",
            checkpoint_ns="test_ns",
        )

        tuple_result = processor.build_checkpoint_tuple(
            sample_checkpoint_event, writes, channel_data, config
        )

        assert isinstance(tuple_result, CheckpointTuple)
        assert tuple_result.checkpoint["id"] == "checkpoint_123"
        assert len(tuple_result.pending_writes) == 1
        assert tuple_result.checkpoint["channel_values"]["default"] == "test_value"

    def test_build_checkpoint_tuple_with_parent(
        self,
        processor,
        sample_checkpoint_event,
    ):
        config = CheckpointerConfig(
            thread_id="test_thread",
            actor_id="test_actor",
            checkpoint_ns="test_ns",
        )

        tuple_result = processor.build_checkpoint_tuple(
            sample_checkpoint_event, [], {}, config
        )

        assert tuple_result.parent_config is not None
        assert (
            tuple_result.parent_config["configurable"]["checkpoint_id"]
            == "parent_checkpoint_id"
        )

    def test_build_checkpoint_tuple_no_parent(
        self,
        processor,
        sample_checkpoint_event,
    ):
        sample_checkpoint_event.parent_checkpoint_id = None
        config = CheckpointerConfig(
            thread_id="test_thread",
            actor_id="test_actor",
            checkpoint_ns="test_ns",
        )

        tuple_result = processor.build_checkpoint_tuple(
            sample_checkpoint_event, [], {}, config
        )

        assert tuple_result.parent_config is None


class TestBedrockAgentCoreClientWithRetry:
    """Test suite for BedrockAgentCoreClientWithRetry retry logic."""

    @pytest.fixture
    def mock_boto_client(self):
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.list_events = MagicMock()
        return mock_client

    @pytest.fixture
    def enhanced_client(self, mock_boto_client):
        return BedrockAgentCoreClientWithRetry(mock_boto_client, max_retries=3)

    def test_successful_call_no_retry(self, enhanced_client, mock_boto_client):
        """Test that successful calls don't trigger retries."""
        mock_boto_client.create_event.return_value = {"eventId": "test-event-id"}

        result = enhanced_client.create_event(
            memoryId="test-memory-id",
            actorId="test-actor",
            sessionId="test-session",
            eventTimestamp="2024-01-01T00:00:00Z",
            payload=[{"blob": "test"}],
        )

        assert result == {"eventId": "test-event-id"}
        assert mock_boto_client.create_event.call_count == 1

    def test_retry_on_retryable_conflict_exception_success(
        self, enhanced_client, mock_boto_client
    ):
        """Test that RetryableConflictException triggers retry and succeeds."""
        # First call fails with RetryableConflictException, second succeeds
        mock_boto_client.create_event.side_effect = [
            ClientError(
                {
                    "Error": {
                        "Code": "RetryableConflictException",
                        "Message": "Conflict",
                    }
                },
                "CreateEvent",
            ),
            {"eventId": "test-event-id"},
        ]

        result = enhanced_client.create_event(
            memoryId="test-memory-id",
            actorId="test-actor",
            sessionId="test-session",
            eventTimestamp="2024-01-01T00:00:00Z",
            payload=[{"blob": "test"}],
        )

        assert result == {"eventId": "test-event-id"}
        assert mock_boto_client.create_event.call_count == 2

    def test_retry_on_retryable_conflict_exception_max_retries(
        self, enhanced_client, mock_boto_client
    ):
        """Test that max retries are respected for RetryableConflictException."""
        # All calls fail with RetryableConflictException
        mock_boto_client.create_event.side_effect = ClientError(
            {"Error": {"Code": "RetryableConflictException", "Message": "Conflict"}},
            "CreateEvent",
        )

        with pytest.raises(ClientError) as exc_info:
            enhanced_client.create_event(
                memoryId="test-memory-id",
                actorId="test-actor",
                sessionId="test-session",
                eventTimestamp="2024-01-01T00:00:00Z",
                payload=[{"blob": "test"}],
            )

        assert exc_info.value.response["Error"]["Code"] == "RetryableConflictException"
        # Should attempt once + max_retries (3) = 4 total
        assert mock_boto_client.create_event.call_count == 4

    def test_non_retryable_error_not_retried(self, enhanced_client, mock_boto_client):
        """Test that non-retryable errors are not retried."""
        mock_boto_client.create_event.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Invalid input"}},
            "CreateEvent",
        )

        with pytest.raises(ClientError) as exc_info:
            enhanced_client.create_event(
                memoryId="test-memory-id",
                actorId="test-actor",
                sessionId="test-session",
                eventTimestamp="2024-01-01T00:00:00Z",
                payload=[{"blob": "test"}],
            )

        assert exc_info.value.response["Error"]["Code"] == "ValidationException"
        # Should only attempt once (no retries)
        assert mock_boto_client.create_event.call_count == 1

    @patch("langgraph_checkpoint_aws.checkpoint.agentcore.helpers.time.sleep")
    def test_retry_exponential_backoff(
        self, mock_sleep, enhanced_client, mock_boto_client
    ):
        """Test that exponential backoff is applied between retries."""
        # First two calls fail, third succeeds
        mock_boto_client.create_event.side_effect = [
            ClientError(
                {
                    "Error": {
                        "Code": "RetryableConflictException",
                        "Message": "Conflict",
                    }
                },
                "CreateEvent",
            ),
            ClientError(
                {
                    "Error": {
                        "Code": "RetryableConflictException",
                        "Message": "Conflict",
                    }
                },
                "CreateEvent",
            ),
            {"eventId": "test-event-id"},
        ]

        result = enhanced_client.create_event(
            memoryId="test-memory-id",
            actorId="test-actor",
            sessionId="test-session",
            eventTimestamp="2024-01-01T00:00:00Z",
            payload=[{"blob": "test"}],
        )

        assert result == {"eventId": "test-event-id"}
        assert mock_boto_client.create_event.call_count == 3
        # Verify exponential backoff was applied
        assert mock_sleep.call_args_list == [
            call(0.1),  # 0.1 * 2^0
            call(0.2),  # 0.1 * 2^1
        ]

    def test_passthrough_non_create_event_methods(
        self, enhanced_client, mock_boto_client
    ):
        """Test that non-create_event methods are passed through without retry logic."""
        mock_boto_client.list_events.return_value = {"events": []}

        result = enhanced_client.list_events(
            memoryId="test-memory-id",
            actorId="test-actor",
            sessionId="test-session",
        )

        assert result == {"events": []}
        mock_boto_client.list_events.assert_called_once()

    def test_custom_max_retries(self, mock_boto_client):
        """Test that custom max_retries value is respected."""
        enhanced_client = BedrockAgentCoreClientWithRetry(
            mock_boto_client, max_retries=1
        )

        mock_boto_client.create_event.side_effect = ClientError(
            {
                "Error": {
                    "Code": "RetryableConflictException",
                    "Message": "Conflict",
                }
            },
            "CreateEvent",
        )

        with pytest.raises(ClientError):
            enhanced_client.create_event(
                memoryId="test-memory-id",
                actorId="test-actor",
                sessionId="test-session",
                eventTimestamp="2024-01-01T00:00:00Z",
                payload=[{"blob": "test"}],
            )

        # Should attempt once + max_retries (1) = 2 total
        assert mock_boto_client.create_event.call_count == 2
