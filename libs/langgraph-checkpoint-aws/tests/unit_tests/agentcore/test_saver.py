"""
Unit tests for AgentCore Memory Checkpoint Saver.
"""

import json
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.constants import TASKS

from langgraph_checkpoint_aws.agentcore.constants import (
    EMPTY_CHANNEL_VALUE,
    EventDecodingError,
    InvalidConfigError,
)
from langgraph_checkpoint_aws.agentcore.helpers import (
    AgentCoreEventClient,
    EventProcessor,
    EventSerializer,
)
from langgraph_checkpoint_aws.agentcore.models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)
from langgraph_checkpoint_aws.agentcore.saver import AgentCoreMemorySaver


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
        assert checkpoint_config.session_id == "test_thread_test_ns"

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
