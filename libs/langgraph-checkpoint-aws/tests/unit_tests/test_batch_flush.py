"""Unit tests for batch flush functionality with AgentCoreMemorySaver."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import MemorySaver

from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver
from langgraph_checkpoint_aws.checkpoint.deferred_saver import (
    AsyncBatchFlushable,
    DeferredCheckpointSaver,
    PendingWrite,
    SyncBatchFlushable,
)


def _make_config(
    thread_id: str = "thread-1",
    actor_id: str = "actor-1",
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
    **extra: Any,
) -> RunnableConfig:
    cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id,
            "checkpoint_ns": checkpoint_ns,
            **extra,
        }
    }
    if checkpoint_id is not None:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


def _make_checkpoint(checkpoint_id: str = "ckpt-1") -> Checkpoint:
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts="2025-01-01T00:00:00+00:00",
        channel_values={"messages": ["hello"]},
        channel_versions={"messages": "v1"},
        versions_seen={"node1": {"messages": "v1"}},
        updated_channels=None,
    )


def _make_metadata(step: int = 0) -> CheckpointMetadata:
    return CheckpointMetadata(
        source="input",
        step=step,
        parents={},
    )


@pytest.fixture
def mock_boto_client():
    mock_client = Mock()
    mock_client.create_event = MagicMock()
    mock_client.list_events = MagicMock(return_value={"events": []})
    mock_client.delete_event = MagicMock()
    return mock_client


@pytest.fixture
def agentcore_saver(mock_boto_client):
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = mock_boto_client
        yield AgentCoreMemorySaver(memory_id="test-memory-id")


class TestProtocolDetection:
    """AgentCoreMemorySaver satisfies batch flush protocols."""

    def test_satisfies_sync_protocol(self, agentcore_saver) -> None:
        assert isinstance(agentcore_saver, SyncBatchFlushable)

    def test_satisfies_async_protocol(self, agentcore_saver) -> None:
        assert isinstance(agentcore_saver, AsyncBatchFlushable)

    def test_memory_saver_does_not_satisfy_sync_protocol(self) -> None:
        saver = MemorySaver()
        assert not isinstance(saver, SyncBatchFlushable)

    def test_memory_saver_does_not_satisfy_async_protocol(self) -> None:
        saver = MemorySaver()
        assert not isinstance(saver, AsyncBatchFlushable)


class TestBatchFlushFastPath:
    """flush() fast path using AgentCoreMemorySaver.put_with_writes."""

    def test_flush_calls_put_with_writes_single_api_call(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """Batch flush sends checkpoint + writes in one create_event call."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")
        deferred.put_writes(write_config, [("ch2", "v2")], "task-2")

        result = deferred.flush()

        assert deferred.is_empty
        assert result is not None
        assert result["configurable"]["checkpoint_id"] == "ckpt-1"
        # Only ONE create_event call for the entire flush
        mock_boto_client.create_event.assert_called_once()

        call_args = mock_boto_client.create_event.call_args[1]
        payload = call_args["payload"]
        # Payload should contain: channel events + checkpoint event + writes events
        # 1 channel ("messages": "v1") + 1 checkpoint + 2 writes = 4 blobs
        assert len(payload) == 4

    def test_flush_fast_path_no_writes(self, agentcore_saver, mock_boto_client) -> None:
        """Batch flush with checkpoint only, no pending writes."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)

        result = deferred.flush()

        assert deferred.is_empty
        assert result is not None
        mock_boto_client.create_event.assert_called_once()
        # 1 channel + 1 checkpoint = 2 blobs
        payload = mock_boto_client.create_event.call_args[1]["payload"]
        assert len(payload) == 2

    def test_flush_fast_path_multiple_channels_and_writes(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """Batch flush with multiple channels and multiple writes."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()
        new_versions: ChannelVersions = {
            "messages": "v1",
            "state": "v2",
            "results": "v3",
        }

        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")
        deferred.put_writes(write_config, [("ch2", "v2"), ("ch3", "v3")], "task-2")
        deferred.put_writes(write_config, [("ch4", "v4")], "task-3")

        deferred.flush()

        mock_boto_client.create_event.assert_called_once()
        # 3 channels + 1 checkpoint + 3 writes events = 7 blobs
        payload = mock_boto_client.create_event.call_args[1]["payload"]
        assert len(payload) == 7

    def test_flush_fast_path_no_checkpoint_restores_writes(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """If only writes are buffered (no checkpoint), they stay in the buffer."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)

        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        result = deferred.flush()

        assert result is None
        assert deferred.has_buffered_writes
        mock_boto_client.create_event.assert_not_called()

    def test_flush_fast_path_failure_restores_all(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """On failure, both checkpoint and writes are restored to the buffer."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")
        deferred.put_writes(write_config, [("ch2", "v2")], "task-2")

        mock_boto_client.create_event.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            deferred.flush()

        assert deferred.has_buffered_checkpoint
        assert deferred.has_buffered_writes

    def test_flush_fast_path_failure_does_not_clobber_new_checkpoint(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """Restore on failure must not overwrite a newer put() that raced in."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        deferred.put(config, _make_checkpoint("ckpt-old"), _make_metadata(), {})

        def failing_create_event(**kwargs):
            # Simulate a new put() arriving during I/O
            deferred.put(config, _make_checkpoint("ckpt-new"), _make_metadata(1), {})
            msg = "network error"
            raise RuntimeError(msg)

        mock_boto_client.create_event.side_effect = failing_create_event

        with pytest.raises(RuntimeError, match="network error"):
            deferred.flush()

        result = deferred.get_tuple(_make_config())
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-new"

    def test_flush_fast_path_preserves_session_and_actor(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """Batch flush sends correct session/actor derived from config."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config(thread_id="my-thread", actor_id="my-actor")

        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})
        deferred.flush()

        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["sessionId"] == "my-thread"
        assert call_args["actorId"] == "my-actor"


class TestBatchFlushFallback:
    """When batch_writes=False, fallback path is used (1 + N calls)."""

    def test_batch_writes_false_uses_sequential(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """With batch_writes=False, flush makes separate API calls."""
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=False)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")
        deferred.put_writes(write_config, [("ch2", "v2")], "task-2")

        deferred.flush()

        assert deferred.is_empty
        # 1 call for put() + 2 calls for put_writes() = 3 total
        assert mock_boto_client.create_event.call_count == 3

    def test_non_batch_saver_uses_sequential(self) -> None:
        """MemorySaver doesn't implement BatchFlushable, so fallback is used."""
        saver = MemorySaver()
        deferred = DeferredCheckpointSaver(saver, batch_writes=True)
        config = _make_config()

        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        result = deferred.flush()

        assert deferred.is_empty
        assert result is not None


class TestBatchFlushAsync:
    """aflush() fast path with AgentCoreMemorySaver."""

    @pytest.mark.asyncio
    async def test_aflush_calls_aput_with_writes(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        result = await deferred.aflush()

        assert deferred.is_empty
        assert result is not None
        # Single API call for the batch
        mock_boto_client.create_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_aflush_fast_path_failure_restores_all(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        mock_boto_client.create_event.side_effect = RuntimeError("async batch fail")

        with pytest.raises(RuntimeError, match="async batch fail"):
            await deferred.aflush()

        assert deferred.has_buffered_checkpoint
        assert deferred.has_buffered_writes

    @pytest.mark.asyncio
    async def test_aflush_fast_path_failure_does_not_clobber(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()

        deferred.put(config, _make_checkpoint("ckpt-old"), _make_metadata(), {})

        def failing_create_event(**kwargs):
            deferred.put(config, _make_checkpoint("ckpt-new"), _make_metadata(1), {})
            msg = "async fail"
            raise RuntimeError(msg)

        mock_boto_client.create_event.side_effect = failing_create_event

        with pytest.raises(RuntimeError, match="async fail"):
            await deferred.aflush()

        result = deferred.get_tuple(_make_config())
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-new"

    @pytest.mark.asyncio
    async def test_aflush_non_batch_saver_uses_sequential(self) -> None:
        saver = MemorySaver()
        deferred = DeferredCheckpointSaver(saver, batch_writes=True)
        config = _make_config()

        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = await deferred.aflush()

        assert deferred.is_empty
        assert result is not None


class TestBatchFlushContextManagers:
    """Context managers use the fast path when batch_writes=True."""

    def test_flush_on_exit_uses_batch(self, agentcore_saver, mock_boto_client) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()
        new_versions: ChannelVersions = {"messages": "v1"}

        with deferred.flush_on_exit():
            deferred.put(
                config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions
            )
            write_config = _make_config(checkpoint_id="ckpt-1")
            deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        assert deferred.is_empty
        mock_boto_client.create_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_aflush_on_exit_uses_batch(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()
        new_versions: ChannelVersions = {"messages": "v1"}

        async with deferred.aflush_on_exit():
            deferred.put(
                config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions
            )
            write_config = _make_config(checkpoint_id="ckpt-1")
            deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        assert deferred.is_empty
        mock_boto_client.create_event.assert_called_once()


class TestBatchFlushThreadSafety:
    """Concurrent operations with batch_writes=True."""

    def test_concurrent_flush_no_double_persist(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        deferred = DeferredCheckpointSaver(agentcore_saver, batch_writes=True)
        config = _make_config()
        new_versions: ChannelVersions = {"messages": "v1"}
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), new_versions)
        write_config = _make_config(checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        errors: list[Exception] = []

        def flusher() -> None:
            try:
                deferred.flush()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=flusher),
            threading.Thread(target=flusher),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert deferred.is_empty
        # Only one thread should have called create_event
        mock_boto_client.create_event.assert_called_once()


class TestPutWithWritesDirect:
    """Test AgentCoreMemorySaver.put_with_writes directly."""

    def test_put_with_writes_creates_correct_events(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """put_with_writes produces channel + checkpoint + writes events."""
        config = _make_config(checkpoint_id="parent-ckpt")
        checkpoint = _make_checkpoint("ckpt-1")
        metadata = _make_metadata()
        new_versions: ChannelVersions = {"messages": "v1", "state": "v2"}

        pending_writes = [
            PendingWrite(
                config=_make_config(checkpoint_id="ckpt-1"),
                writes=[("ch1", "val1")],
                task_id="task-1",
                task_path="/path/1",
            ),
            PendingWrite(
                config=_make_config(checkpoint_id="ckpt-1"),
                writes=[("ch2", "val2"), ("ch3", "val3")],
                task_id="task-2",
                task_path="/path/2",
            ),
        ]

        result = agentcore_saver.put_with_writes(
            config, checkpoint, metadata, new_versions, pending_writes
        )

        assert result["configurable"]["checkpoint_id"] == "ckpt-1"
        assert result["configurable"]["thread_id"] == "thread-1"
        assert result["configurable"]["actor_id"] == "actor-1"

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        payload = call_args["payload"]
        # 2 channels + 1 checkpoint + 2 writes events = 5
        assert len(payload) == 5
        assert call_args["memoryId"] == "test-memory-id"
        assert call_args["sessionId"] == "thread-1"
        assert call_args["actorId"] == "actor-1"

    def test_put_with_writes_empty_writes_list(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """put_with_writes with no pending writes works like put."""
        config = _make_config()
        checkpoint = _make_checkpoint("ckpt-1")
        new_versions: ChannelVersions = {"messages": "v1"}

        result = agentcore_saver.put_with_writes(
            config, checkpoint, _make_metadata(), new_versions, []
        )

        assert result["configurable"]["checkpoint_id"] == "ckpt-1"
        mock_boto_client.create_event.assert_called_once()
        # 1 channel + 1 checkpoint = 2
        payload = mock_boto_client.create_event.call_args[1]["payload"]
        assert len(payload) == 2

    @pytest.mark.asyncio
    async def test_aput_with_writes_delegates_to_sync(
        self, agentcore_saver, mock_boto_client
    ) -> None:
        """aput_with_writes runs put_with_writes in executor."""
        config = _make_config()
        checkpoint = _make_checkpoint("ckpt-1")
        new_versions: ChannelVersions = {"messages": "v1"}

        pending_writes = [
            PendingWrite(
                config=_make_config(checkpoint_id="ckpt-1"),
                writes=[("ch1", "val1")],
                task_id="task-1",
                task_path="",
            ),
        ]

        result = await agentcore_saver.aput_with_writes(
            config, checkpoint, _make_metadata(), new_versions, pending_writes
        )

        assert result["configurable"]["checkpoint_id"] == "ckpt-1"
        mock_boto_client.create_event.assert_called_once()


class TestPendingWriteType:
    """PendingWrite public type is correctly constructed."""

    def test_pending_write_fields(self) -> None:
        config = _make_config(checkpoint_id="ckpt-1")
        pw = PendingWrite(
            config=config,
            writes=[("ch1", "v1"), ("ch2", "v2")],
            task_id="task-1",
            task_path="path/to/task",
        )
        assert pw.config is config
        assert pw.writes == [("ch1", "v1"), ("ch2", "v2")]
        assert pw.task_id == "task-1"
        assert pw.task_path == "path/to/task"

    def test_pending_write_is_namedtuple(self) -> None:
        config = _make_config()
        pw = PendingWrite(config=config, writes=[], task_id="t", task_path="")
        assert hasattr(pw, "_fields")
        assert set(pw._fields) == {"config", "writes", "task_id", "task_path"}
