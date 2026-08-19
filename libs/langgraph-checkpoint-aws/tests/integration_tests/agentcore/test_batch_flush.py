"""Integration tests for batch flush with AgentCoreMemorySaver.

Tests the DeferredCheckpointSaver batch_writes=True path end-to-end
against a real AgentCore Memory resource.

Requires:
    - AGENTCORE_MEMORY_ID environment variable set to a valid memory resource
    - AWS credentials with bedrock-agentcore:CreateEvent, ListEvents, DeleteEvent
"""

from __future__ import annotations

import datetime
import os
import random
import string

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    uuid6,
)

from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver
from langgraph_checkpoint_aws.checkpoint.deferred_saver import DeferredCheckpointSaver

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")


def _generate_id(prefix: str = "test") -> str:
    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=6))


def _make_checkpoint(
    channel_values: dict | None = None,
    channel_versions: dict | None = None,
) -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        channel_values=channel_values or {"messages": ["test message"]},
        channel_versions=channel_versions or {"messages": "v1"},
        versions_seen={"node1": {"messages": "v1"}},
        updated_channels=None,
    )


@pytest.fixture(scope="module")
def memory_id():
    mid = os.environ.get("AGENTCORE_MEMORY_ID")
    if not mid:
        pytest.skip("AGENTCORE_MEMORY_ID environment variable not set")
    return mid


@pytest.fixture(scope="module")
def memory_saver(memory_id):
    return AgentCoreMemorySaver(memory_id=memory_id, region_name=AWS_REGION)


class TestBatchFlushIntegration:
    """End-to-end tests for DeferredCheckpointSaver with batch_writes=True."""

    def test_batch_flush_persists_checkpoint_and_writes(self, memory_saver) -> None:
        """Batch flush persists checkpoint + writes in a single operation."""
        thread_id = _generate_id("batchflush")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {"messages": "v1"},
            )
            deferred.put_writes(write_config, [("messages", "hello")], "task-1")
            deferred.put_writes(write_config, [("tools", "result")], "task-2")

            result = deferred.flush()

            assert deferred.is_empty
            assert result is not None
            assert result["configurable"]["checkpoint_id"] == checkpoint["id"]

            # Verify checkpoint persisted
            persisted = memory_saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]

            # Verify writes persisted
            assert persisted.pending_writes is not None
            task_ids = {pw[0] for pw in persisted.pending_writes}
            assert "task-1" in task_ids
            assert "task-2" in task_ids
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_no_writes(self, memory_saver) -> None:
        """Batch flush with only a checkpoint (no writes) works correctly."""
        thread_id = _generate_id("batchnowrites")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {"messages": "v1"},
            )

            result = deferred.flush()

            assert deferred.is_empty
            assert result is not None

            saved_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            persisted = memory_saver.get_tuple(saved_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_multiple_channels(self, memory_saver) -> None:
        """Batch flush correctly persists multiple channel versions."""
        thread_id = _generate_id("batchmulti")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        new_versions: ChannelVersions = {
            "messages": "v2",
            "state": "v2",
            "results": "v2",
        }
        checkpoint = _make_checkpoint(
            channel_values={
                "messages": ["msg1", "msg2"],
                "state": {"key": "value"},
                "results": [1, 2, 3],
            },
            channel_versions=new_versions,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "loop", "step": 3},
                new_versions,
            )

            deferred.flush()

            saved_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            persisted = memory_saver.get_tuple(saved_config)
            assert persisted is not None
            assert persisted.checkpoint["channel_values"]["messages"] == [
                "msg1",
                "msg2",
            ]
            assert persisted.checkpoint["channel_values"]["state"] == {"key": "value"}
            assert persisted.checkpoint["channel_values"]["results"] == [1, 2, 3]
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_with_context_manager(self, memory_saver) -> None:
        """flush_on_exit with batch_writes=True works end-to-end."""
        thread_id = _generate_id("batchctx")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

        try:
            with deferred.flush_on_exit():
                deferred.put(
                    config,
                    checkpoint,
                    {"source": "input", "step": 1},
                    {"messages": "v1"},
                )
                deferred.put_writes(write_config, [("messages", "batched")], "task-ctx")

            assert deferred.is_empty

            persisted = memory_saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]
            assert persisted.pending_writes is not None
            assert any(pw[0] == "task-ctx" for pw in persisted.pending_writes)
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    @pytest.mark.asyncio
    async def test_async_batch_flush(self, memory_saver) -> None:
        """aflush with batch_writes=True works end-to-end."""
        thread_id = _generate_id("asyncbatch")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

        try:
            await deferred.aput(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {"messages": "v1"},
            )
            await deferred.aput_writes(
                write_config, [("messages", "async-batched")], "task-async"
            )

            result = await deferred.aflush()

            assert deferred.is_empty
            assert result is not None

            persisted = memory_saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]
            assert persisted.pending_writes is not None
            assert any(pw[0] == "task-async" for pw in persisted.pending_writes)
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_many_writes(self, memory_saver) -> None:
        """Batch flush handles many writes from multiple tasks."""
        thread_id = _generate_id("batchmany")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "loop", "step": 5},
                {"messages": "v1"},
            )

            # Simulate a 5-node graph with writes from each node
            for i in range(5):
                deferred.put_writes(
                    write_config,
                    [(f"channel_{i}", f"value_{i}")],
                    f"task-{i}",
                    f"/node/{i}",
                )

            deferred.flush()
            assert deferred.is_empty

            persisted = memory_saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.pending_writes is not None
            persisted_task_ids = {pw[0] for pw in persisted.pending_writes}
            for i in range(5):
                assert f"task-{i}" in persisted_task_ids
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_get_tuple_after_flush(self, memory_saver) -> None:
        """get_tuple reads from backend after batch flush."""
        thread_id = _generate_id("batchget")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {"messages": "v1"},
            )
            deferred.flush()

            # Buffer is empty — get_tuple should delegate to backend
            assert deferred.is_empty
            result = deferred.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_list_shows_persisted(self, memory_saver) -> None:
        """list() returns batch-flushed checkpoints."""
        thread_id = _generate_id("batchlist")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            # Buffer and flush — list should not show buffered data
            deferred.put(
                config,
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {"messages": "v1"},
            )
            assert len(list(deferred.list(config))) == 0

            deferred.flush()
            assert len(list(deferred.list(config))) >= 1
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_batch_flush_splits_when_exceeding_item_limit(self, memory_saver) -> None:
        """Payload exceeding MAX_PAYLOAD_ITEMS_PER_EVENT splits into multiple calls.

        We create enough writes to push total payload items over 100.
        The chunking in store_blob_events_batch should split the batch
        transparently and all data should persist correctly.
        """
        thread_id = _generate_id("batchsplit")
        actor_id = _generate_id("actor")
        deferred = DeferredCheckpointSaver(memory_saver, batch_writes=True)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

        try:
            # 1 channel event + 1 checkpoint event = 2 payload items from put
            # Each put_writes adds 1 WritesEvent = 1 payload item
            # To exceed 100: we need 99+ writes (2 + 99 = 101 > 100)
            deferred.put(
                config,
                checkpoint,
                {"source": "loop", "step": 10},
                {"messages": "v1"},
            )

            num_writes = 105
            for i in range(num_writes):
                deferred.put_writes(
                    write_config,
                    [(f"channel_{i}", f"value_{i}")],
                    f"task-{i}",
                    f"/node/{i}",
                )

            deferred.flush()
            assert deferred.is_empty

            # Verify checkpoint persisted
            persisted = memory_saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]

            # Verify all writes persisted
            assert persisted.pending_writes is not None
            persisted_task_ids = {pw[0] for pw in persisted.pending_writes}
            for i in range(num_writes):
                assert f"task-{i}" in persisted_task_ids
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_store_blob_events_batch_chunking_by_bytes_direct(
        self, memory_saver
    ) -> None:
        """Directly test byte-based chunking with a lowered max_payload_bytes.

        Calls store_blob_events_batch with max_payload_bytes set below the
        total serialized size, forcing the batch to split across multiple
        CreateEvent API calls. Verifies all events persist correctly.
        """
        from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
            ChannelDataEvent,
            CheckpointEvent,
            WriteItem,
            WritesEvent,
        )

        thread_id = _generate_id("chunkbytes")
        actor_id = _generate_id("actor")

        try:
            large_value = "Y" * 5000  # ~5 KB per channel value
            events: list[ChannelDataEvent | CheckpointEvent | WritesEvent] = []
            for i in range(4):
                events.append(
                    ChannelDataEvent(
                        channel=f"big_ch_{i}",
                        version="v1",
                        value=large_value,
                        thread_id=thread_id,
                        checkpoint_ns="",
                    )
                )
            events.append(
                CheckpointEvent(
                    checkpoint_id="byte-chunk-ckpt",
                    checkpoint_data={
                        "v": 1,
                        "id": "byte-chunk-ckpt",
                        "ts": "2025-01-01",
                    },
                    metadata={"source": "input", "step": 0},
                    parent_checkpoint_id=None,
                    thread_id=thread_id,
                    checkpoint_ns="",
                )
            )
            for i in range(3):
                events.append(
                    WritesEvent(
                        checkpoint_id="byte-chunk-ckpt",
                        writes=[
                            WriteItem(
                                task_id=f"task-byte-{i}",
                                channel=f"write_ch_{i}",
                                value=large_value,
                                task_path=f"/node/{i}",
                            )
                        ],
                    )
                )

            # Measure actual serialized size of one large event
            serialized_sample = (
                memory_saver.checkpoint_event_client.serializer.serialize_event(
                    events[0]
                )
            )
            one_blob_bytes = len(serialized_sample.encode("utf-8"))

            # Set max_payload_bytes to fit ~2 blobs per call
            max_bytes = one_blob_bytes * 2 + 100

            memory_saver.checkpoint_event_client.store_blob_events_batch(
                events,
                thread_id,
                actor_id,
                max_payload_bytes=max_bytes,
            )

            # Verify all 8 events are retrievable
            retrieved = memory_saver.checkpoint_event_client.get_events(
                thread_id, actor_id
            )
            assert len(retrieved) == 8

            # Verify checkpoint present
            checkpoints = [
                e
                for e in retrieved
                if hasattr(e, "checkpoint_id") and hasattr(e, "checkpoint_data")
            ]
            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint_id == "byte-chunk-ckpt"

            # Verify channel data present
            channel_events = [
                e for e in retrieved if hasattr(e, "channel") and hasattr(e, "version")
            ]
            assert len(channel_events) == 4

            # Verify writes present
            writes_events = [
                e
                for e in retrieved
                if hasattr(e, "writes") and not hasattr(e, "checkpoint_data")
            ]
            assert len(writes_events) == 3
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_store_blob_events_batch_chunking_direct(self, memory_saver) -> None:
        """Directly test store_blob_events_batch with a lowered item limit.

        Calls the client directly with max_payload_items=3 and verifies
        all events persist correctly across multiple API calls.
        """
        from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
            ChannelDataEvent,
            CheckpointEvent,
            WriteItem,
            WritesEvent,
        )

        thread_id = _generate_id("chunkdirect")
        actor_id = _generate_id("actor")

        try:
            # Build 7 events: 3 channels + 1 checkpoint + 3 writes
            events: list[ChannelDataEvent | CheckpointEvent | WritesEvent] = []
            for i in range(3):
                events.append(
                    ChannelDataEvent(
                        channel=f"ch_{i}",
                        version="v1",
                        value=f"value_{i}",
                        thread_id=thread_id,
                        checkpoint_ns="",
                    )
                )
            events.append(
                CheckpointEvent(
                    checkpoint_id="chunk-ckpt-1",
                    checkpoint_data={"v": 1, "id": "chunk-ckpt-1", "ts": "2025-01-01"},
                    metadata={"source": "input", "step": 0},
                    parent_checkpoint_id=None,
                    thread_id=thread_id,
                    checkpoint_ns="",
                )
            )
            for i in range(3):
                events.append(
                    WritesEvent(
                        checkpoint_id="chunk-ckpt-1",
                        writes=[
                            WriteItem(
                                task_id=f"task-chunk-{i}",
                                channel=f"write_ch_{i}",
                                value=f"write_val_{i}",
                                task_path=f"/node/{i}",
                            )
                        ],
                    )
                )

            # Store with max_payload_items=3 — forces 3 API calls: [3, 3, 1]
            memory_saver.checkpoint_event_client.store_blob_events_batch(
                events,
                thread_id,
                actor_id,
                max_payload_items=3,
            )

            # Verify all events are retrievable
            retrieved = memory_saver.checkpoint_event_client.get_events(
                thread_id, actor_id
            )
            assert len(retrieved) == 7

            # Verify checkpoint is present
            checkpoints = [
                e
                for e in retrieved
                if hasattr(e, "checkpoint_id") and hasattr(e, "checkpoint_data")
            ]
            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint_id == "chunk-ckpt-1"
        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_put_with_writes_direct(self, memory_saver) -> None:
        """Test put_with_writes directly on AgentCoreMemorySaver."""
        from langgraph_checkpoint_aws.checkpoint.deferred_saver import PendingWrite

        thread_id = _generate_id("directpww")
        actor_id = _generate_id("actor")
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        pending_writes = [
            PendingWrite(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "actor_id": actor_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": checkpoint["id"],
                    }
                },
                writes=[("messages", "direct-write")],
                task_id="task-direct",
                task_path="/direct",
            ),
        ]

        try:
            result = memory_saver.put_with_writes(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {"messages": "v1"},
                pending_writes,
            )

            assert result["configurable"]["checkpoint_id"] == checkpoint["id"]

            saved_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            persisted = memory_saver.get_tuple(saved_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]
            assert persisted.pending_writes is not None
            assert any(pw[0] == "task-direct" for pw in persisted.pending_writes)
        finally:
            memory_saver.delete_thread(thread_id, actor_id)
