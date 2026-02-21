"""Unit tests for BufferedCheckpointSaver."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint

from langgraph_checkpoint_aws.buffered_saver import BufferedCheckpointSaver


class TestBufferedCheckpointSaver:
    """Tests for BufferedCheckpointSaver."""

    def test_init_with_saver(self, simple_checkpoint_saver):
        """Test initialization with a checkpoint saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        assert buffered.saver is simple_checkpoint_saver
        assert buffered._checkpointer is simple_checkpoint_saver
        assert buffered._pending_checkpoint is None
        assert buffered._pending_writes == []
        assert buffered._last_config is None

    def test_init_inherits_serde(self, simple_checkpoint_saver):
        """Test that serde is inherited from the wrapped saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        assert buffered.serde is simple_checkpoint_saver.serde

    def test_saver_property(self, simple_checkpoint_saver):
        """Test saver property returns the underlying checkpointer."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        assert buffered.saver is simple_checkpoint_saver

    def test_has_buffered_checkpoint_false(self, simple_checkpoint_saver):
        """Test has_buffered_checkpoint when no checkpoint is buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        assert buffered.has_buffered_checkpoint is False

    def test_has_buffered_checkpoint_true(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test has_buffered_checkpoint when a checkpoint is buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        assert buffered.has_buffered_checkpoint is True

    def test_has_buffered_writes_false(self, simple_checkpoint_saver):
        """Test has_buffered_writes when no writes are buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        assert buffered.has_buffered_writes is False

    def test_has_buffered_writes_true(self, simple_checkpoint_saver):
        """Test has_buffered_writes when writes are buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )
        buffered.put_writes(config, [("channel1", "value1")], "task1")
        assert buffered.has_buffered_writes is True

    def test_is_empty_true(self, simple_checkpoint_saver):
        """Test is_empty when no data is buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        assert buffered.is_empty is True

    def test_is_empty_false_with_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test is_empty when a checkpoint is buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        assert buffered.is_empty is False

    def test_is_empty_false_with_writes(self, simple_checkpoint_saver):
        """Test is_empty when writes are buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )
        buffered.put_writes(config, [("channel1", "value1")], "task1")
        assert buffered.is_empty is False

    def test_put_buffers_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that put buffers the checkpoint instead of persisting."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        result = buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Verify checkpoint is buffered
        assert buffered._pending_checkpoint is not None
        assert buffered._pending_checkpoint[1] is sample_checkpoint

        # Verify underlying saver has no data (checkpoint not persisted yet)
        underlying_result = simple_checkpoint_saver.get_tuple(config)
        assert underlying_result is None

        # Verify returned config
        assert result["configurable"]["thread_id"] == "thread1"
        assert result["configurable"]["checkpoint_ns"] == "ns1"
        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_put_overwrites_previous_buffered_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that put overwrites a previously buffered checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # First put
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Second put with different checkpoint
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_456",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config, checkpoint2, sample_checkpoint_metadata, {})

        # Verify only the second checkpoint is buffered
        assert buffered._pending_checkpoint[1]["id"] == "checkpoint_456"

    @pytest.mark.asyncio
    async def test_aput_buffers_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that aput buffers the checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        result = await buffered.aput(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # Verify checkpoint is buffered
        assert buffered._pending_checkpoint is not None
        assert buffered._pending_checkpoint[1] is sample_checkpoint

        # Verify returned config
        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_put_writes_buffers_writes(self, simple_checkpoint_saver):
        """Test that put_writes buffers the writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )
        writes = [("channel1", "value1"), ("channel2", "value2")]

        buffered.put_writes(config, writes, "task1", "/path")

        # Verify writes are buffered
        assert len(buffered._pending_writes) == 1
        assert buffered._pending_writes[0] == (config, writes, "task1", "/path")

    def test_put_writes_accumulates(self, simple_checkpoint_saver):
        """Test that multiple put_writes calls accumulate."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        buffered.put_writes(config, [("ch1", "v1")], "task1")
        buffered.put_writes(config, [("ch2", "v2")], "task2")

        assert len(buffered._pending_writes) == 2

    @pytest.mark.asyncio
    async def test_aput_writes_buffers_writes(self, simple_checkpoint_saver):
        """Test that aput_writes buffers the writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        await buffered.aput_writes(config, [("channel1", "value1")], "task1")

        assert len(buffered._pending_writes) == 1

    def test_get_tuple_returns_buffered_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple returns the buffered checkpoint when it matches."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        result = buffered.get_tuple(config)

        assert result is not None
        assert result.checkpoint is sample_checkpoint
        assert result.metadata is sample_checkpoint_metadata

    def test_get_tuple_returns_buffered_with_specific_checkpoint_id(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple returns buffered checkpoint when checkpoint_id matches."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request with specific checkpoint_id
        request_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        result = buffered.get_tuple(request_config)

        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    def test_get_tuple_delegates_when_no_buffer(self, simple_checkpoint_saver):
        """Test get_tuple delegates to underlying saver when no buffer."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        result = buffered.get_tuple(config)

        # MemorySaver has no data, so result is None
        assert result is None

    def test_get_tuple_delegates_when_thread_id_mismatch(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple delegates when thread_id doesn't match."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request with different thread_id
        request_config = RunnableConfig(
            configurable={"thread_id": "thread2", "checkpoint_ns": "ns1"}
        )
        result = buffered.get_tuple(request_config)

        # Buffered checkpoint doesn't match, and underlying saver has no data
        assert result is None

    def test_get_tuple_delegates_when_checkpoint_ns_mismatch(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple delegates when checkpoint_ns doesn't match."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request with different checkpoint_ns
        request_config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns2"}
        )
        result = buffered.get_tuple(request_config)

        # Buffered checkpoint doesn't match, and underlying saver has no data
        assert result is None

    def test_get_tuple_delegates_when_checkpoint_id_mismatch(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple delegates when checkpoint_id doesn't match."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request with different checkpoint_id
        request_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "different_id",
            }
        )
        result = buffered.get_tuple(request_config)

        # Buffered checkpoint doesn't match, and underlying saver has no data
        assert result is None

    def test_get_tuple_includes_pending_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple includes buffered pending writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Add some writes
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1"), ("ch2", "v2")], "task1")

        result = buffered.get_tuple(config)

        assert result is not None
        assert len(result.pending_writes) == 2
        assert result.pending_writes[0] == ("task1", "ch1", "v1")
        assert result.pending_writes[1] == ("task1", "ch2", "v2")

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_buffered_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aget_tuple returns the buffered checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        await buffered.aput(config, sample_checkpoint, sample_checkpoint_metadata, {})

        result = await buffered.aget_tuple(config)

        assert result is not None
        assert result.checkpoint is sample_checkpoint

    @pytest.mark.asyncio
    async def test_aget_tuple_delegates_when_no_buffer(self, simple_checkpoint_saver):
        """Test aget_tuple delegates to underlying saver when no buffer."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        result = await buffered.aget_tuple(config)

        # MemorySaver has no data
        assert result is None

    def test_flush_persists_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush persists the buffered checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Before flush: underlying saver has no data
        assert simple_checkpoint_saver.get_tuple(config) is None

        result = buffered.flush()

        # After flush: underlying saver has the checkpoint
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["id"] == sample_checkpoint["id"]

        # Verify buffer is cleared
        assert buffered._pending_checkpoint is None

        # Verify returned config
        assert result is not None
        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_flush_persists_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush persists buffered writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First persist a checkpoint to the underlying saver so writes have a target
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        simple_checkpoint_saver.put(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # Now buffer writes via BufferedCheckpointSaver
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")
        buffered.put_writes(writes_config, [("ch2", "v2")], "task2")

        buffered.flush()

        # Verify buffer is cleared
        assert len(buffered._pending_writes) == 0

        # Verify writes were persisted (check via get_tuple)
        persisted = simple_checkpoint_saver.get_tuple(writes_config)
        assert persisted is not None
        # MemorySaver stores pending writes
        assert len(persisted.pending_writes) >= 2

    def test_flush_persists_checkpoint_then_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush persists checkpoint before writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        buffered.put_writes(
            RunnableConfig(
                configurable={
                    "thread_id": "thread1",
                    "checkpoint_ns": "ns1",
                    "checkpoint_id": sample_checkpoint["id"],
                }
            ),
            [("ch1", "v1")],
            "task1",
        )

        buffered.flush()

        # Verify both checkpoint and writes were persisted
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["id"] == sample_checkpoint["id"]

    def test_flush_returns_none_when_empty(self, simple_checkpoint_saver):
        """Test flush returns None when nothing to flush."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        result = buffered.flush()

        assert result is None

    @pytest.mark.asyncio
    async def test_aflush_persists_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aflush persists the buffered checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        await buffered.aput(config, sample_checkpoint, sample_checkpoint_metadata, {})

        result = await buffered.aflush()

        # Verify checkpoint was persisted
        persisted = await simple_checkpoint_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["id"] == sample_checkpoint["id"]

        # Verify buffer is cleared
        assert buffered._pending_checkpoint is None

        # Verify returned config
        assert result is not None

    @pytest.mark.asyncio
    async def test_aflush_persists_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aflush persists buffered writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First persist a checkpoint to the underlying saver
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        await simple_checkpoint_saver.aput(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # Buffer writes
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        await buffered.aput_writes(writes_config, [("ch1", "v1")], "task1")

        await buffered.aflush()

        # Verify buffer is cleared
        assert len(buffered._pending_writes) == 0

    def test_flush_on_exit_flushes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush_on_exit context manager flushes on exit."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        with buffered.flush_on_exit():
            buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
            # Not flushed yet - underlying saver has no data
            assert simple_checkpoint_saver.get_tuple(config) is None

        # Flushed after exiting context
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["id"] == sample_checkpoint["id"]

    def test_flush_on_exit_yields_self(self, simple_checkpoint_saver):
        """Test flush_on_exit yields the buffered saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        with buffered.flush_on_exit() as ctx:
            assert ctx is buffered

    def test_flush_on_exit_flushes_on_exception(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush_on_exit flushes even when an exception occurs."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        with pytest.raises(ValueError):
            with buffered.flush_on_exit():
                buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
                raise ValueError("Test exception")

        # Still flushed despite exception
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None

    @pytest.mark.asyncio
    async def test_aflush_on_exit_flushes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aflush_on_exit context manager flushes on exit."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        async with buffered.aflush_on_exit():
            await buffered.aput(
                config, sample_checkpoint, sample_checkpoint_metadata, {}
            )
            # Not flushed yet
            assert await simple_checkpoint_saver.aget_tuple(config) is None

        # Flushed after exiting context
        persisted = await simple_checkpoint_saver.aget_tuple(config)
        assert persisted is not None

    @pytest.mark.asyncio
    async def test_aflush_on_exit_yields_self(self, simple_checkpoint_saver):
        """Test aflush_on_exit yields the buffered saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        async with buffered.aflush_on_exit() as ctx:
            assert ctx is buffered

    @pytest.mark.asyncio
    async def test_aflush_on_exit_flushes_on_exception(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aflush_on_exit flushes even when an exception occurs."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        with pytest.raises(ValueError):
            async with buffered.aflush_on_exit():
                await buffered.aput(
                    config, sample_checkpoint, sample_checkpoint_metadata, {}
                )
                raise ValueError("Test exception")

        # Still flushed despite exception
        persisted = await simple_checkpoint_saver.aget_tuple(config)
        assert persisted is not None

    def test_list_delegates_to_underlying_saver(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test list delegates to the underlying saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist a checkpoint directly to the underlying saver
        simple_checkpoint_saver.put(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # List via buffered saver should return the persisted checkpoint
        result = list(buffered.list(config))
        assert len(result) == 1
        assert result[0].checkpoint["id"] == sample_checkpoint["id"]

    def test_list_does_not_include_buffered_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test list does not include buffered checkpoint (only persisted)."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Buffer a checkpoint (not persisted)
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # List should not include the buffered checkpoint
        result = list(buffered.list(config))
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_alist_delegates_to_underlying_saver(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test alist delegates to the underlying saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist a checkpoint directly to the underlying saver
        await simple_checkpoint_saver.aput(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # List via buffered saver should return the persisted checkpoint
        result = []
        async for item in buffered.alist(config):
            result.append(item)

        assert len(result) == 1
        assert result[0].checkpoint["id"] == sample_checkpoint["id"]

    def test_clear_discards_buffered_checkpoint(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test clear discards buffered checkpoint without persisting."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        assert buffered.has_buffered_checkpoint is True

        buffered.clear()

        assert buffered.has_buffered_checkpoint is False
        assert buffered._pending_checkpoint is None

        # Verify nothing was persisted
        assert simple_checkpoint_saver.get_tuple(config) is None

    def test_clear_discards_buffered_writes(self, simple_checkpoint_saver):
        """Test clear discards buffered writes without persisting."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        buffered.put_writes(config, [("ch1", "v1")], "task1")
        assert buffered.has_buffered_writes is True

        buffered.clear()

        assert buffered.has_buffered_writes is False
        assert len(buffered._pending_writes) == 0

    def test_clear_resets_last_config(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test clear resets _last_config."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        assert buffered._last_config is not None

        buffered.clear()

        assert buffered._last_config is None

    def test_get_next_version_delegates(self, simple_checkpoint_saver):
        """Test get_next_version delegates to underlying saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # MemorySaver returns incrementing integers as strings
        result = buffered.get_next_version(None)
        assert result is not None

        # Subsequent calls should return incrementing values
        result2 = buffered.get_next_version(result)
        assert result2 is not None
        assert result2 != result
