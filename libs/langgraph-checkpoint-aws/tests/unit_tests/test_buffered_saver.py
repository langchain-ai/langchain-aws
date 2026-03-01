"""Unit tests for BufferedCheckpointSaver."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph_checkpoint_aws.buffered_saver import BufferedCheckpointSaver


class TestBufferedCheckpointSaver:
    """Tests for BufferedCheckpointSaver."""

    def test_init_with_saver(self, simple_checkpoint_saver):
        """Test initialization with a checkpoint saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        assert buffered.saver is simple_checkpoint_saver
        assert buffered._saver is simple_checkpoint_saver
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


class TestBufferedCheckpointSaverEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "config",
        [
            RunnableConfig(configurable={}),
            {},
        ],
        ids=["empty_configurable", "no_configurable"],
    )
    def test_put_with_missing_or_empty_configurable(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata, config
    ):
        """Test put with empty or missing configurable dict."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        result = buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        assert buffered._pending_checkpoint is not None
        # Should handle missing/empty configurable gracefully
        assert result["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_get_tuple_with_empty_configurable(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple with empty configurable matches buffered checkpoint with empty configurable."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        # Put with empty configurable
        config = RunnableConfig(configurable={})
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Get with empty configurable should match
        result = buffered.get_tuple(RunnableConfig(configurable={}))
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    @pytest.mark.parametrize(
        "request_configurable",
        [
            {"thread_id": None, "checkpoint_ns": "ns1"},
            {"checkpoint_ns": "ns1"},  # missing thread_id treated as None
        ],
        ids=["explicit_none_thread_id", "missing_thread_id"],
    )
    def test_get_tuple_with_none_or_missing_thread_id(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata, request_configurable
    ):
        """Test get_tuple with None or missing thread_id matches buffered with None thread_id."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(configurable={"thread_id": None, "checkpoint_ns": "ns1"})
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request should match (both resolve to None)
        result = buffered.get_tuple(RunnableConfig(configurable=request_configurable))
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    def test_put_writes_with_empty_writes_list(self, simple_checkpoint_saver):
        """Test put_writes with empty writes list."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        buffered.put_writes(config, [], "task1")

        assert len(buffered._pending_writes) == 1
        assert buffered._pending_writes[0][1] == []

    def test_flush_with_empty_writes_list(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test flush correctly handles empty writes list."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Put checkpoint
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Put empty writes
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [], "task1")

        # Flush should not raise
        buffered.flush()
        assert len(buffered._pending_writes) == 0

    def test_put_after_flush(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that put works correctly after a flush."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # First put and flush
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        buffered.flush()

        # Second put should work
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_new",
            ts="2024-01-02T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        result = buffered.put(config, checkpoint2, sample_checkpoint_metadata, {})

        assert buffered.has_buffered_checkpoint
        assert result["configurable"]["checkpoint_id"] == "checkpoint_new"

    def test_put_after_clear(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that put works correctly after a clear."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # First put and clear
        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        buffered.clear()

        # Second put should work
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_new",
            ts="2024-01-02T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        result = buffered.put(config, checkpoint2, sample_checkpoint_metadata, {})

        assert buffered.has_buffered_checkpoint
        assert result["configurable"]["checkpoint_id"] == "checkpoint_new"


class TestBufferedCheckpointSaverMultipleThreadsAndNamespaces:
    """Tests for handling multiple threads and namespaces."""

    @pytest.mark.parametrize(
        "mismatched_key,mismatched_value",
        [
            ("thread_id", "thread2"),
            ("checkpoint_ns", "ns2"),
            ("checkpoint_id", "different_checkpoint_id"),
        ],
        ids=["different_thread", "different_namespace", "different_checkpoint_id"],
    )
    def test_writes_for_mismatched_config_not_included(
        self,
        simple_checkpoint_saver,
        sample_checkpoint,
        sample_checkpoint_metadata,
        mismatched_key,
        mismatched_value,
    ):
        """Test that writes with mismatched thread_id/namespace/checkpoint_id are not included."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Add writes for the buffered checkpoint
        writes_config1 = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config1, [("ch1", "v1")], "task1")

        # Add writes with a mismatched config key
        mismatched_configurable = {
            "thread_id": "thread1",
            "checkpoint_ns": "ns1",
            "checkpoint_id": sample_checkpoint["id"],
        }
        mismatched_configurable[mismatched_key] = mismatched_value
        writes_config2 = RunnableConfig(configurable=mismatched_configurable)
        buffered.put_writes(writes_config2, [("ch2", "v2")], "task2")

        result = buffered.get_tuple(config)

        # Only writes for the matching config should be included
        assert result is not None
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0] == ("task1", "ch1", "v1")

    def test_flush_persists_writes_for_all_threads(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that flush persists writes for all threads."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # Create checkpoints for two threads in underlying saver
        config1 = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        config2 = RunnableConfig(
            configurable={"thread_id": "thread2", "checkpoint_ns": "ns1"}
        )

        simple_checkpoint_saver.put(config1, sample_checkpoint, sample_checkpoint_metadata, {})

        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_456",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        simple_checkpoint_saver.put(config2, checkpoint2, sample_checkpoint_metadata, {})

        # Buffer writes for both threads
        writes_config1 = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        writes_config2 = RunnableConfig(
            configurable={
                "thread_id": "thread2",
                "checkpoint_ns": "ns1",
                "checkpoint_id": checkpoint2["id"],
            }
        )

        buffered.put_writes(writes_config1, [("ch1", "v1")], "task1")
        buffered.put_writes(writes_config2, [("ch2", "v2")], "task2")

        buffered.flush()

        # Both writes should be persisted
        assert len(buffered._pending_writes) == 0

        # Check writes were persisted to underlying saver
        result1 = simple_checkpoint_saver.get_tuple(writes_config1)
        result2 = simple_checkpoint_saver.get_tuple(writes_config2)

        assert result1 is not None
        assert result2 is not None
        assert len(result1.pending_writes) >= 1
        assert len(result2.pending_writes) >= 1


class TestBufferedCheckpointSaverParentConfig:
    """Tests for parent config handling."""

    def test_parent_config_is_set_when_checkpoint_id_present(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that parent_config is set when config has checkpoint_id."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "parent_checkpoint_id",
            }
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        result = buffered.get_tuple(
            RunnableConfig(
                configurable={
                    "thread_id": "thread1",
                    "checkpoint_ns": "ns1",
                    "checkpoint_id": sample_checkpoint["id"],
                }
            )
        )

        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent_checkpoint_id"

    def test_parent_config_is_none_when_no_checkpoint_id(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that parent_config is None when config has no checkpoint_id."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        result = buffered.get_tuple(config)

        assert result is not None
        assert result.parent_config is None

    def test_checkpoint_chaining(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test checkpoint chaining - multiple checkpoints with parent references."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First checkpoint (no parent)
        config1 = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        checkpoint1 = Checkpoint(
            v=1,
            id="checkpoint_1",
            ts="2024-01-01T00:00:00Z",
            channel_values={"state": "initial"},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config1, checkpoint1, sample_checkpoint_metadata, {})
        buffered.flush()

        # Second checkpoint (parent is first)
        config2 = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "checkpoint_1",
            }
        )
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_2",
            ts="2024-01-01T00:01:00Z",
            channel_values={"state": "updated"},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config2, checkpoint2, sample_checkpoint_metadata, {})

        # Get the buffered checkpoint
        result = buffered.get_tuple(
            RunnableConfig(configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"})
        )

        assert result is not None
        assert result.checkpoint["id"] == "checkpoint_2"
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "checkpoint_1"


class TestBufferedCheckpointSaverWritesOrdering:
    """Tests for writes ordering and filtering."""

    def test_writes_preserve_order(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that writes preserve their insertion order."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Add writes in specific order
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")
        buffered.put_writes(writes_config, [("ch2", "v2")], "task2")
        buffered.put_writes(writes_config, [("ch3", "v3")], "task3")

        result = buffered.get_tuple(config)

        assert result is not None
        assert len(result.pending_writes) == 3
        assert result.pending_writes[0] == ("task1", "ch1", "v1")
        assert result.pending_writes[1] == ("task2", "ch2", "v2")
        assert result.pending_writes[2] == ("task3", "ch3", "v3")

    def test_multiple_writes_in_single_put_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that multiple writes in a single put_writes call are all included."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Add multiple writes in a single call
        buffered.put_writes(
            writes_config,
            [("ch1", "v1"), ("ch2", "v2"), ("ch3", "v3")],
            "task1",
        )

        result = buffered.get_tuple(config)

        assert result is not None
        assert len(result.pending_writes) == 3
        assert result.pending_writes[0] == ("task1", "ch1", "v1")
        assert result.pending_writes[1] == ("task1", "ch2", "v2")
        assert result.pending_writes[2] == ("task1", "ch3", "v3")

    def test_writes_with_same_channel_different_tasks(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test writes to the same channel from different tasks."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Same channel, different tasks
        buffered.put_writes(writes_config, [("messages", "msg1")], "task1")
        buffered.put_writes(writes_config, [("messages", "msg2")], "task2")

        result = buffered.get_tuple(config)

        assert result is not None
        assert len(result.pending_writes) == 2
        assert result.pending_writes[0] == ("task1", "messages", "msg1")
        assert result.pending_writes[1] == ("task2", "messages", "msg2")

    def test_flush_preserves_writes_order(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that flush persists writes in order."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # First persist checkpoint to underlying saver
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Add writes in specific order
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")
        buffered.put_writes(writes_config, [("ch2", "v2")], "task2")
        buffered.put_writes(writes_config, [("ch3", "v3")], "task3")

        buffered.flush()

        # Check writes were persisted to underlying saver
        result = simple_checkpoint_saver.get_tuple(writes_config)
        assert result is not None
        # MemorySaver preserves write order
        assert len(result.pending_writes) == 3


class TestBufferedCheckpointSaverErrorHandling:
    """Tests for error handling scenarios."""

    def test_flush_on_exit_propagates_exception(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that flush_on_exit propagates exceptions after flushing."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        exception_raised = False
        try:
            with buffered.flush_on_exit():
                buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
                raise ValueError("Test exception")
        except ValueError:
            exception_raised = True

        assert exception_raised
        # Data should still be flushed
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None

    @pytest.mark.asyncio
    async def test_aflush_on_exit_propagates_exception(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that aflush_on_exit propagates exceptions after flushing."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        exception_raised = False
        try:
            async with buffered.aflush_on_exit():
                await buffered.aput(config, sample_checkpoint, sample_checkpoint_metadata, {})
                raise ValueError("Test exception")
        except ValueError:
            exception_raised = True

        assert exception_raised
        # Data should still be flushed
        persisted = await simple_checkpoint_saver.aget_tuple(config)
        assert persisted is not None

    def test_nested_flush_on_exit(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test nested flush_on_exit context managers."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        with buffered.flush_on_exit():
            buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
            with buffered.flush_on_exit():
                # Inner context should flush (buffer is now empty)
                pass
            # After inner context, buffer should be empty
            assert buffered.is_empty

        # Verify checkpoint was persisted
        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None


class TestBufferedCheckpointSaverComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_realistic_graph_execution_pattern(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test a realistic pattern of checkpoint/write operations during graph execution."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": ""}
        )

        # Simulate START node checkpoint
        checkpoint_start = Checkpoint(
            v=1,
            id="cp_start",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": []},
            channel_versions={"messages": "1"},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config, checkpoint_start, sample_checkpoint_metadata, {})

        # Simulate writes from START
        writes_config_start = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp_start",
            }
        )
        buffered.put_writes(writes_config_start, [("messages", {"role": "user", "content": "Hello"})], "start_task")

        # Simulate agent node checkpoint (overwrites previous)
        config_with_parent = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp_start",
            }
        )
        checkpoint_agent = Checkpoint(
            v=1,
            id="cp_agent",
            ts="2024-01-01T00:00:01Z",
            channel_values={"messages": [{"role": "user", "content": "Hello"}]},
            channel_versions={"messages": "2"},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config_with_parent, checkpoint_agent, sample_checkpoint_metadata, {})

        # Simulate writes from agent
        writes_config_agent = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp_agent",
            }
        )
        buffered.put_writes(writes_config_agent, [("messages", {"role": "assistant", "content": "Hi!"})], "agent_task")

        # Verify only the latest checkpoint is buffered
        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "cp_agent"
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "cp_start"

        # Verify only writes for the current checkpoint are included
        # (writes for cp_start should not be included since checkpoint was overwritten)
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0][0] == "agent_task"

        # Flush and verify persistence
        buffered.flush()

        persisted = simple_checkpoint_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["id"] == "cp_agent"

    def test_subgraph_execution_pattern(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test checkpoint operations with subgraph namespaces."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # Main graph checkpoint
        main_config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": ""}
        )
        main_checkpoint = Checkpoint(
            v=1,
            id="main_cp",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(main_config, main_checkpoint, sample_checkpoint_metadata, {})
        buffered.flush()

        # Subgraph checkpoint (different namespace)
        subgraph_config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "subgraph:node1"}
        )
        subgraph_checkpoint = Checkpoint(
            v=1,
            id="subgraph_cp",
            ts="2024-01-01T00:00:01Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(subgraph_config, subgraph_checkpoint, sample_checkpoint_metadata, {})

        # Get main graph checkpoint (should delegate to underlying saver)
        main_result = buffered.get_tuple(main_config)
        assert main_result is not None
        assert main_result.checkpoint["id"] == "main_cp"

        # Get subgraph checkpoint (should return buffered)
        subgraph_result = buffered.get_tuple(subgraph_config)
        assert subgraph_result is not None
        assert subgraph_result.checkpoint["id"] == "subgraph_cp"

    def test_writes_only_no_checkpoint(self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata):
        """Test buffering only writes without a checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First persist a checkpoint to underlying saver
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Buffer only writes
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")

        assert buffered.has_buffered_writes
        assert not buffered.has_buffered_checkpoint

        # get_tuple should delegate to underlying saver (no buffered checkpoint)
        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]
        # Note: buffered writes are NOT included when checkpoint is from underlying saver
        # This is correct behavior - writes are associated with the buffered checkpoint

        # Flush should persist the writes
        buffered.flush()
        assert not buffered.has_buffered_writes

    def test_large_number_of_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test handling a large number of buffered writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Add many writes
        num_writes = 100
        for i in range(num_writes):
            buffered.put_writes(writes_config, [(f"channel_{i}", f"value_{i}")], f"task_{i}")

        result = buffered.get_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == num_writes

        # Verify order is preserved
        for i in range(num_writes):
            assert result.pending_writes[i] == (f"task_{i}", f"channel_{i}", f"value_{i}")

        # Flush should handle all writes
        buffered.flush()
        assert len(buffered._pending_writes) == 0

    def test_complex_channel_values(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test checkpoint with complex channel values."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        complex_checkpoint = Checkpoint(
            v=1,
            id="complex_cp",
            ts="2024-01-01T00:00:00Z",
            channel_values={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "state": {
                    "nested": {
                        "deeply": {
                            "value": [1, 2, 3, {"key": "value"}]
                        }
                    }
                },
                "binary_data": b"some binary data",
                "none_value": None,
                "empty_list": [],
                "empty_dict": {},
            },
            channel_versions={"messages": "3", "state": "1"},
            versions_seen={"node1": {"messages": "2"}},
            pending_sends=[{"type": "send", "data": "test"}],
        )

        buffered.put(config, complex_checkpoint, sample_checkpoint_metadata, {})

        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"]["messages"][0]["content"] == "Hello"
        assert result.checkpoint["channel_values"]["state"]["nested"]["deeply"]["value"][3]["key"] == "value"
        assert result.checkpoint["pending_sends"][0]["type"] == "send"

    def test_metadata_preservation(self, simple_checkpoint_saver, sample_checkpoint):
        """Test that checkpoint metadata is preserved correctly."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        complex_metadata = CheckpointMetadata(
            source="update",
            step=5,
            writes={
                "node1": {"key": "value"},
                "node2": [1, 2, 3],
            },
            parents={
                "": "parent_cp_1",
                "subgraph:node1": "parent_cp_2",
            },
        )

        buffered.put(config, sample_checkpoint, complex_metadata, {})

        result = buffered.get_tuple(config)
        assert result is not None
        assert result.metadata["source"] == "update"
        assert result.metadata["step"] == 5
        assert result.metadata["writes"]["node1"]["key"] == "value"
        assert result.metadata["parents"][""] == "parent_cp_1"


class TestBufferedCheckpointSaverListBehavior:
    """Tests for list method behavior."""

    def test_list_with_filter(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test list with filter parameter."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist a checkpoint with specific metadata
        metadata_with_source = CheckpointMetadata(
            source="input",
            step=0,
            writes={},
            parents={},
        )
        simple_checkpoint_saver.put(config, sample_checkpoint, metadata_with_source, {})

        # List with filter should delegate to underlying saver
        results = list(buffered.list(config, filter={"source": "input"}))
        assert len(results) == 1

    def test_list_with_limit(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test list with limit parameter."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist multiple checkpoints
        for i in range(5):
            checkpoint = Checkpoint(
                v=1,
                id=f"checkpoint_{i}",
                ts=f"2024-01-01T00:00:0{i}Z",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )
            simple_checkpoint_saver.put(config, checkpoint, sample_checkpoint_metadata, {})

        # List with limit
        results = list(buffered.list(config, limit=3))
        assert len(results) == 3

    def test_list_with_before(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test list with before parameter."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint = Checkpoint(
                v=1,
                id=f"checkpoint_{i}",
                ts=f"2024-01-01T00:00:0{i}Z",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )
            simple_checkpoint_saver.put(config, checkpoint, sample_checkpoint_metadata, {})
            checkpoint_ids.append(f"checkpoint_{i}")

        # List before the last checkpoint
        before_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": checkpoint_ids[-1],
            }
        )
        results = list(buffered.list(config, before=before_config))
        # Should return checkpoints before the specified one
        assert all(r.checkpoint["id"] != checkpoint_ids[-1] for r in results)

    @pytest.mark.asyncio
    async def test_alist_with_parameters(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test async list with various parameters."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist multiple checkpoints
        for i in range(3):
            checkpoint = Checkpoint(
                v=1,
                id=f"checkpoint_{i}",
                ts=f"2024-01-01T00:00:0{i}Z",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )
            await simple_checkpoint_saver.aput(config, checkpoint, sample_checkpoint_metadata, {})

        # Async list with limit
        results = []
        async for item in buffered.alist(config, limit=2):
            results.append(item)

        assert len(results) == 2


class TestBufferedCheckpointSaverUnderlyingSaverFailures:
    """Tests for handling underlying saver failures."""

    def test_flush_checkpoint_failure_preserves_all_data(
        self, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that if checkpoint flush fails, all data is preserved for retry."""
        from unittest.mock import patch
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        buffered = BufferedCheckpointSaver(saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")

        # Mock the underlying saver's put to raise an exception
        with patch.object(saver, "put", side_effect=Exception("Storage failure")):
            with pytest.raises(Exception, match="Storage failure"):
                buffered.flush()

        # Both checkpoint and writes should be preserved for retry
        # (implementation doesn't clear checkpoint on failure)
        assert buffered._pending_checkpoint is not None
        assert len(buffered._pending_writes) == 1

        # After fixing the issue, retry should succeed
        result = buffered.flush()
        assert result is not None
        assert buffered.is_empty

    def test_flush_writes_failure_preserves_remaining_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that if a write flush fails, remaining writes are preserved."""
        from unittest.mock import patch

        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First persist checkpoint to underlying saver
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )

        # Add multiple writes
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")
        buffered.put_writes(writes_config, [("ch2", "v2")], "task2")
        buffered.put_writes(writes_config, [("ch3", "v3")], "task3")

        # Make put_writes fail on the second call
        call_count = [0]
        original_put_writes = simple_checkpoint_saver.put_writes

        def failing_put_writes(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Write failure")
            return original_put_writes(*args, **kwargs)

        with patch.object(simple_checkpoint_saver, "put_writes", side_effect=failing_put_writes):
            with pytest.raises(Exception, match="Write failure"):
                buffered.flush()

        # First write should be removed (succeeded)
        # Second and third writes should still be in buffer
        assert len(buffered._pending_writes) == 2

    @pytest.mark.asyncio
    async def test_aflush_checkpoint_failure_preserves_all_data(
        self, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that if async checkpoint flush fails, all data is preserved for retry."""
        from unittest.mock import patch
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        buffered = BufferedCheckpointSaver(saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        await buffered.aput(config, sample_checkpoint, sample_checkpoint_metadata, {})
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        await buffered.aput_writes(writes_config, [("ch1", "v1")], "task1")

        # Mock the underlying saver's aput to raise an exception
        async def failing_aput(*args, **kwargs):
            raise Exception("Async storage failure")

        with patch.object(saver, "aput", side_effect=failing_aput):
            with pytest.raises(Exception, match="Async storage failure"):
                await buffered.aflush()

        # Both checkpoint and writes should be preserved for retry
        assert buffered._pending_checkpoint is not None
        assert len(buffered._pending_writes) == 1

        # After fixing the issue, retry should succeed
        result = await buffered.aflush()
        assert result is not None
        assert buffered.is_empty


class TestBufferedCheckpointSaverConfigPreservation:
    """Tests for ensuring config is properly preserved and passed through."""

    def test_put_preserves_extra_configurable_keys(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that put preserves extra keys in configurable."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "custom_key": "custom_value",
                "another_key": 123,
            }
        )

        result = buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Extra keys should be preserved in the returned config
        assert result["configurable"]["custom_key"] == "custom_value"
        assert result["configurable"]["another_key"] == 123

    def test_get_tuple_config_includes_all_keys(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that get_tuple returns config with all original keys."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "custom_key": "custom_value",
            }
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        result = buffered.get_tuple(config)

        assert result is not None
        assert result.config["configurable"]["custom_key"] == "custom_value"
        assert result.config["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_parent_config_preserves_extra_keys(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that parent_config preserves extra keys from original config."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "parent_checkpoint",
                "custom_key": "custom_value",
            }
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        result = buffered.get_tuple(
            RunnableConfig(
                configurable={
                    "thread_id": "thread1",
                    "checkpoint_ns": "ns1",
                    "checkpoint_id": sample_checkpoint["id"],
                }
            )
        )

        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["custom_key"] == "custom_value"
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent_checkpoint"


class TestBufferedCheckpointSaverChannelVersions:
    """Tests for channel versions handling."""

    def test_channel_versions_passed_to_underlying_saver(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that channel versions are passed through to underlying saver on flush."""
        from unittest.mock import patch, MagicMock

        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        new_versions = {"messages": "v5", "state": "v3"}

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, new_versions)

        # Capture the call to underlying saver
        with patch.object(simple_checkpoint_saver, "put", wraps=simple_checkpoint_saver.put) as mock_put:
            buffered.flush()
            mock_put.assert_called_once()
            call_args = mock_put.call_args
            # Fourth argument should be new_versions
            assert call_args[0][3] == new_versions


class TestBufferedCheckpointSaverTaskPath:
    """Tests for task_path handling in writes."""

    def test_task_path_preserved_in_buffered_writes(self, simple_checkpoint_saver):
        """Test that task_path is preserved in buffered writes."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        buffered.put_writes(config, [("ch1", "v1")], "task1", "/path/to/task")

        assert len(buffered._pending_writes) == 1
        assert buffered._pending_writes[0][3] == "/path/to/task"

    def test_task_path_passed_to_underlying_saver(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that task_path is passed to underlying saver on flush."""
        from unittest.mock import patch

        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # First persist checkpoint
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1", "/custom/path")

        with patch.object(
            simple_checkpoint_saver, "put_writes", wraps=simple_checkpoint_saver.put_writes
        ) as mock_put_writes:
            buffered.flush()
            mock_put_writes.assert_called_once()
            call_args = mock_put_writes.call_args
            # Fourth argument should be task_path
            assert call_args[0][3] == "/custom/path"

    def test_default_task_path_is_empty_string(self, simple_checkpoint_saver):
        """Test that default task_path is empty string."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )

        # Call without task_path
        buffered.put_writes(config, [("ch1", "v1")], "task1")

        assert buffered._pending_writes[0][3] == ""


class TestBufferedCheckpointSaverSpecialValues:
    """Tests for handling special values in checkpoints and writes."""

    def test_write_with_none_value(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test writes with None values."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("channel", None)], "task1")

        result = buffered.get_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0] == ("task1", "channel", None)

    def test_write_with_complex_nested_value(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test writes with complex nested values."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        complex_value = {
            "messages": [
                {"role": "user", "content": "Hello", "metadata": {"timestamp": 123}},
                {"role": "assistant", "content": "Hi!", "tool_calls": [{"name": "search"}]},
            ],
            "nested": {"deep": {"value": [1, 2, {"key": "value"}]}},
        }

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("state", complex_value)], "task1")

        result = buffered.get_tuple(config)
        assert result is not None
        assert result.pending_writes[0][2] == complex_value

    def test_checkpoint_with_empty_channel_values(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test checkpoint with empty channel values."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        empty_checkpoint = Checkpoint(
            v=1,
            id="empty_cp",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )

        buffered.put(config, empty_checkpoint, sample_checkpoint_metadata, {})

        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"] == {}

    def test_write_with_empty_channel_name(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test writes with empty channel name."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("", "value")], "task1")

        result = buffered.get_tuple(config)
        assert result is not None
        assert result.pending_writes[0] == ("task1", "", "value")


class TestBufferedCheckpointSaverWrappedSaverTypes:
    """Tests for ensuring BufferedCheckpointSaver works with different saver types."""

    def test_wrapping_another_buffered_saver(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test wrapping a BufferedCheckpointSaver with another BufferedCheckpointSaver."""
        inner_buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        outer_buffered = BufferedCheckpointSaver(inner_buffered)

        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        outer_buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Outer flush should persist to inner buffer
        outer_buffered.flush()
        assert outer_buffered.is_empty
        assert inner_buffered.has_buffered_checkpoint

        # Inner flush should persist to underlying saver
        inner_buffered.flush()
        assert inner_buffered.is_empty

        # Verify persisted
        result = simple_checkpoint_saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    def test_serde_inheritance(self, simple_checkpoint_saver):
        """Test that serde is properly inherited from wrapped saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # The serde should be the same object
        assert buffered.serde is simple_checkpoint_saver.serde


class TestBufferedCheckpointSaverIdempotency:
    """Tests for idempotent operations."""

    def test_clear_is_idempotent(self, simple_checkpoint_saver):
        """Test that clear can be called multiple times safely."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # Clear on empty buffer
        buffered.clear()
        assert buffered.is_empty

        # Clear again
        buffered.clear()
        assert buffered.is_empty

    def test_flush_is_idempotent(self, simple_checkpoint_saver):
        """Test that flush can be called multiple times safely."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # Flush on empty buffer
        result1 = buffered.flush()
        assert result1 is None

        # Flush again
        result2 = buffered.flush()
        assert result2 is None

    @pytest.mark.asyncio
    async def test_aflush_is_idempotent(self, simple_checkpoint_saver):
        """Test that aflush can be called multiple times safely."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        # Aflush on empty buffer
        result1 = await buffered.aflush()
        assert result1 is None

        # Aflush again
        result2 = await buffered.aflush()
        assert result2 is None


class TestBufferedCheckpointSaverContextManagerEdgeCases:
    """Tests for context manager edge cases."""

    def test_flush_on_exit_with_no_operations(self, simple_checkpoint_saver):
        """Test flush_on_exit when no operations are performed inside."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        with buffered.flush_on_exit():
            pass  # No operations

        assert buffered.is_empty

    @pytest.mark.asyncio
    async def test_aflush_on_exit_with_no_operations(self, simple_checkpoint_saver):
        """Test aflush_on_exit when no operations are performed inside."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)

        async with buffered.aflush_on_exit():
            pass  # No operations

        assert buffered.is_empty

    def test_flush_on_exit_can_be_reused(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that flush_on_exit can be used multiple times on same instance."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config1 = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        config2 = RunnableConfig(
            configurable={"thread_id": "thread2", "checkpoint_ns": "ns1"}
        )

        # First use
        with buffered.flush_on_exit():
            buffered.put(config1, sample_checkpoint, sample_checkpoint_metadata, {})

        # Second use with different data
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_456",
            ts="2024-01-02T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        with buffered.flush_on_exit():
            buffered.put(config2, checkpoint2, sample_checkpoint_metadata, {})

        # Both should be persisted
        result1 = simple_checkpoint_saver.get_tuple(config1)
        result2 = simple_checkpoint_saver.get_tuple(config2)
        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_aflush_on_exit_can_be_reused(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that aflush_on_exit can be used multiple times on same instance."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config1 = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )
        config2 = RunnableConfig(
            configurable={"thread_id": "thread2", "checkpoint_ns": "ns1"}
        )

        # First use
        async with buffered.aflush_on_exit():
            await buffered.aput(config1, sample_checkpoint, sample_checkpoint_metadata, {})

        # Second use with different data
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_456",
            ts="2024-01-02T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        async with buffered.aflush_on_exit():
            await buffered.aput(config2, checkpoint2, sample_checkpoint_metadata, {})

        # Both should be persisted
        result1 = await simple_checkpoint_saver.aget_tuple(config1)
        result2 = await simple_checkpoint_saver.aget_tuple(config2)
        assert result1 is not None
        assert result2 is not None


class TestBufferedCheckpointSaverGetTupleEdgeCases:
    """Additional edge case tests for get_tuple behavior."""

    def test_get_tuple_with_checkpoint_id_none_matches_buffered(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test get_tuple with checkpoint_id=None explicitly matches buffered."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Request with explicit checkpoint_id=None
        request_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": None,
            }
        )
        result = buffered.get_tuple(request_config)

        # Should match because None means "get latest"
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    def test_get_tuple_after_clear_delegates_to_underlying(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that get_tuple delegates to underlying saver after clear."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # First persist to underlying saver
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Buffer a different checkpoint
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_new",
            ts="2024-01-02T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        buffered.put(config, checkpoint2, sample_checkpoint_metadata, {})

        # get_tuple should return buffered checkpoint
        result = buffered.get_tuple(config)
        assert result.checkpoint["id"] == "checkpoint_new"

        # Clear the buffer
        buffered.clear()

        # Now get_tuple should return from underlying saver
        result = buffered.get_tuple(config)
        assert result.checkpoint["id"] == sample_checkpoint["id"]

    @pytest.mark.asyncio
    async def test_aget_tuple_with_buffered_and_persisted_data(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test aget_tuple correctly prioritizes buffered data over persisted."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist to underlying saver
        await simple_checkpoint_saver.aput(
            config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # Buffer a newer checkpoint
        checkpoint2 = Checkpoint(
            v=1,
            id="checkpoint_newer",
            ts="2024-01-02T00:00:00Z",
            channel_values={"updated": True},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        await buffered.aput(config, checkpoint2, sample_checkpoint_metadata, {})

        # Should return buffered (newer) checkpoint
        result = await buffered.aget_tuple(config)
        assert result.checkpoint["id"] == "checkpoint_newer"
        assert result.checkpoint["channel_values"]["updated"] is True


class TestBufferedCheckpointSaverLastConfig:
    """Tests for _last_config tracking."""

    def test_last_config_updated_on_put(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that _last_config is updated when put is called."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        assert buffered._last_config is None

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        assert buffered._last_config is not None
        assert buffered._last_config["configurable"]["thread_id"] == "thread1"
        assert buffered._last_config["configurable"]["checkpoint_id"] == sample_checkpoint["id"]

    def test_last_config_updated_on_each_put(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test that _last_config is updated on each put."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        checkpoint1 = Checkpoint(
            v=1,
            id="cp1",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        checkpoint2 = Checkpoint(
            v=1,
            id="cp2",
            ts="2024-01-01T00:01:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )

        buffered.put(config, checkpoint1, sample_checkpoint_metadata, {})
        assert buffered._last_config["configurable"]["checkpoint_id"] == "cp1"

        buffered.put(config, checkpoint2, sample_checkpoint_metadata, {})
        assert buffered._last_config["configurable"]["checkpoint_id"] == "cp2"

    def test_last_config_not_updated_on_put_writes(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that _last_config is NOT updated when put_writes is called."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        original_last_config = buffered._last_config

        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")

        # _last_config should not change
        assert buffered._last_config == original_last_config

    def test_last_config_preserved_after_flush(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that _last_config is preserved after flush (not cleared)."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        buffered.put(config, sample_checkpoint, sample_checkpoint_metadata, {})
        buffered.flush()

        # _last_config should still be set after flush
        assert buffered._last_config is not None
        assert buffered._last_config["configurable"]["checkpoint_id"] == sample_checkpoint["id"]


class TestBufferedCheckpointSaverPendingWritesWithUnderlyingData:
    """Tests for pending writes interaction with underlying saver data."""

    def test_buffered_writes_not_returned_when_checkpoint_from_underlying(
        self, simple_checkpoint_saver, sample_checkpoint, sample_checkpoint_metadata
    ):
        """Test that buffered writes are NOT returned when checkpoint comes from underlying saver."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        # Persist checkpoint to underlying saver
        simple_checkpoint_saver.put(config, sample_checkpoint, sample_checkpoint_metadata, {})

        # Buffer writes (but no buffered checkpoint)
        writes_config = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": sample_checkpoint["id"],
            }
        )
        buffered.put_writes(writes_config, [("ch1", "v1")], "task1")

        # get_tuple should return checkpoint from underlying saver
        # The buffered writes should NOT be included (they're only associated with buffered checkpoint)
        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == sample_checkpoint["id"]
        # Writes from buffer are NOT included when checkpoint is from underlying saver
        # This is because the _get_tuple_if_buffered returns None when there's no buffered checkpoint

    def test_writes_for_old_checkpoint_not_included_after_new_put(
        self, simple_checkpoint_saver, sample_checkpoint_metadata
    ):
        """Test that writes for old checkpoint are not included after putting new checkpoint."""
        buffered = BufferedCheckpointSaver(simple_checkpoint_saver)
        config = RunnableConfig(
            configurable={"thread_id": "thread1", "checkpoint_ns": "ns1"}
        )

        checkpoint1 = Checkpoint(
            v=1,
            id="cp1",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )

        # Put first checkpoint
        buffered.put(config, checkpoint1, sample_checkpoint_metadata, {})

        # Add writes for first checkpoint
        writes_config1 = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )
        buffered.put_writes(writes_config1, [("ch1", "old_value")], "task1")

        # Put second checkpoint (overwrites first)
        checkpoint2 = Checkpoint(
            v=1,
            id="cp2",
            ts="2024-01-01T00:01:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        config_with_parent = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        )
        buffered.put(config_with_parent, checkpoint2, sample_checkpoint_metadata, {})

        # Add writes for second checkpoint
        writes_config2 = RunnableConfig(
            configurable={
                "thread_id": "thread1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp2",
            }
        )
        buffered.put_writes(writes_config2, [("ch2", "new_value")], "task2")

        # get_tuple should return second checkpoint with only its writes
        result = buffered.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "cp2"
        # Only writes for cp2 should be included
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0] == ("task2", "ch2", "new_value")
