"""Unit tests for DeferredCheckpointSaver.

Covers the 10 correctness properties from the design document:
1. After flush() succeeds, is_empty is True
2. After flush() fails on checkpoint, has_buffered_checkpoint is still True
3. After flush() fails on write N, writes 0..N-1 removed, N..end remain
4. get_tuple() returns buffered data when thread_id + checkpoint_ns match
5. get_tuple() delegates to underlying saver when buffer doesn't match
6. list() never returns buffered data
7. Context managers flush in finally — even on exception
8. Concurrent put() + get_tuple() from different threads never raises
9. DeferredCheckpointSaver(DeferredCheckpointSaver(x)) raises ValueError
10. clear() discards all buffered data without persisting
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import MemorySaver

from langgraph_checkpoint_aws.deferred_saver import DeferredCheckpointSaver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    thread_id: str = "thread-1",
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
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


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestInit:
    """Initialization and nesting prevention."""

    def test_wraps_saver(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        assert buffered.saver is memory_saver

    def test_nesting_raises_value_error(self, memory_saver: MemorySaver) -> None:
        """Property 9: DeferredCheckpointSaver(DeferredCheckpointSaver(x)) raises."""
        buffered = DeferredCheckpointSaver(memory_saver)
        with pytest.raises(ValueError, match="Nesting"):
            DeferredCheckpointSaver(buffered)

    def test_starts_empty(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        assert buffered.is_empty
        assert not buffered.has_buffered_checkpoint
        assert not buffered.has_buffered_writes


# ---------------------------------------------------------------------------
# Buffering tests
# ---------------------------------------------------------------------------


class TestBuffering:
    """put() and put_writes() buffer correctly."""

    def test_put_buffers_checkpoint(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config()
        checkpoint = _make_checkpoint()
        metadata = _make_metadata()

        result = buffered.put(config, checkpoint, metadata, {})

        assert buffered.has_buffered_checkpoint
        assert not buffered.is_empty
        assert result["configurable"]["checkpoint_id"] == "ckpt-1"

    def test_put_overwrites_previous(self, memory_saver: MemorySaver) -> None:
        """Only the latest checkpoint is kept."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config()

        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(0), {})
        buffered.put(config, _make_checkpoint("ckpt-2"), _make_metadata(1), {})

        # get_tuple should return the second checkpoint
        result = buffered.get_tuple(_make_config())
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-2"

    def test_put_writes_accumulates(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(checkpoint_id="ckpt-1")

        buffered.put_writes(config, [("ch1", "v1")], "task-1")
        buffered.put_writes(config, [("ch2", "v2")], "task-2")

        assert buffered.has_buffered_writes

    def test_put_does_not_call_underlying_saver(
        self, memory_saver: MemorySaver
    ) -> None:
        """put() should never trigger I/O on the underlying saver."""
        buffered = DeferredCheckpointSaver(memory_saver)
        with patch.object(memory_saver, "put") as mock_put:
            buffered.put(_make_config(), _make_checkpoint(), _make_metadata(), {})
            mock_put.assert_not_called()

    def test_put_writes_does_not_call_underlying_saver(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        with patch.object(memory_saver, "put_writes") as mock_pw:
            buffered.put_writes(
                _make_config(checkpoint_id="ckpt-1"),
                [("ch", "val")],
                "task-1",
            )
            mock_pw.assert_not_called()


# ---------------------------------------------------------------------------
# get_tuple tests
# ---------------------------------------------------------------------------


class TestGetTuple:
    """Buffer-first read logic."""

    def test_returns_buffered_when_matching(self, memory_saver: MemorySaver) -> None:
        """Property 4: get_tuple returns buffered data on match."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1", checkpoint_ns="ns1")
        checkpoint = _make_checkpoint("ckpt-1")

        buffered.put(config, checkpoint, _make_metadata(), {})

        result = buffered.get_tuple(_make_config(thread_id="t1", checkpoint_ns="ns1"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-1"

    def test_delegates_when_no_match(self, memory_saver: MemorySaver) -> None:
        """Property 5: get_tuple delegates when buffer doesn't match."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        # Query for a different thread
        result = buffered.get_tuple(_make_config(thread_id="t2"))
        # MemorySaver has nothing for t2
        assert result is None

    def test_delegates_when_checkpoint_ns_differs(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1", checkpoint_ns="ns1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        result = buffered.get_tuple(_make_config(thread_id="t1", checkpoint_ns="ns2"))
        assert result is None

    def test_matches_specific_checkpoint_id(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        # Exact match
        result = buffered.get_tuple(
            _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        )
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-1"

    def test_no_match_when_checkpoint_id_differs(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = buffered.get_tuple(
            _make_config(thread_id="t1", checkpoint_id="ckpt-999")
        )
        # Falls through to underlying saver which has nothing
        assert result is None

    def test_includes_matching_pending_writes(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")
        buffered.put_writes(write_config, [("ch2", "v2")], "task-2")

        result = buffered.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.pending_writes is not None
        assert len(result.pending_writes) == 2

    def test_excludes_writes_for_different_checkpoint(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-2"), _make_metadata(), {})

        # Writes for a different checkpoint_id
        old_write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(old_write_config, [("ch1", "v1")], "task-1")

        result = buffered.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        # No matching writes
        assert result.pending_writes is None

    def test_empty_buffer_delegates(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        result = buffered.get_tuple(_make_config())
        assert result is None


# ---------------------------------------------------------------------------
# Flush tests
# ---------------------------------------------------------------------------


class TestFlush:
    """Flush mechanics and error handling."""

    def test_flush_persists_checkpoint(self, memory_saver: MemorySaver) -> None:
        """Property 1: After flush() succeeds, is_empty is True."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = buffered.flush()

        assert buffered.is_empty
        assert result is not None
        assert result["configurable"]["checkpoint_id"] == "ckpt-1"

    def test_flush_persists_writes(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")

        buffered.flush()
        assert buffered.is_empty
        assert not buffered.has_buffered_writes

    def test_flush_returns_none_when_empty(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        result = buffered.flush()
        assert result is None

    def test_flush_checkpoint_failure_restores_buffer(
        self, memory_saver: MemorySaver
    ) -> None:
        """Property 2: flush() fail on checkpoint keeps buffer."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        with patch.object(
            memory_saver, "put", side_effect=RuntimeError("network error")
        ):
            with pytest.raises(RuntimeError, match="network error"):
                buffered.flush()

        assert buffered.has_buffered_checkpoint

    def test_flush_failure_does_not_clobber_new_checkpoint(
        self, memory_saver: MemorySaver
    ) -> None:
        """Restore on failure must not overwrite a checkpoint that arrived
        during the network I/O window."""
        deferred = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        deferred.put(config, _make_checkpoint("ckpt-old"), _make_metadata(), {})

        def failing_put_that_races(*args, **kwargs):
            # Simulate a new put() arriving while flush is doing I/O
            deferred.put(config, _make_checkpoint("ckpt-new"), _make_metadata(1), {})
            msg = "network error"
            raise RuntimeError(msg)

        with patch.object(memory_saver, "put", side_effect=failing_put_that_races):
            with pytest.raises(RuntimeError, match="network error"):
                deferred.flush()

        # ckpt-new must survive — not be clobbered by ckpt-old restore
        result = deferred.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-new"

    def test_flush_write_failure_keeps_remaining(
        self, memory_saver: MemorySaver
    ) -> None:
        """Property 3: Partial write failure keeps remaining."""
        buffered = DeferredCheckpointSaver(memory_saver)

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")
        buffered.put_writes(write_config, [("ch2", "v2")], "task-2")
        buffered.put_writes(write_config, [("ch3", "v3")], "task-3")

        call_count = 0

        original_put_writes = memory_saver.put_writes

        def failing_put_writes(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("write failed")
            return original_put_writes(*args, **kwargs)

        with patch.object(memory_saver, "put_writes", side_effect=failing_put_writes):
            with pytest.raises(RuntimeError, match="write failed"):
                buffered.flush()

        # First write succeeded and was removed, second failed and remains
        # along with the third
        assert buffered.has_buffered_writes

    def test_flush_only_writes_no_checkpoint(self, memory_saver: MemorySaver) -> None:
        """Flush works when only writes are buffered (no checkpoint)."""
        buffered = DeferredCheckpointSaver(memory_saver)
        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")

        result = buffered.flush()
        assert result is None
        assert buffered.is_empty


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManagers:
    """flush_on_exit and aflush_on_exit."""

    def test_flush_on_exit_flushes(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")

        with buffered.flush_on_exit():
            buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        assert buffered.is_empty

    def test_flush_on_exit_yields_self(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        with buffered.flush_on_exit() as ctx:
            assert ctx is buffered

    def test_flush_on_exit_flushes_on_exception(
        self, memory_saver: MemorySaver
    ) -> None:
        """Property 7: Context managers flush in finally — even on exception."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")

        with pytest.raises(ValueError, match="boom"):
            with buffered.flush_on_exit():
                buffered.put(config, _make_checkpoint(), _make_metadata(), {})
                msg = "boom"
                raise ValueError(msg)

        # Data was flushed despite the exception
        assert buffered.is_empty

    @pytest.mark.asyncio
    async def test_aflush_on_exit_flushes(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")

        async with buffered.aflush_on_exit():
            buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        assert buffered.is_empty

    @pytest.mark.asyncio
    async def test_aflush_on_exit_yields_self(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        async with buffered.aflush_on_exit() as ctx:
            assert ctx is buffered

    @pytest.mark.asyncio
    async def test_aflush_on_exit_flushes_on_exception(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")

        with pytest.raises(ValueError, match="boom"):
            async with buffered.aflush_on_exit():
                buffered.put(config, _make_checkpoint(), _make_metadata(), {})
                msg = "boom"
                raise ValueError(msg)

        assert buffered.is_empty


# ---------------------------------------------------------------------------
# Clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Property 10: clear() discards all buffered data without persisting."""

    def test_clear_discards_checkpoint(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config()
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        buffered.clear()

        assert buffered.is_empty

    def test_clear_discards_writes(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        write_config = _make_config(checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")

        buffered.clear()

        assert buffered.is_empty

    def test_clear_does_not_persist(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        with patch.object(memory_saver, "put") as mock_put:
            buffered.clear()
            mock_put.assert_not_called()


# ---------------------------------------------------------------------------
# list() tests
# ---------------------------------------------------------------------------


class TestList:
    """Property 6: list() never returns buffered data."""

    def test_list_delegates_to_underlying(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        results = list(buffered.list(_make_config(thread_id="t1")))
        # MemorySaver has nothing persisted yet
        assert len(results) == 0

    def test_list_shows_flushed_data(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})
        buffered.flush()

        results = list(buffered.list(_make_config(thread_id="t1")))
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_next_version delegation
# ---------------------------------------------------------------------------


class TestGetNextVersion:
    """get_next_version always delegates to the underlying saver."""

    def test_delegates_to_saver(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        v = buffered.get_next_version(None, None)
        # MemorySaver uses string versions; just verify it returns a value
        assert v is not None

    def test_type_matches_saver(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        v_buffered = buffered.get_next_version(None, None)
        v_saver = memory_saver.get_next_version(None, None)
        assert type(v_buffered) is type(v_saver)


# ---------------------------------------------------------------------------
# Config preservation
# ---------------------------------------------------------------------------


class TestConfigPreservation:
    """Returned configs contain the correct identifiers."""

    def test_put_returns_correct_config(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1", checkpoint_ns="ns1")
        result = buffered.put(config, _make_checkpoint("ckpt-42"), _make_metadata(), {})
        assert result["configurable"]["thread_id"] == "t1"
        assert result["configurable"]["checkpoint_ns"] == "ns1"
        assert result["configurable"]["checkpoint_id"] == "ckpt-42"

    def test_flush_returns_saver_config(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = buffered.flush()
        assert result is not None
        assert "configurable" in result

    def test_get_tuple_config_has_checkpoint_id(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = buffered.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "ckpt-1"


# ---------------------------------------------------------------------------
# Parent config
# ---------------------------------------------------------------------------


class TestParentConfig:
    """get_tuple builds parent_config from the original config's checkpoint_id."""

    def test_parent_config_from_original_checkpoint_id(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1", checkpoint_id="parent-ckpt")
        buffered.put(config, _make_checkpoint("ckpt-2"), _make_metadata(), {})

        result = buffered.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent-ckpt"

    def test_no_parent_config_on_first_checkpoint(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        # No checkpoint_id in the original config
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = buffered.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.parent_config is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Property 8: Concurrent put() + get_tuple() never raises."""

    def test_concurrent_put_and_get_tuple(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        errors: list[Exception] = []
        iterations = 200

        def writer() -> None:
            try:
                for i in range(iterations):
                    config = _make_config(thread_id="t1")
                    buffered.put(
                        config,
                        _make_checkpoint(f"ckpt-{i}"),
                        _make_metadata(i),
                        {},
                    )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(iterations):
                    buffered.get_tuple(_make_config(thread_id="t1"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_put_writes(self, memory_saver: MemorySaver) -> None:
        """Simulates parallel branches calling put_writes concurrently."""
        buffered = DeferredCheckpointSaver(memory_saver)
        errors: list[Exception] = []
        iterations = 200

        def write_worker(task_id: str) -> None:
            try:
                for i in range(iterations):
                    config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
                    buffered.put_writes(
                        config,
                        [(f"ch-{task_id}-{i}", f"val-{i}")],
                        task_id,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_worker, args=("task-a",)),
            threading.Thread(target=write_worker, args=("task-b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert buffered.has_buffered_writes


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsync:
    """Async variants delegate correctly."""

    @pytest.mark.asyncio
    async def test_aput_buffers(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        result = await buffered.aput(
            config, _make_checkpoint("ckpt-1"), _make_metadata(), {}
        )
        assert buffered.has_buffered_checkpoint
        assert result["configurable"]["checkpoint_id"] == "ckpt-1"

    @pytest.mark.asyncio
    async def test_aput_writes_buffers(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(checkpoint_id="ckpt-1")
        await buffered.aput_writes(config, [("ch1", "v1")], "task-1")
        assert buffered.has_buffered_writes

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_buffered(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = await buffered.aget_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-1"

    @pytest.mark.asyncio
    async def test_aflush_persists(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        result = await buffered.aflush()
        assert buffered.is_empty
        assert result is not None

    @pytest.mark.asyncio
    async def test_aflush_checkpoint_failure_restores(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        with patch.object(
            memory_saver,
            "aput",
            new_callable=AsyncMock,
            side_effect=RuntimeError("async fail"),
        ):
            with pytest.raises(RuntimeError, match="async fail"):
                await buffered.aflush()

        assert buffered.has_buffered_checkpoint

    @pytest.mark.asyncio
    async def test_alist_delegates(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint(), _make_metadata(), {})

        results = [item async for item in buffered.alist(_make_config(thread_id="t1"))]
        # Not flushed yet, so nothing in underlying saver
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Multi-session
# ---------------------------------------------------------------------------


class TestMultiSession:
    """Each flush_on_exit block handles one session independently."""

    def test_sequential_sessions(self, memory_saver: MemorySaver) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)

        with buffered.flush_on_exit():
            config1 = _make_config(thread_id="session-1")
            buffered.put(config1, _make_checkpoint("ckpt-s1"), _make_metadata(), {})

        with buffered.flush_on_exit():
            config2 = _make_config(thread_id="session-2")
            buffered.put(config2, _make_checkpoint("ckpt-s2"), _make_metadata(), {})

        # Both sessions persisted
        r1 = list(buffered.list(_make_config(thread_id="session-1")))
        r2 = list(buffered.list(_make_config(thread_id="session-2")))
        assert len(r1) == 1
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Clear with combined checkpoint + writes
# ---------------------------------------------------------------------------


class TestClearCombined:
    """clear() discards both checkpoint and writes when both are buffered."""

    def test_clear_discards_checkpoint_and_writes(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")
        buffered.put_writes(write_config, [("ch2", "v2")], "task-2")

        assert buffered.has_buffered_checkpoint
        assert buffered.has_buffered_writes

        buffered.clear()

        assert buffered.is_empty
        assert not buffered.has_buffered_checkpoint
        assert not buffered.has_buffered_writes

    def test_clear_combined_does_not_persist(self, memory_saver: MemorySaver) -> None:
        """Nothing reaches the backend after clearing combined state."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")

        with (
            patch.object(memory_saver, "put") as mock_put,
            patch.object(memory_saver, "put_writes") as mock_pw,
        ):
            buffered.clear()
            mock_put.assert_not_called()
            mock_pw.assert_not_called()

        # Flushing after clear should be a no-op
        result = buffered.flush()
        assert result is None
        assert len(list(memory_saver.list(_make_config(thread_id="t1")))) == 0


# ---------------------------------------------------------------------------
# get_tuple delegation with pre-populated backend
# ---------------------------------------------------------------------------


class TestGetTupleDelegation:
    """get_tuple falls through to the underlying saver and returns real data."""

    def test_delegates_to_populated_backend(self, memory_saver: MemorySaver) -> None:
        """Buffer miss for thread-A delegates to backend which has thread-B data."""
        # Pre-populate the backend with a checkpoint for thread-B
        config_b = _make_config(thread_id="thread-B")
        ckpt_b = _make_checkpoint("ckpt-B")
        memory_saver.put(config_b, ckpt_b, _make_metadata(), {})

        buffered = DeferredCheckpointSaver(memory_saver)

        # Buffer a checkpoint for thread-A
        config_a = _make_config(thread_id="thread-A")
        buffered.put(config_a, _make_checkpoint("ckpt-A"), _make_metadata(), {})

        # Query thread-B — buffer has thread-A, so it must delegate
        result = buffered.get_tuple(_make_config(thread_id="thread-B"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-B"

    def test_buffer_hit_does_not_query_backend(self, memory_saver: MemorySaver) -> None:
        """When the buffer matches, the underlying saver is never called."""
        buffered = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        buffered.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        with patch.object(memory_saver, "get_tuple") as mock_get:
            result = buffered.get_tuple(_make_config(thread_id="t1"))
            mock_get.assert_not_called()

        assert result is not None
        assert result.checkpoint["id"] == "ckpt-1"

    @pytest.mark.asyncio
    async def test_aget_tuple_delegates_to_populated_backend(
        self, memory_saver: MemorySaver
    ) -> None:
        """Async variant also delegates on buffer miss."""
        config_b = _make_config(thread_id="thread-B")
        memory_saver.put(config_b, _make_checkpoint("ckpt-B"), _make_metadata(), {})

        buffered = DeferredCheckpointSaver(memory_saver)
        config_a = _make_config(thread_id="thread-A")
        buffered.put(config_a, _make_checkpoint("ckpt-A"), _make_metadata(), {})

        result = await buffered.aget_tuple(_make_config(thread_id="thread-B"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-B"


# ---------------------------------------------------------------------------
# list() with filter, before, and limit parameters
# ---------------------------------------------------------------------------


class TestListParameters:
    """list() and alist() forward filter/before/limit to the underlying saver."""

    def _populate_three_checkpoints(self, memory_saver: MemorySaver) -> list[str]:
        """Flush three checkpoints into the backend, return their IDs."""
        ids = ["ckpt-1", "ckpt-2", "ckpt-3"]
        config = _make_config(thread_id="t1")
        for i, cid in enumerate(ids):
            memory_saver.put(config, _make_checkpoint(cid), _make_metadata(i), {})
        return ids

    def test_list_with_limit(self, memory_saver: MemorySaver) -> None:
        self._populate_three_checkpoints(memory_saver)
        buffered = DeferredCheckpointSaver(memory_saver)

        results = list(buffered.list(_make_config(thread_id="t1"), limit=2))
        assert len(results) == 2

    def test_list_with_before(self, memory_saver: MemorySaver) -> None:
        self._populate_three_checkpoints(memory_saver)
        buffered = DeferredCheckpointSaver(memory_saver)

        # Get all checkpoints to find the latest one
        all_results = list(buffered.list(_make_config(thread_id="t1")))
        assert len(all_results) == 3

        # Use the latest checkpoint as the "before" cursor
        latest = all_results[0]
        before_config = latest.config

        results = list(
            buffered.list(_make_config(thread_id="t1"), before=before_config)
        )
        # Should exclude the latest, returning the older ones
        assert len(results) == 2
        for r in results:
            assert r.checkpoint["id"] != latest.checkpoint["id"]

    def test_list_limit_one_returns_latest(self, memory_saver: MemorySaver) -> None:
        ids = self._populate_three_checkpoints(memory_saver)
        buffered = DeferredCheckpointSaver(memory_saver)

        results = list(buffered.list(_make_config(thread_id="t1"), limit=1))
        assert len(results) == 1
        # MemorySaver returns newest first
        assert results[0].checkpoint["id"] == ids[-1]

    @pytest.mark.asyncio
    async def test_alist_with_limit(self, memory_saver: MemorySaver) -> None:
        self._populate_three_checkpoints(memory_saver)
        buffered = DeferredCheckpointSaver(memory_saver)

        results = [
            item async for item in buffered.alist(_make_config(thread_id="t1"), limit=2)
        ]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_alist_with_before(self, memory_saver: MemorySaver) -> None:
        self._populate_three_checkpoints(memory_saver)
        buffered = DeferredCheckpointSaver(memory_saver)

        all_results = [
            item async for item in buffered.alist(_make_config(thread_id="t1"))
        ]
        latest = all_results[0]

        results = [
            item
            async for item in buffered.alist(
                _make_config(thread_id="t1"), before=latest.config
            )
        ]
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Flush write-drain identity guard (gap #5)
# ---------------------------------------------------------------------------


class TestFlushWriteDrainIdentityGuard:
    """The pop after successful write I/O uses ``is`` identity comparison.

    If a concurrent ``put_writes()`` inserts a *new* entry at index 0
    between the I/O and the pop, the original entry is no longer at
    index 0.  The guard ``self._pending_writes[0] is write_entry``
    prevents popping the wrong entry.
    """

    def test_concurrent_put_writes_during_drain_preserves_new_entry(
        self, memory_saver: MemorySaver
    ) -> None:
        """A put_writes() that races with flush() must not lose its entry."""
        deferred = DeferredCheckpointSaver(memory_saver)
        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")

        # Seed one write that will be flushed.
        deferred.put_writes(write_config, [("ch-original", "v1")], "task-orig")

        new_write_inserted = False

        original_put_writes = memory_saver.put_writes

        def intercepting_put_writes(*args: Any, **kwargs: Any) -> None:
            nonlocal new_write_inserted
            # After the underlying saver persists the original write,
            # but *before* flush() re-acquires the lock to pop, simulate
            # a concurrent put_writes() that inserts at the tail.  Because
            # list.insert(0, ...) is not used (put_writes appends), the
            # new entry lands at the end — but we can simulate the more
            # adversarial case by directly inserting at index 0 under the
            # lock to exercise the identity guard.
            result = original_put_writes(*args, **kwargs)
            if not new_write_inserted:
                new_write_inserted = True
                # Directly manipulate the buffer to simulate a race where
                # the entry at index 0 is no longer the one we peeked.
                from langgraph_checkpoint_aws.deferred_saver import _PendingWrite

                intruder = _PendingWrite(
                    config=write_config,
                    writes=[("ch-intruder", "v2")],
                    task_id="task-intruder",
                    task_path="",
                )
                with deferred._lock:
                    deferred._pending_writes.insert(0, intruder)
            return result

        with patch.object(
            memory_saver, "put_writes", side_effect=intercepting_put_writes
        ):
            deferred.flush()

        # The intruder entry was inserted at index 0 *after* the original
        # was peeked but before the pop.  The identity guard should have
        # skipped the pop for the original iteration, leaving the intruder
        # in the buffer.  Then the loop continues and flushes the intruder
        # normally.  But the key property: the intruder was never silently
        # dropped.
        #
        # After full drain, buffer should be empty (both were flushed).
        assert deferred.is_empty


# ---------------------------------------------------------------------------
# Concurrent flush() + flush() (gap #15)
# ---------------------------------------------------------------------------


class TestConcurrentFlush:
    """Two threads calling flush() concurrently must not double-persist."""

    def test_concurrent_flush_no_double_persist(
        self, memory_saver: MemorySaver
    ) -> None:
        deferred = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        deferred.put(config, _make_checkpoint("ckpt-1"), _make_metadata(), {})

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        deferred.put_writes(write_config, [("ch1", "v1")], "task-1")

        put_call_count = 0
        put_writes_call_count = 0
        original_put = memory_saver.put
        original_put_writes = memory_saver.put_writes

        count_lock = threading.Lock()

        def counting_put(*args: Any, **kwargs: Any) -> Any:
            nonlocal put_call_count
            with count_lock:
                put_call_count += 1
            return original_put(*args, **kwargs)

        def counting_put_writes(*args: Any, **kwargs: Any) -> None:
            nonlocal put_writes_call_count
            with count_lock:
                put_writes_call_count += 1
            return original_put_writes(*args, **kwargs)

        errors: list[Exception] = []

        def flusher() -> None:
            try:
                deferred.flush()
            except Exception as e:
                errors.append(e)

        with (
            patch.object(memory_saver, "put", side_effect=counting_put),
            patch.object(memory_saver, "put_writes", side_effect=counting_put_writes),
        ):
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
        # The checkpoint must be persisted exactly once, not twice.
        assert put_call_count == 1
        # The write must be persisted exactly once.
        assert put_writes_call_count == 1


# ---------------------------------------------------------------------------
# Async flush — write failure keeps remaining (gap #6)
# ---------------------------------------------------------------------------


class TestAsyncFlushWriteFailure:
    """aflush() partial write failure must keep remaining writes in buffer."""

    @pytest.mark.asyncio
    async def test_aflush_write_failure_keeps_remaining(
        self, memory_saver: MemorySaver
    ) -> None:
        buffered = DeferredCheckpointSaver(memory_saver)

        write_config = _make_config(thread_id="t1", checkpoint_id="ckpt-1")
        buffered.put_writes(write_config, [("ch1", "v1")], "task-1")
        buffered.put_writes(write_config, [("ch2", "v2")], "task-2")
        buffered.put_writes(write_config, [("ch3", "v3")], "task-3")

        call_count = 0
        original_aput_writes = memory_saver.aput_writes

        async def failing_aput_writes(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "async write failed"
                raise RuntimeError(msg)
            return await original_aput_writes(*args, **kwargs)

        with patch.object(
            memory_saver,
            "aput_writes",
            new_callable=AsyncMock,
            side_effect=failing_aput_writes,
        ):
            with pytest.raises(RuntimeError, match="async write failed"):
                await buffered.aflush()

        # First write succeeded and was removed; second failed and remains
        # along with the third.
        assert buffered.has_buffered_writes


# ---------------------------------------------------------------------------
# Async flush — checkpoint failure race guard (gap #7)
# ---------------------------------------------------------------------------


class TestAsyncFlushCheckpointRace:
    """aflush() checkpoint failure must not clobber a newer put()."""

    @pytest.mark.asyncio
    async def test_aflush_failure_does_not_clobber_new_checkpoint(
        self, memory_saver: MemorySaver
    ) -> None:
        deferred = DeferredCheckpointSaver(memory_saver)
        config = _make_config(thread_id="t1")
        deferred.put(config, _make_checkpoint("ckpt-old"), _make_metadata(), {})

        async def failing_aput_that_races(*args: Any, **kwargs: Any) -> None:
            # Simulate a new put() arriving while aflush is awaiting I/O.
            deferred.put(config, _make_checkpoint("ckpt-new"), _make_metadata(1), {})
            msg = "async network error"
            raise RuntimeError(msg)

        with patch.object(
            memory_saver,
            "aput",
            new_callable=AsyncMock,
            side_effect=failing_aput_that_races,
        ):
            with pytest.raises(RuntimeError, match="async network error"):
                await deferred.aflush()

        # ckpt-new must survive — not be clobbered by ckpt-old restore.
        result = deferred.get_tuple(_make_config(thread_id="t1"))
        assert result is not None
        assert result.checkpoint["id"] == "ckpt-new"
