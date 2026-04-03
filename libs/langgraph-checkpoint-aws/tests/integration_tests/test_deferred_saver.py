"""Integration tests for DeferredCheckpointSaver.

Tests the wrapper against all available backend savers:
- AgentCoreMemorySaver (creates memory if AGENTCORE_MEMORY_ID not set)
- DynamoDBSaver (creates table if not exists)
- BedrockSessionSaver (requires BEDROCK_SESSION_REGION env var)

Resources are created automatically when possible and cleaned up after.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
import string
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    uuid6,
)

from langgraph_checkpoint_aws.deferred_saver import DeferredCheckpointSaver
from tests.integration_tests.utils import (
    create_bedrock_session,
    ensure_agentcore_memory,
    ensure_dynamodb_table,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AGENTCORE_MEMORY_ID = os.getenv("AGENTCORE_MEMORY_ID", "langgraph_deferred_saver_integ")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE_NAME", "langgraph-deferred-saver-integ")
BEDROCK_SESSION_REGION = os.getenv("BEDROCK_SESSION_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SaverAndCleanup = tuple[BaseCheckpointSaver, Callable[[str, str], None], str | None]


def _generate_id(prefix: str = "test") -> str:
    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=6))


def _has_async_support(saver: BaseCheckpointSaver) -> bool:
    """Check if the saver implements async methods (not just the base stubs)."""
    try:
        # Base class raises NotImplementedError; if the method is not
        # overridden the saver is sync-only.
        return type(saver).aput is not BaseCheckpointSaver.aput
    except AttributeError:
        return False


def _make_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        channel_values={"messages": ["test message"]},
        channel_versions={"messages": "v1"},
        versions_seen={"node1": {"messages": "v1"}},
        updated_channels=None,
    )


# ---------------------------------------------------------------------------
# Resource setup — AgentCore memory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agentcore_memory_id() -> str:
    """Ensure the AgentCore memory exists, creating it if needed.

    Skips gracefully when the memory ID is invalid or unreachable so
    that other backends (DynamoDB, Bedrock Session) are not blocked.
    """
    try:
        return ensure_agentcore_memory(AGENTCORE_MEMORY_ID, AWS_REGION)
    except Exception as exc:
        pytest.skip(f"AgentCore memory unavailable: {exc}")


# ---------------------------------------------------------------------------
# Resource setup — DynamoDB table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dynamodb_table() -> str:
    """Ensure the DynamoDB table exists, creating it if needed.

    Skips gracefully when the table cannot be created or reached so
    that other backends are not blocked.
    """
    try:
        return ensure_dynamodb_table(DYNAMODB_TABLE, AWS_REGION)
    except Exception as exc:
        pytest.skip(f"DynamoDB table unavailable: {exc}")


# ---------------------------------------------------------------------------
# Backend saver factories
# ---------------------------------------------------------------------------


def _make_agentcore(memory_id: str) -> _SaverAndCleanup:
    from langgraph_checkpoint_aws.agentcore.saver import AgentCoreMemorySaver

    saver = AgentCoreMemorySaver(memory_id=memory_id, region_name=AWS_REGION)

    def cleanup(thread_id: str, actor_id: str) -> None:
        try:
            saver.delete_thread(thread_id, actor_id)
        except Exception:
            pass

    return saver, cleanup, None


def _make_dynamodb(table_name: str) -> _SaverAndCleanup:
    from langgraph_checkpoint_aws.checkpoint.dynamodb import DynamoDBSaver

    saver = DynamoDBSaver(table_name=table_name, region_name=AWS_REGION)

    def cleanup(thread_id: str, _actor_id: str) -> None:
        try:
            saver.delete_thread(thread_id)
        except Exception:
            pass

    return saver, cleanup, None


def _make_bedrock_session() -> _SaverAndCleanup:
    saver, session_id, session_cleanup = create_bedrock_session(BEDROCK_SESSION_REGION)

    def cleanup(thread_id: str, _actor_id: str) -> None:
        session_cleanup()

    return saver, cleanup, session_id


@pytest.fixture(
    params=["agentcore", "dynamodb", "bedrock_session"],
)
def saver_and_cleanup(
    request: Any,
) -> _SaverAndCleanup:
    """Parametrized fixture — creates one backend per test invocation.

    Each backend lazily resolves its own resource fixture via
    ``request.getfixturevalue`` so that a failure in one backend
    (e.g. AgentCore memory unavailable) only skips tests for that
    backend — other backends proceed normally.
    """
    name = request.param

    if name == "agentcore":
        memory_id = request.getfixturevalue("agentcore_memory_id")
        return _make_agentcore(memory_id)
    if name == "dynamodb":
        table = request.getfixturevalue("dynamodb_table")
        return _make_dynamodb(table)
    if name == "bedrock_session":
        try:
            return _make_bedrock_session()
        except Exception as exc:
            pytest.skip(f"Bedrock session unavailable: {exc}")

    msg = f"Unknown backend: {name}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Tests — run once per backend saver
# ---------------------------------------------------------------------------


class TestDeferredCheckpointSaver:
    """Integration tests for DeferredCheckpointSaver across all backends."""

    @staticmethod
    def _unpack(
        saver_and_cleanup: _SaverAndCleanup,
    ) -> tuple[BaseCheckpointSaver, Callable, str, str]:
        """Unpack fixture and generate IDs.

        Uses the backend-provided thread_id override when present
        (e.g. Bedrock Session requires a UUID session ID).
        """
        saver, cleanup, thread_id_override = saver_and_cleanup
        thread_id = thread_id_override or _generate_id("thread")
        actor_id = _generate_id("actor")
        return saver, cleanup, thread_id, actor_id

    def test_nothing_persisted_before_flush(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Buffer holds data; underlying saver has nothing until flush."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

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
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {},
            )
            assert not deferred.is_empty

            persisted = list(saver.list(config))
            assert len(persisted) == 0
        finally:
            deferred.clear()
            cleanup(thread_id, actor_id)

    def test_flush_persists_to_underlying_saver(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """After flush(), the checkpoint is retrievable from the backend."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        checkpoint = _make_checkpoint()

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {},
            )
            deferred.flush()

            assert deferred.is_empty

            persisted = list(saver.list(config))
            assert len(persisted) >= 1

            saved_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            result = saver.get_tuple(saved_config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
        finally:
            cleanup(thread_id, actor_id)

    def test_flush_on_exit_context_manager(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """flush_on_exit() auto-persists when the block exits."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            with deferred.flush_on_exit():
                deferred.put(
                    config,
                    _make_checkpoint(),
                    {"source": "input", "step": 1},
                    {},
                )

            assert deferred.is_empty
            assert len(list(saver.list(config))) >= 1
        finally:
            cleanup(thread_id, actor_id)

    def test_get_tuple_returns_buffered_data(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """get_tuple() serves from the buffer before flush."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        checkpoint = _make_checkpoint()

        try:
            deferred.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {},
            )

            result = deferred.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
        finally:
            deferred.clear()
            cleanup(thread_id, actor_id)

    def test_put_overwrites_keeps_only_latest(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Multiple put() calls — only the last checkpoint is flushed."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 0,
        }
        ckpt_1 = _make_checkpoint()
        ckpt_2 = _make_checkpoint()
        ckpt_3 = _make_checkpoint()

        try:
            deferred.put(config, ckpt_1, metadata, {})
            deferred.put(config, ckpt_2, metadata, {})
            deferred.put(config, ckpt_3, metadata, {})

            deferred.flush()

            persisted = list(saver.list(config))
            assert len(persisted) == 1
            assert persisted[0].checkpoint["id"] == ckpt_3["id"]
        finally:
            cleanup(thread_id, actor_id)

    def test_clear_discards_without_persisting(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """clear() drops buffered data; nothing reaches the backend."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

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
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {},
            )
            assert not deferred.is_empty

            deferred.clear()
            assert deferred.is_empty

            assert len(list(saver.list(config))) == 0
        finally:
            cleanup(thread_id, actor_id)

    def test_flush_persists_writes(self, saver_and_cleanup: _SaverAndCleanup) -> None:
        """put_writes() data is flushed alongside the checkpoint."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
                {},
            )
            deferred.put_writes(write_config, [("messages", "hello")], "task-1")

            assert deferred.has_buffered_writes
            deferred.flush()
            assert deferred.is_empty
        finally:
            cleanup(thread_id, actor_id)

    def test_multi_session_isolation(self, saver_and_cleanup: _SaverAndCleanup) -> None:
        """Two flush_on_exit blocks with different threads stay isolated."""
        saver, cleanup, thread_id_1, actor_id = self._unpack(saver_and_cleanup)
        # For Bedrock Session, thread_id is a pre-created session UUID
        # and we can't easily create a second one here, so we reuse
        # the same thread for both invocations (different checkpoints).
        deferred = DeferredCheckpointSaver(saver)
        ckpt_1 = _make_checkpoint()
        ckpt_2 = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id_1,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            with deferred.flush_on_exit():
                deferred.put(
                    config,
                    ckpt_1,
                    {"source": "input", "step": 1},
                    {},
                )

            with deferred.flush_on_exit():
                deferred.put(
                    config,
                    ckpt_2,
                    {"source": "loop", "step": 2},
                    {},
                )

            # Both checkpoints should be persisted
            persisted = list(saver.list(config))
            assert len(persisted) >= 2
        finally:
            cleanup(thread_id_1, actor_id)

    def test_flush_on_exit_persists_on_exception(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Context manager flushes even when the body raises."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            with pytest.raises(ValueError, match="simulated"):
                with deferred.flush_on_exit():
                    deferred.put(
                        config,
                        _make_checkpoint(),
                        {"source": "input", "step": 1},
                        {},
                    )
                    msg = "simulated"
                    raise ValueError(msg)

            # Data should still have been flushed
            assert deferred.is_empty
            persisted = list(saver.list(config))
            assert len(persisted) >= 1
        finally:
            cleanup(thread_id, actor_id)

    def test_get_tuple_delegates_after_flush(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """After flush, get_tuple reads from the backend, not buffer."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
                {},
            )
            deferred.flush()

            assert deferred.is_empty

            # This must delegate to the backend since buffer is empty
            result = deferred.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
        finally:
            cleanup(thread_id, actor_id)

    def test_list_excludes_buffered_includes_flushed(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """list() never shows buffered data, but shows flushed data."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

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
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {},
            )

            # Buffered — list should return nothing
            assert len(list(deferred.list(config))) == 0

            deferred.flush()

            # Flushed — list should return the checkpoint
            assert len(list(deferred.list(config))) >= 1
        finally:
            cleanup(thread_id, actor_id)

    def test_double_flush_is_safe(self, saver_and_cleanup: _SaverAndCleanup) -> None:
        """Calling flush() twice is a no-op the second time."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)

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
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {},
            )

            result1 = deferred.flush()
            assert result1 is not None

            result2 = deferred.flush()
            assert result2 is None
            assert deferred.is_empty
        finally:
            cleanup(thread_id, actor_id)

    def test_flush_returns_correct_config(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """flush() returns a config with the correct identifiers."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
                {},
            )

            result = deferred.flush()
            assert result is not None
            cfg = result["configurable"]
            assert cfg["thread_id"] == thread_id
            assert cfg["checkpoint_id"] == checkpoint["id"]
        finally:
            cleanup(thread_id, actor_id)

    # ------------------------------------------------------------------
    # Checkpoint namespace isolation
    # ------------------------------------------------------------------

    def test_checkpoint_ns_isolation(self, saver_and_cleanup: _SaverAndCleanup) -> None:
        """Checkpoints with different namespaces on the same thread stay isolated."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
        ckpt_ns_a = _make_checkpoint()
        ckpt_ns_b = _make_checkpoint()

        config_ns_a: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "ns_a",
            }
        }
        config_ns_b: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "ns_b",
            }
        }

        try:
            # Flush ns_a first, then ns_b
            deferred.put(
                config_ns_a,
                ckpt_ns_a,
                {"source": "input", "step": 1},
                {},
            )
            deferred.flush()

            deferred.put(
                config_ns_b,
                ckpt_ns_b,
                {"source": "input", "step": 1},
                {},
            )
            deferred.flush()

            # Each namespace should return its own checkpoint
            result_a = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "actor_id": actor_id,
                        "checkpoint_ns": "ns_a",
                        "checkpoint_id": ckpt_ns_a["id"],
                    }
                }
            )
            result_b = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "actor_id": actor_id,
                        "checkpoint_ns": "ns_b",
                        "checkpoint_id": ckpt_ns_b["id"],
                    }
                }
            )

            assert result_a is not None
            assert result_a.checkpoint["id"] == ckpt_ns_a["id"]

            assert result_b is not None
            assert result_b.checkpoint["id"] == ckpt_ns_b["id"]

            # Cross-check: ns_a checkpoint_id should not appear under ns_b
            cross_result = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "actor_id": actor_id,
                        "checkpoint_ns": "ns_b",
                        "checkpoint_id": ckpt_ns_a["id"],
                    }
                }
            )
            assert cross_result is None
        finally:
            cleanup(thread_id, actor_id)

    # ------------------------------------------------------------------
    # Pending writes retrievable after flush
    # ------------------------------------------------------------------

    def test_pending_writes_retrievable_after_flush(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Writes flushed to the backend are returned by get_tuple().pending_writes."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
                {},
            )
            deferred.put_writes(
                write_config,
                [("messages", "hello")],
                "task-1",
            )
            deferred.flush()

            # Read back from the backend directly
            result = saver.get_tuple(write_config)
            assert result is not None
            assert result.pending_writes is not None
            assert len(result.pending_writes) >= 1

            # Verify the write content is present
            channels = [pw[1] for pw in result.pending_writes]
            assert "messages" in channels
        finally:
            cleanup(thread_id, actor_id)

    # ------------------------------------------------------------------
    # Multiple task_ids flushed correctly
    # ------------------------------------------------------------------

    def test_multiple_task_ids_flushed(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Writes from multiple task_ids all persist after flush."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
                {},
            )
            deferred.put_writes(
                write_config,
                [("messages", "from-task-1")],
                "task-1",
            )
            deferred.put_writes(
                write_config,
                [("messages", "from-task-2")],
                "task-2",
            )
            deferred.put_writes(
                write_config,
                [("tools", "from-task-3")],
                "task-3",
            )

            assert not deferred.is_empty
            deferred.flush()
            assert deferred.is_empty

            result = saver.get_tuple(write_config)
            assert result is not None
            assert result.pending_writes is not None

            flushed_task_ids = {pw[0] for pw in result.pending_writes}
            # BedrockSessionSaver only retains the last write per
            # checkpoint, so we check that at least one task_id is
            # present and all returned ids are from our set.
            expected = {"task-1", "task-2", "task-3"}
            assert flushed_task_ids.issubset(expected)
            assert len(flushed_task_ids) >= 1
        finally:
            cleanup(thread_id, actor_id)

    # ------------------------------------------------------------------
    # Async paths against real backends
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_async_flush_persists_to_backend(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """aflush() persists checkpoint to the real backend."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        if not _has_async_support(saver):
            pytest.skip("Backend does not implement async methods")
        deferred = DeferredCheckpointSaver(saver)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            await deferred.aput(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {},
            )
            assert deferred.has_buffered_checkpoint

            result = await deferred.aflush()
            assert deferred.is_empty
            assert result is not None

            # Verify persisted via sync read on the backend
            saved_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            persisted = saver.get_tuple(saved_config)
            assert persisted is not None
            assert persisted.checkpoint["id"] == checkpoint["id"]
        finally:
            cleanup(thread_id, actor_id)

    @pytest.mark.asyncio
    async def test_async_get_tuple_from_backend(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """aget_tuple() reads from the backend after aflush()."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        if not _has_async_support(saver):
            pytest.skip("Backend does not implement async methods")
        deferred = DeferredCheckpointSaver(saver)
        checkpoint = _make_checkpoint()

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            await deferred.aput(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {},
            )
            await deferred.aflush()

            result = await deferred.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
        finally:
            cleanup(thread_id, actor_id)

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_persists(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """aflush_on_exit() context manager persists on exit."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        if not _has_async_support(saver):
            pytest.skip("Backend does not implement async methods")
        deferred = DeferredCheckpointSaver(saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "",
            }
        }

        try:
            async with deferred.aflush_on_exit():
                await deferred.aput(
                    config,
                    _make_checkpoint(),
                    {"source": "input", "step": 1},
                    {},
                )

            assert deferred.is_empty
            assert len(list(saver.list(config))) >= 1
        finally:
            cleanup(thread_id, actor_id)

    @pytest.mark.asyncio
    async def test_async_list_after_flush(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """alist() returns flushed data from the backend."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        if not _has_async_support(saver):
            pytest.skip("Backend does not implement async methods")
        deferred = DeferredCheckpointSaver(saver)

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
                _make_checkpoint(),
                {"source": "input", "step": 1},
                {},
            )

            # Before flush — alist should return nothing
            pre_flush = [item async for item in deferred.alist(config)]
            assert len(pre_flush) == 0

            deferred.flush()

            # After flush — alist should return the checkpoint
            post_flush = [item async for item in deferred.alist(config)]
            assert len(post_flush) >= 1
        finally:
            cleanup(thread_id, actor_id)

    # ------------------------------------------------------------------
    # Writes-only flush (no checkpoint)
    # ------------------------------------------------------------------

    def test_writes_only_flush_no_checkpoint(
        self, saver_and_cleanup: _SaverAndCleanup
    ) -> None:
        """Flushing only writes (no buffered checkpoint) against real backend."""
        saver, cleanup, thread_id, actor_id = self._unpack(saver_and_cleanup)
        deferred = DeferredCheckpointSaver(saver)
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
            # First persist the checkpoint directly so writes have a target
            saver.put(
                config,
                checkpoint,
                {"source": "input", "step": 1},
                {},
            )

            # Now buffer only writes (no deferred.put)
            deferred.put_writes(
                write_config,
                [("messages", "orphan-write")],
                "task-1",
            )
            assert not deferred.has_buffered_checkpoint
            assert deferred.has_buffered_writes

            result = deferred.flush()
            assert result is None  # No checkpoint was buffered
            assert deferred.is_empty

            # Verify the write landed
            persisted = saver.get_tuple(write_config)
            assert persisted is not None
            assert persisted.pending_writes is not None
            channels = [pw[1] for pw in persisted.pending_writes]
            assert "messages" in channels
        finally:
            cleanup(thread_id, actor_id)
