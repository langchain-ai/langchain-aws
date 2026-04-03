"""Deferred checkpoint saver that defers persistence until explicit flush.

Wraps any ``BaseCheckpointSaver`` and defers ``put()`` / ``put_writes()``
calls in memory, reducing API calls from dozens per workflow to a single
batch on flush.  Designed for use cases (chatbots, single-turn agents) that
only need the final conversation state persisted for session continuity.

.. warning::
    Mid-workflow crash recovery is **not** available when using this wrapper.
    Buffered data is only persisted when ``flush()`` / ``aflush()`` is called
    or a ``flush_on_exit()`` / ``aflush_on_exit()`` context manager exits.

Example::

    saver = AgentCoreMemorySaver(memory_id, region_name="us-east-1")
    deferred = DeferredCheckpointSaver(saver)
    graph = create_react_agent(model, tools=tools, checkpointer=deferred)

    with deferred.flush_on_exit():
        response = graph.invoke({"messages": [user_input]}, config)
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, NamedTuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    V,
)

# ---------------------------------------------------------------------------
# Internal buffer entry types
# ---------------------------------------------------------------------------
# NamedTuple was chosen over the alternatives for these internal types:
#
#   - Plain tuple: works but positional-only access (entry[0]) hurts
#     readability in the flush/match logic.
#   - TypedDict: mutable dict at runtime with hash-table overhead and no
#     immutability guarantee — wrong fit for a lock-protected buffer where
#     entries must not be modified after creation.
#   - dataclass (mutable): allows accidental mutation of buffered state,
#     which would introduce subtle concurrency bugs.
#   - dataclass (frozen=True, slots=True): immutable and memory-efficient,
#     but cannot be unpacked with ``a, b, c = entry`` and has slightly
#     higher object-creation cost than a tuple.
#
# NamedTuple gives us named fields (``entry.config``), immutability,
# zero overhead vs plain tuples, and tuple-compatible unpacking — the
# best balance for simple internal containers guarded by a lock.


class _PendingCheckpoint(NamedTuple):
    """Captured arguments to ``BaseCheckpointSaver.put()``.

    Stored in the buffer so they can be replayed on flush.
    """

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    new_versions: ChannelVersions


class _PendingWrite(NamedTuple):
    """Captured arguments to ``BaseCheckpointSaver.put_writes()``.

    Stored in the buffer so they can be replayed on flush.
    """

    config: RunnableConfig
    writes: list[tuple[str, Any]]
    task_id: str
    task_path: str


class DeferredCheckpointSaver(BaseCheckpointSaver):
    """Checkpoint saver wrapper that buffers writes and flushes on demand.

    Intercepts every ``put()`` and ``put_writes()`` call from the LangGraph
    runtime, keeping only the latest checkpoint in memory and accumulating
    all writes.  Nothing is persisted until the user explicitly calls
    ``flush()`` / ``aflush()`` or uses the ``flush_on_exit()`` /
    ``aflush_on_exit()`` context managers.

    This reduces API overhead from *O(nodes)* calls per invocation down to a
    single batch, at the cost of mid-workflow fault tolerance.

    Args:
        saver: The underlying checkpoint saver to delegate persistence to.

    Raises:
        ValueError: If *saver* is itself a ``DeferredCheckpointSaver``
            (nesting is not supported).

    Example::

        deferred = DeferredCheckpointSaver(my_saver)
        graph = workflow.compile(checkpointer=buffered)

        with buffered.flush_on_exit():
            graph.invoke(input, config)
    """

    def __init__(self, saver: BaseCheckpointSaver) -> None:
        if isinstance(saver, DeferredCheckpointSaver):
            msg = (
                "Nesting DeferredCheckpointSaver is not supported. "
                "Pass the underlying saver directly."
            )
            raise ValueError(msg)
        super().__init__()
        self._saver = saver
        self._lock = threading.Lock()
        self._pending_checkpoint: _PendingCheckpoint | None = None
        self._pending_writes: list[_PendingWrite] = []
        self._last_config: RunnableConfig | None = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def saver(self) -> BaseCheckpointSaver:
        """The wrapped checkpoint saver."""
        return self._saver

    @property
    def has_buffered_checkpoint(self) -> bool:
        """Whether a checkpoint is currently buffered."""
        with self._lock:
            return self._pending_checkpoint is not None

    @property
    def has_buffered_writes(self) -> bool:
        """Whether any writes are currently buffered."""
        with self._lock:
            return len(self._pending_writes) > 0

    @property
    def is_empty(self) -> bool:
        """Whether the buffer contains no pending data."""
        with self._lock:
            return self._pending_checkpoint is None and len(self._pending_writes) == 0

    # ------------------------------------------------------------------
    # BaseCheckpointSaver interface — buffered writes
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Buffer a checkpoint instead of persisting immediately.

        Each call overwrites the previously buffered checkpoint — only the
        latest one is kept.  This is the core optimization: for a 6-node
        workflow LangGraph calls ``put()`` 6 times, but we only persist once.

        Args:
            config: The runnable config associated with this checkpoint.
            checkpoint: The checkpoint data to buffer.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel version information.

        Returns:
            A config pointing to the buffered checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        with self._lock:
            self._pending_checkpoint = _PendingCheckpoint(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                new_versions=new_versions,
            )
            self._last_config = result_config
        return result_config

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of :meth:`put`. Delegates to the sync implementation.

        Args:
            config: The runnable config associated with this checkpoint.
            checkpoint: The checkpoint data to buffer.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel version information.

        Returns:
            A config pointing to the buffered checkpoint.
        """
        return self.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Buffer writes instead of persisting immediately.

        All writes accumulate in the buffer until flushed.

        Args:
            config: The runnable config for this write batch.
            writes: Sequence of ``(channel, value)`` pairs.
            task_id: Identifier of the task that produced these writes.
            task_path: Path of the task within the graph.
        """
        with self._lock:
            self._pending_writes.append(
                _PendingWrite(
                    config=config,
                    writes=list(writes),
                    task_id=task_id,
                    task_path=task_path,
                )
            )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of :meth:`put_writes`. Delegates to sync implementation.

        Args:
            config: The runnable config for this write batch.
            writes: Sequence of ``(channel, value)`` pairs.
            task_id: Identifier of the task that produced these writes.
            task_path: Path of the task within the graph.
        """
        self.put_writes(config, writes, task_id, task_path)

    # ------------------------------------------------------------------
    # BaseCheckpointSaver interface — reads (buffer-first)
    # ------------------------------------------------------------------

    def _get_buffered_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return a ``CheckpointTuple`` from the buffer if it matches *config*.

        Must be called while ``self._lock`` is held.

        Args:
            config: The config to match against the buffered checkpoint.

        Returns:
            A ``CheckpointTuple`` built from buffered data, or ``None`` if
            the buffer doesn't match.
        """
        if self._pending_checkpoint is None:
            return None

        buf_config = self._pending_checkpoint.config
        buf_checkpoint = self._pending_checkpoint.checkpoint
        buf_metadata = self._pending_checkpoint.metadata

        # Extract identifiers from the request
        req_thread = config["configurable"].get("thread_id")
        req_ns = config["configurable"].get("checkpoint_ns", "")
        req_ckpt_id = config["configurable"].get("checkpoint_id")

        # Extract identifiers from the buffer
        buf_thread = buf_config["configurable"].get("thread_id")
        buf_ns = buf_config["configurable"].get("checkpoint_ns", "")
        buf_ckpt_id = buf_checkpoint["id"]

        # Match thread_id and checkpoint_ns
        if req_thread != buf_thread or req_ns != buf_ns:
            return None

        # If a specific checkpoint_id was requested, it must match
        if req_ckpt_id is not None and req_ckpt_id != buf_ckpt_id:
            return None

        # Collect matching pending writes
        pending_writes: list[tuple[str, str, Any]] = []
        for entry in self._pending_writes:
            w_thread = entry.config["configurable"].get("thread_id")
            w_ns = entry.config["configurable"].get("checkpoint_ns", "")
            w_ckpt_id = entry.config["configurable"].get("checkpoint_id")
            if w_thread == buf_thread and w_ns == buf_ns and w_ckpt_id == buf_ckpt_id:
                for channel, value in entry.writes:
                    pending_writes.append((entry.task_id, channel, value))

        # Build parent config if available
        parent_config: RunnableConfig | None = None
        parent_id = (
            buf_metadata.get("parents", {}).get(buf_ns) if buf_metadata else None
        )
        if parent_id is None and buf_metadata:
            parent_id = buf_config["configurable"].get("checkpoint_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": buf_thread,
                    "checkpoint_ns": buf_ns,
                    "checkpoint_id": parent_id,
                }
            }

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": buf_thread,
                "checkpoint_ns": buf_ns,
                "checkpoint_id": buf_ckpt_id,
            }
        }

        return CheckpointTuple(
            config=result_config,
            checkpoint=buf_checkpoint,
            metadata=buf_metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the checkpoint tuple, checking the buffer first.

        If the requested ``thread_id`` + ``checkpoint_ns`` match the buffered
        checkpoint, the result is returned from memory with zero API calls.
        Otherwise the call is delegated to the underlying saver.

        Args:
            config: The runnable config identifying the checkpoint.

        Returns:
            The matching ``CheckpointTuple``, or ``None``.
        """
        with self._lock:
            buffered = self._get_buffered_tuple(config)
            if buffered is not None:
                return buffered
        return self._saver.get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of :meth:`get_tuple`.

        Args:
            config: The runnable config identifying the checkpoint.

        Returns:
            The matching ``CheckpointTuple``, or ``None``.
        """
        with self._lock:
            buffered = self._get_buffered_tuple(config)
            if buffered is not None:
                return buffered
        return await self._saver.aget_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the underlying saver.

        Buffered data is **not** included — only persisted checkpoints appear.

        Args:
            config: Optional config to scope the listing.
            filter: Optional filter criteria.
            before: Only return checkpoints before this config.
            limit: Maximum number of results.

        Yields:
            ``CheckpointTuple`` instances from the underlying saver.
        """
        yield from self._saver.list(config, filter=filter, before=before, limit=limit)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of :meth:`list`.

        Args:
            config: Optional config to scope the listing.
            filter: Optional filter criteria.
            before: Only return checkpoints before this config.
            limit: Maximum number of results.

        Yields:
            ``CheckpointTuple`` instances from the underlying saver.
        """
        async for item in self._saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    def get_next_version(self, current: V | None, channel: None) -> V:
        """Delegate version generation to the underlying saver.

        Args:
            current: The current version identifier.
            channel: Deprecated, kept for backwards compatibility.

        Returns:
            The next version identifier.
        """
        return self._saver.get_next_version(current, channel)

    # ------------------------------------------------------------------
    # Flush mechanics
    # ------------------------------------------------------------------

    def flush(self) -> RunnableConfig | None:
        """Persist all buffered data to the underlying saver.

        Flushes the buffered checkpoint first, then all accumulated writes.
        The lock is never held during network I/O.

        On checkpoint flush failure the checkpoint is restored to the buffer
        and the exception is re-raised.  On write flush failure, successfully
        flushed writes are removed; remaining writes stay in the buffer.

        Returns:
            The config returned by the underlying saver's ``put()``, or
            ``None`` if no checkpoint was buffered.

        Raises:
            Exception: Any exception raised by the underlying saver during
                persistence is propagated after restoring unflushed data.
        """
        result_config: RunnableConfig | None = None

        # --- Flush checkpoint ---
        # We clear _pending_checkpoint before I/O to prevent a concurrent
        # flush() from double-persisting the same checkpoint.  If the I/O
        # fails, we restore it — but only when no new put() has written a
        # newer checkpoint while we were blocked on the network call.
        # Without this guard, a stale checkpoint would overwrite the fresh
        # one, silently dropping data.
        with self._lock:
            pending_cp = self._pending_checkpoint
            self._pending_checkpoint = None

        if pending_cp is not None:
            try:
                result_config = self._saver.put(
                    pending_cp.config,
                    pending_cp.checkpoint,
                    pending_cp.metadata,
                    pending_cp.new_versions,
                )
            except Exception:
                with self._lock:
                    if self._pending_checkpoint is None:
                        self._pending_checkpoint = pending_cp
                raise

        # --- Flush writes one at a time ---
        # Unlike the checkpoint, writes use a peek-then-pop pattern: we
        # read _pending_writes[0] but leave it in the list during I/O.
        # On success we pop it; on failure it stays at index 0 so the
        # next flush() retries it.  No restore logic is needed because
        # the failed entry was never removed.  New put_writes() calls
        # append to the tail, so they never interfere with the drain.
        while True:
            with self._lock:
                if not self._pending_writes:
                    break
                write_entry = self._pending_writes[0]

            try:
                self._saver.put_writes(
                    write_entry.config,
                    write_entry.writes,
                    write_entry.task_id,
                    write_entry.task_path,
                )
            except Exception:
                raise

            with self._lock:
                if self._pending_writes and self._pending_writes[0] is write_entry:
                    self._pending_writes.pop(0)

        return result_config

    async def aflush(self) -> RunnableConfig | None:
        """Async version of :meth:`flush`.

        Returns:
            The config returned by the underlying saver's ``aput()``, or
            ``None`` if no checkpoint was buffered.

        Raises:
            Exception: Any exception raised by the underlying saver during
                persistence is propagated after restoring unflushed data.
        """
        result_config: RunnableConfig | None = None

        # --- Flush checkpoint ---
        # Same race-condition-safe pattern as flush(). See comments there.
        with self._lock:
            pending_cp = self._pending_checkpoint
            self._pending_checkpoint = None

        if pending_cp is not None:
            try:
                result_config = await self._saver.aput(
                    pending_cp.config,
                    pending_cp.checkpoint,
                    pending_cp.metadata,
                    pending_cp.new_versions,
                )
            except Exception:
                with self._lock:
                    if self._pending_checkpoint is None:
                        self._pending_checkpoint = pending_cp
                raise

        # --- Flush writes one at a time ---
        # Same peek-then-pop pattern as flush(). See comments there.
        while True:
            with self._lock:
                if not self._pending_writes:
                    break
                write_entry = self._pending_writes[0]

            try:
                await self._saver.aput_writes(
                    write_entry.config,
                    write_entry.writes,
                    write_entry.task_id,
                    write_entry.task_path,
                )
            except Exception:
                raise

            with self._lock:
                if self._pending_writes and self._pending_writes[0] is write_entry:
                    self._pending_writes.pop(0)

        return result_config

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def flush_on_exit(self) -> Iterator[DeferredCheckpointSaver]:
        """Context manager that flushes buffered data on exit.

        Flushes in the ``finally`` block so data is persisted even when the
        graph raises an exception.

        Yields:
            This ``DeferredCheckpointSaver`` instance.

        Example::

            with deferred.flush_on_exit():
                graph.invoke(input, config)
        """
        try:
            yield self
        finally:
            self.flush()

    @contextlib.asynccontextmanager
    async def aflush_on_exit(self) -> AsyncIterator[DeferredCheckpointSaver]:
        """Async context manager that flushes buffered data on exit.

        Yields:
            This ``DeferredCheckpointSaver`` instance.

        Example::

            async with deferred.aflush_on_exit():
                await graph.ainvoke(input, config)
        """
        try:
            yield self
        finally:
            await self.aflush()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all buffered data without persisting.

        This is useful when you want to abandon an in-progress workflow
        without saving any intermediate state.
        """
        with self._lock:
            self._pending_checkpoint = None
            self._pending_writes.clear()
            self._last_config = None
