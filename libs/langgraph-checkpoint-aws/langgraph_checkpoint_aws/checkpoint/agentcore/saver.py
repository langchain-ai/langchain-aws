"""
AgentCore Memory Checkpoint Saver implementation.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextvars import ContextVar
from typing import Any, TypeAlias, cast

from langchain_core.runnables import RunnableConfig, run_in_executor
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from langgraph_checkpoint_aws.checkpoint.deferred_saver import PendingWrite

from .constants import (
    EMPTY_CHANNEL_VALUE,
    InvalidConfigError,
)
from .helpers import (
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_BACKOFF,
    DEFAULT_MAX_RETRIES,
    AgentCoreEventClient,
    EventProcessor,
    EventSerializer,
)
from .models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)

RunnableConfigDict: TypeAlias = dict[str, Any]

# Request-scoped (thread/async-task isolated) capture of the (thread_id, actor_id)
# seen on a read. Lets nested subgraph reads, whose derived config omits actor_id,
# inherit the actor from the current request's parent read instead of failing.
_request_actor: ContextVar[tuple[str, str] | None] = ContextVar(
    "_agentcore_request_actor", default=None
)


class AgentCoreMemorySaver(BaseCheckpointSaver[str]):
    """
    AgentCore Memory checkpoint saver.

    This saver persists Checkpoints as serialized blob events in AgentCore Memory.

    Every read and write requires an ``actor_id`` in the config's ``configurable``.
    The one exception is a *subgraph* read: LangGraph's derived subgraph configs
    (built during ``get_state(subgraphs=True)``) omit ``actor_id`` but carry a
    non-empty ``checkpoint_ns``. Such a read inherits the ``actor_id`` captured —
    for the same ``thread_id`` — from the current request's parent read. That
    capture is request-scoped (a ``ContextVar`` isolated per thread / async task),
    never stored on the saver, so a single saver instance can safely serve many
    actors concurrently. Top-level reads (empty ``checkpoint_ns``) never inherit,
    and writes never inherit; both always require an explicit ``actor_id``.

    !!! warning "Thread-reusing synchronous servers"
        The request-scoped capture is not reset at the end of a request. Under
        asyncio each task runs in its own copied context, so requests do not
        share captures. But a purely synchronous server that reuses an OS thread
        across sequential requests keeps the last ``(thread_id, actor_id)`` in
        that thread's context. Inheritance is gated on both a matching
        ``thread_id`` and a non-empty ``checkpoint_ns``, so a stale capture can
        only be picked up by a later *subgraph-shaped* read for the same
        ``thread_id`` that is not first re-primed by its parent read — a very
        narrow case that also requires reused (non-unique) thread ids across
        actors. Use globally unique thread ids (or always pass ``actor_id`` on
        reads) to avoid it.

    Args:
        memory_id: the ID of the memory resource created in AgentCore Memory
        serde: serialization protocol to be used. Defaults to JSONPlusSerializer
        limit: maximum number of events to parse from ListEvents.
        max_results: maximum number of results to retrieve from AgentCore Memory.
        max_retries: maximum number of retry attempts for retryable errors.
        initial_backoff: initial backoff time in seconds for exponential backoff.
        max_backoff: maximum backoff time in seconds.
    """

    def __init__(
        self,
        memory_id: str,
        *,
        serde: SerializerProtocol | None = None,
        limit: int | None = None,
        max_results: int | None = 100,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        **boto3_kwargs: Any,
    ) -> None:
        super().__init__(serde=serde)

        self.memory_id = memory_id
        self.limit = limit
        self.max_results = max_results
        self.serializer = EventSerializer(self.serde)
        self.checkpoint_event_client = AgentCoreEventClient(
            memory_id,
            self.serializer,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            **boto3_kwargs,
        )
        self.processor = EventProcessor()

    def _resolve_actor_id(self, config: RunnableConfig | None) -> str | None:
        """Resolve the actor id for a read, scoped to the current request.

        When ``config`` supplies both ``thread_id`` and ``actor_id``, record the
        pair for the current request context so a later read of a derived subgraph
        config (which omits ``actor_id``) can inherit it. When ``config`` omits
        ``actor_id``, fall back to the recorded actor only if it was captured for
        the same ``thread_id`` **and** the current read is a derived subgraph read
        (identified by a non-empty ``checkpoint_ns``); otherwise return ``None`` so
        the caller fails closed.

        Restricting inheritance to non-empty ``checkpoint_ns`` reads is what makes
        this safe: LangGraph only omits ``actor_id`` on the per-subgraph configs it
        derives during ``get_state(subgraphs=True)``, and those always carry a
        non-empty namespace. A top-level read has an empty ``checkpoint_ns``, so an
        actor-less top-level read (only possible if a caller omits ``actor_id``)
        fails closed instead of inheriting a possibly stale actor.

        The capture lives in a request-scoped ``ContextVar`` rather than on the
        saver instance, so a shared multi-tenant saver never resolves a subgraph
        read to another request's actor.

        Args:
            config: RunnableConfig for the current read.

        Returns:
            The resolved actor id, or ``None`` if it cannot be resolved from the
            config or the current request context.
        """
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id")
        actor_id = configurable.get("actor_id")
        if actor_id:
            if thread_id:  # only record when we can key the capture by thread
                _request_actor.set((thread_id, actor_id))
            return actor_id
        # Only derived subgraph reads (non-empty checkpoint_ns) may inherit; a
        # top-level read (empty checkpoint_ns) must carry its own actor_id.
        if not configurable.get("checkpoint_ns"):
            return None
        recorded = _request_actor.get()
        if recorded is not None and thread_id is not None and recorded[0] == thread_id:
            return recorded[1]  # returns actor ID as it is the 2nd element in the tuple
        return None

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
        """Get a checkpoint tuple from Bedrock AgentCore Memory.

        Args:
            config: The runnable config containing checkpoint information

        Returns:
            CheckpointTuple if found, None otherwise
        """

        # TODO: There is room for caching here on the client side

        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config),
            default_actor_id=self._resolve_actor_id(config),
        )

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id,
            checkpoint_config.actor_id,
            self.limit,
            self.max_results,
        )

        checkpoints, writes_by_checkpoint, channel_data = self.processor.process_events(
            events
        )

        if not checkpoints:
            return None

        # Find the specific checkpoint if `checkpoint_id` is provided or return the latest one # noqa: E501
        if checkpoint_config.checkpoint_id:
            checkpoint_event = checkpoints.get(checkpoint_config.checkpoint_id)
            if not checkpoint_event:
                return None
        else:
            latest_checkpoint_id = max(checkpoints.keys())
            checkpoint_event = checkpoints[latest_checkpoint_id]

        # Build and return checkpoint tuple
        writes = writes_by_checkpoint.get(checkpoint_event.checkpoint_id, [])
        return self.processor.build_checkpoint_tuple(
            checkpoint_event, writes, channel_data, checkpoint_config
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Bedrock AgentCore Memory."""

        # TODO: There is room for caching here on the client side

        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config) if config else {},
            default_actor_id=self._resolve_actor_id(config),
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id,
            checkpoint_config.actor_id,
            limit,
            self.max_results,
        )

        checkpoints, writes_by_checkpoint, channel_data = self.processor.process_events(
            events
        )

        # Build and yield CheckpointTuples
        count = 0
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        # Sort checkpoints by ID in descending order (most recent first)
        for checkpoint_id in sorted(checkpoints.keys(), reverse=True):
            checkpoint_event = checkpoints[checkpoint_id]
            # Apply filters
            if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                continue

            if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                continue

            if limit is not None and count >= limit:
                break

            writes = writes_by_checkpoint.get(checkpoint_id, [])

            yield self.processor.build_checkpoint_tuple(
                checkpoint_event, writes, channel_data, checkpoint_config
            )

            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to AgentCore Memory."""
        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config)
        )

        # Extract channel values
        checkpoint_copy = dict(checkpoint)
        channel_values: dict[str, Any] = {}
        if "channel_values" in checkpoint_copy:
            channel_values_obj = checkpoint_copy.pop("channel_values")
            if isinstance(channel_values_obj, dict):
                channel_values = channel_values_obj.copy()

        # Create all events to be stored in a single batch
        events_to_store: list[CheckpointEvent | ChannelDataEvent | WritesEvent] = []

        # Create channel data events
        for channel, version in new_versions.items():
            channel_event = ChannelDataEvent(
                channel=channel,
                version=str(version),
                value=channel_values.get(channel, EMPTY_CHANNEL_VALUE),
                thread_id=checkpoint_config.thread_id,
                checkpoint_ns=checkpoint_config.checkpoint_ns,
            )
            events_to_store.append(channel_event)

        checkpoint_event = CheckpointEvent(
            checkpoint_id=checkpoint["id"],
            checkpoint_data=checkpoint_copy,
            metadata=dict(get_checkpoint_metadata(config, metadata)),
            parent_checkpoint_id=checkpoint_config.checkpoint_id,
            thread_id=checkpoint_config.thread_id,
            checkpoint_ns=checkpoint_config.checkpoint_ns,
        )
        events_to_store.append(checkpoint_event)
        typed_events = cast(
            list[CheckpointEvent | ChannelDataEvent | WritesEvent], events_to_store
        )
        self.checkpoint_event_client.store_blob_events_batch(
            typed_events, checkpoint_config.session_id, checkpoint_config.actor_id
        )

        return {
            "configurable": {
                "thread_id": checkpoint_config.thread_id,
                "actor_id": checkpoint_config.actor_id,
                "checkpoint_ns": checkpoint_config.checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes to AgentCore Memory."""
        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config)
        )

        if not checkpoint_config.checkpoint_id:
            raise InvalidConfigError("checkpoint_id is required for put_writes")

        # Create write items
        write_items = [
            WriteItem(
                task_id=task_id,
                channel=channel,
                value=value,
                task_path=task_path,
            )
            for channel, value in writes
        ]

        writes_event = WritesEvent(
            checkpoint_id=checkpoint_config.checkpoint_id,
            writes=write_items,
        )

        self.checkpoint_event_client.store_blob_event(
            writes_event, checkpoint_config.session_id, checkpoint_config.actor_id
        )

    def put_with_writes(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        pending_writes: Sequence[PendingWrite],
    ) -> RunnableConfig:
        """Persist checkpoint and all pending writes in a single API call.

        Args:
            config: The runnable config associated with this checkpoint.
            checkpoint: The checkpoint data to persist.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel version information.
            pending_writes: All buffered writes to persist alongside the
                checkpoint.

        Returns:
            A config pointing to the persisted checkpoint.
        """
        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config)
        )

        checkpoint_copy = dict(checkpoint)
        channel_values: dict[str, Any] = {}
        if "channel_values" in checkpoint_copy:
            channel_values_obj = checkpoint_copy.pop("channel_values")
            if isinstance(channel_values_obj, dict):
                channel_values = channel_values_obj.copy()

        events_to_store: list[CheckpointEvent | ChannelDataEvent | WritesEvent] = []

        for channel, version in new_versions.items():
            channel_event = ChannelDataEvent(
                channel=channel,
                version=str(version),
                value=channel_values.get(channel, EMPTY_CHANNEL_VALUE),
                thread_id=checkpoint_config.thread_id,
                checkpoint_ns=checkpoint_config.checkpoint_ns,
            )
            events_to_store.append(channel_event)

        checkpoint_event = CheckpointEvent(
            checkpoint_id=checkpoint["id"],
            checkpoint_data=checkpoint_copy,
            metadata=dict(get_checkpoint_metadata(config, metadata)),
            parent_checkpoint_id=checkpoint_config.checkpoint_id,
            thread_id=checkpoint_config.thread_id,
            checkpoint_ns=checkpoint_config.checkpoint_ns,
        )
        events_to_store.append(checkpoint_event)

        for pw in pending_writes:
            write_items = [
                WriteItem(
                    task_id=pw.task_id,
                    channel=channel,
                    value=value,
                    task_path=pw.task_path,
                )
                for channel, value in pw.writes
            ]
            writes_event = WritesEvent(
                checkpoint_id=checkpoint["id"],
                writes=write_items,
            )
            events_to_store.append(writes_event)

        typed_events = cast(
            list[CheckpointEvent | ChannelDataEvent | WritesEvent], events_to_store
        )
        self.checkpoint_event_client.store_blob_events_batch(
            typed_events, checkpoint_config.session_id, checkpoint_config.actor_id
        )

        return {
            "configurable": {
                "thread_id": checkpoint_config.thread_id,
                "actor_id": checkpoint_config.actor_id,
                "checkpoint_ns": checkpoint_config.checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_with_writes(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        pending_writes: Sequence[PendingWrite],
    ) -> RunnableConfig:
        """Async version of :meth:`put_with_writes`.

        Args:
            config: The runnable config associated with this checkpoint.
            checkpoint: The checkpoint data to persist.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel version information.
            pending_writes: All buffered writes to persist alongside the
                checkpoint.

        Returns:
            A config pointing to the persisted checkpoint.
        """
        return await run_in_executor(
            None,
            self.put_with_writes,
            config,
            checkpoint,
            metadata,
            new_versions,
            pending_writes,
        )

    def delete_thread(self, thread_id: str, actor_id: str = "") -> None:
        """Delete all checkpoints and writes associated with a thread."""
        self.checkpoint_event_client.delete_events(thread_id, actor_id)

    # ===== Async methods ( Running sync methods inside executor ) =====
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        # Capture the actor in the async context so a later async subgraph read
        # inherits it; run_in_executor copies this context into the worker thread.
        self._resolve_actor_id(config)
        return await run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Capture the actor in the async context so a later async subgraph read
        # inherits it; run_in_executor copies this context into the worker thread.
        self._resolve_actor_id(config)

        def _sync_list() -> list[CheckpointTuple]:
            return list(self.list(config, filter=filter, before=before, limit=limit))

        items = await run_in_executor(None, _sync_list)
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str, actor_id: str = "") -> None:
        await run_in_executor(None, self.delete_thread, thread_id, actor_id)
        return None

    def get_next_version(
        self, current: str | int | None, channel: str | None = None
    ) -> str:
        """Generate next version string."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])

        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
