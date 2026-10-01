"""
AgentCore Memory Checkpoint Saver implementation.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

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

if TYPE_CHECKING:
    # `DeltaChannelHistory` was added to `langgraph-checkpoint` alongside
    # `DeltaChannel`/`get_delta_channel_history` support and does not exist
    # in `langgraph-checkpoint` 3.0.0, the package's declared floor
    # (`langgraph-checkpoint>=3.0.0,<5.0.0`). A module-level import would
    # make importing this whole package fail on 3.0.x. It is only ever used
    # here as a type annotation, and `from __future__ import annotations`
    # (above) makes all annotations in this module lazy strings, so a
    # `TYPE_CHECKING`-only import is sufficient: static type checkers still
    # resolve it, and nothing evaluates it at runtime.
    from langgraph.checkpoint.base import DeltaChannelHistory

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
    EventType,
)
from .models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)

RunnableConfigDict: TypeAlias = dict[str, Any]


class AgentCoreMemorySaver(BaseCheckpointSaver[str]):
    """
    AgentCore Memory checkpoint saver.

    This saver persists Checkpoints as serialized blob events in AgentCore Memory.

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
            RunnableConfigDict(config)
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
            RunnableConfigDict(config) if config else {}
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

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Walk the parent chain returning per-channel writes and seed.

        !!! warning "Beta"

            `get_delta_channel_history` is part of langgraph's `DeltaChannel`
            support surface and is marked Beta upstream: the signature,
            return shape (`DeltaChannelHistory`), and interaction with
            `_DeltaSnapshot` blobs may change. Re-check this override
            against `BaseCheckpointSaver.get_delta_channel_history` on any
            `langgraph-checkpoint` upgrade.

        The base implementation issues one `get_tuple` (one AgentCore
        `ListEvents` round trip) per ancestor checkpoint, which is O(depth)
        network calls and dominates resume latency on long-running threads.
        This override instead pages through the thread's events in bulk via
        `AgentCoreEventClient.iter_event_pages`, reconstructs checkpoints
        from them locally, and replays the identical parent-chain walk —
        stopping as soon as the walk is satisfied rather than always
        fetching the full thread history.

        Args:
            config: Configuration identifying the target checkpoint.
            channels: Channel names to walk for. Empty returns an empty
                mapping.

        Returns:
            Per-channel `DeltaChannelHistory` for every name in `channels`.
        """
        if not channels:
            return {}

        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config)
        )
        target_tuple = self.get_tuple(config)

        tuples_by_id: dict[str, CheckpointTuple] = {}
        result, complete = self._replay_delta_channel_history(
            target_tuple, tuples_by_id, channels
        )

        if not complete:
            all_events: list[EventType] = []
            for page in self.checkpoint_event_client.iter_event_pages(
                checkpoint_config.session_id,
                checkpoint_config.actor_id,
                max_results=self.max_results,
            ):
                all_events.extend(page)
                tuples_by_id = self._tuples_by_checkpoint_id(
                    all_events, checkpoint_config
                )
                result, complete = self._replay_delta_channel_history(
                    target_tuple, tuples_by_id, channels
                )
                if complete:
                    break

        return result

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Async version of `get_delta_channel_history`.

        !!! warning "Beta"

            See `get_delta_channel_history` for caveats; this method shares
            the same beta status upstream.
        """
        return await run_in_executor(
            None, self.get_delta_channel_history, config=config, channels=channels
        )

    def _tuples_by_checkpoint_id(
        self,
        events: Sequence[EventType],
        checkpoint_config: CheckpointerConfig,
    ) -> dict[str, CheckpointTuple]:
        """Reconstruct `CheckpointTuple`s keyed by checkpoint id from events.

        Args:
            events: All events fetched so far for the thread.
            checkpoint_config: The thread/actor/namespace being walked.

        Returns:
            Every checkpoint reconstructable from `events`, keyed by its id.
        """
        checkpoints, writes_by_checkpoint, channel_data = self.processor.process_events(
            list(events)
        )
        return {
            checkpoint_id: self.processor.build_checkpoint_tuple(
                checkpoint_event,
                writes_by_checkpoint.get(checkpoint_id, []),
                channel_data,
                checkpoint_config,
            )
            for checkpoint_id, checkpoint_event in checkpoints.items()
        }

    @staticmethod
    def _replay_delta_channel_history(
        target_tuple: CheckpointTuple | None,
        tuples_by_id: dict[str, CheckpointTuple],
        channels: Sequence[str],
    ) -> tuple[dict[str, DeltaChannelHistory], bool]:
        """Replay the base `get_delta_channel_history` walk against known tuples.

        Mirrors `BaseCheckpointSaver.get_delta_channel_history` exactly:
        same write ordering (writes are collected newest-to-oldest per
        ancestor then reversed once at the end, giving oldest-to-newest
        overall), and the same seed selection at the nearest ancestor whose
        `channel_values[ch]` is populated. The only difference is that
        ancestors are looked up in an in-memory `tuples_by_id` map — built
        from `parent_config` links, never from fetch order — instead of one
        `get_tuple` call per ancestor.

        Args:
            target_tuple: The checkpoint tuple identified by the caller's
                config.
            tuples_by_id: Checkpoints reconstructed so far, keyed by
                checkpoint id. Ancestors not yet fetched are simply absent.
            channels: Channel names to walk for.

        Returns:
            The per-channel result built from the tuples available so far,
            and whether the walk is complete (reached the root, or every
            channel is seeded) as opposed to stalled on an ancestor that
            has not been fetched yet.
        """
        collected_by_ch: dict[str, list[tuple[str, str, Any]]] = {
            c: [] for c in channels
        }
        seed_by_ch: dict[str, Any] = {}
        remaining: set[str] = set(channels)
        cursor_config = target_tuple.parent_config if target_tuple else None
        complete = True

        while cursor_config is not None and remaining:
            cursor_id = get_checkpoint_id(cursor_config)
            tup = tuples_by_id.get(cursor_id) if cursor_id is not None else None
            if tup is None:
                complete = False
                break
            if tup.pending_writes:
                for write in reversed(tup.pending_writes):
                    if write[1] in remaining:
                        collected_by_ch[write[1]].append(write)
            for ch in list(remaining):
                if ch in tup.checkpoint["channel_values"]:
                    seed_by_ch[ch] = tup.checkpoint["channel_values"][ch]
                    remaining.discard(ch)
            cursor_config = tup.parent_config

        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            entry: DeltaChannelHistory = {"writes": list(reversed(collected_by_ch[ch]))}
            if ch in seed_by_ch:
                entry["seed"] = seed_by_ch[ch]
            result[ch] = entry

        return result, complete

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
        return await run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()

        def _sync_list():
            return list(self.list(config, filter=filter, before=before, limit=limit))

        items = await loop.run_in_executor(None, _sync_list)
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
