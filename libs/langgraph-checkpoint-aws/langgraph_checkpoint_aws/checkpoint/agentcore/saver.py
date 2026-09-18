"""
AgentCore Memory Checkpoint Saver implementation.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, TypeAlias

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
from .snapshot import AgentCoreSnapshotClient, writes_session_id

RunnableConfigDict: TypeAlias = dict[str, Any]


class AgentCoreMemorySaver(BaseCheckpointSaver[str]):
    """
    AgentCore Memory checkpoint saver.

    This saver persists checkpoints as blob events in AgentCore Memory.

    Args:
        memory_id: the ID of the memory resource created in AgentCore Memory
        serde: serialization protocol to be used. Defaults to JSONPlusSerializer
        limit: Maximum decoded payload blobs for legacy checkpoint reads.
            Snapshot mode requires None so snapshot data is never truncated.
        max_results: Maximum number of service events per pagination request.
        max_retries: maximum number of retry attempts for retryable errors.
        initial_backoff: initial backoff time in seconds for exponential backoff.
        max_backoff: maximum backoff time in seconds.
        checkpoint_format: "legacy" preserves the existing storage format.
            "snapshot" is experimental and stores complete checkpoints for
            bounded reads. All workers sharing a snapshot thread must use snapshot mode.
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
        checkpoint_format: Literal["legacy", "snapshot"] = "legacy",
        **boto3_kwargs: Any,
    ) -> None:
        super().__init__(serde=serde)

        self.memory_id = memory_id
        if checkpoint_format not in ("legacy", "snapshot"):
            msg = "checkpoint_format must be 'legacy' or 'snapshot'."
            raise ValueError(msg)
        if checkpoint_format == "snapshot":
            if limit is not None:
                msg = (
                    "checkpoint_format='snapshot' requires limit=None. "
                    "Snapshot reads retrieve the complete snapshot and its writes."
                )
                raise ValueError(msg)
            if max_results is not None and not 1 <= max_results <= 100:
                msg = "max_results must be between 1 and 100, or None."
                raise ValueError(msg)
        self.limit = limit
        self.max_results = max_results
        self.checkpoint_format = checkpoint_format
        self.serializer = EventSerializer(self.serde)
        client_class = (
            AgentCoreSnapshotClient
            if checkpoint_format == "snapshot"
            else AgentCoreEventClient
        )
        self.checkpoint_event_client = client_class(
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

        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config)
        )

        events = None
        if isinstance(self.checkpoint_event_client, AgentCoreSnapshotClient):
            events = self.checkpoint_event_client.get_snapshot(
                checkpoint_config.session_id,
                checkpoint_config.actor_id,
                checkpoint_config.checkpoint_id,
                self.max_results,
            )
        if events is None:
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

        self._check_read_format(checkpoint_event)
        # Build and return checkpoint tuple
        writes = writes_by_checkpoint.get(checkpoint_event.checkpoint_id, [])
        if isinstance(self.checkpoint_event_client, AgentCoreSnapshotClient):
            writes.extend(
                self.checkpoint_event_client.get_pending_writes(
                    checkpoint_config.thread_id,
                    checkpoint_config.actor_id,
                    checkpoint_event.checkpoint_id,
                    self.max_results,
                    checkpoint_ns=checkpoint_config.checkpoint_ns,
                )
            )
        return self.processor.build_checkpoint_tuple(
            checkpoint_event, writes, channel_data, checkpoint_config
        )

    def _check_read_format(self, checkpoint: CheckpointEvent) -> None:
        if (
            checkpoint.snapshot_version is not None
            and self.checkpoint_format != "snapshot"
        ):
            msg = (
                "This checkpoint uses snapshot storage. Initialize "
                "AgentCoreMemorySaver with checkpoint_format='snapshot' to restore "
                "its pending writes."
            )
            raise InvalidConfigError(msg)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Bedrock AgentCore Memory."""

        checkpoint_config = CheckpointerConfig.from_runnable_config(
            RunnableConfigDict(config) if config else {}
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        snapshot_mode = self.checkpoint_format == "snapshot"
        if snapshot_mode and limit is not None and limit <= 0:
            return

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id,
            checkpoint_config.actor_id,
            None if snapshot_mode else limit,
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

            if (
                snapshot_mode
                and filter
                and any(
                    checkpoint_event.metadata.get(key) != value
                    for key, value in filter.items()
                )
            ):
                continue

            if limit is not None and count >= limit:
                break

            self._check_read_format(checkpoint_event)
            writes = writes_by_checkpoint.get(checkpoint_id, [])
            if isinstance(self.checkpoint_event_client, AgentCoreSnapshotClient):
                writes.extend(
                    self.checkpoint_event_client.get_pending_writes(
                        checkpoint_config.thread_id,
                        checkpoint_config.actor_id,
                        checkpoint_id,
                        self.max_results,
                        checkpoint_ns=checkpoint_config.checkpoint_ns,
                    )
                )

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
        return self.put_with_writes(config, checkpoint, metadata, new_versions, [])

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

        session_id = checkpoint_config.session_id
        if self.checkpoint_format == "snapshot":
            session_id = writes_session_id(
                checkpoint_config.thread_id,
                checkpoint_config.checkpoint_id,
                checkpoint_config.checkpoint_ns,
            )
        self.checkpoint_event_client.store_blob_event(
            writes_event, session_id, checkpoint_config.actor_id
        )

    def put_with_writes(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        pending_writes: Sequence[PendingWrite],
    ) -> RunnableConfig:
        """Persist a checkpoint and all buffered writes together.

        Args:
            config: The runnable config associated with this checkpoint.
            checkpoint: The checkpoint data to persist.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel version changes supplied by LangGraph. Snapshots
                include all current channels, including unchanged versions.
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

        # Snapshots include unchanged values so a resumed reader never needs to
        # walk older checkpoints to reconstruct this state.
        versions = (
            checkpoint.get("channel_versions", {})
            if self.checkpoint_format == "snapshot"
            else new_versions
        )
        for channel, version in versions.items():
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

        if isinstance(self.checkpoint_event_client, AgentCoreSnapshotClient):
            self.checkpoint_event_client.store_snapshot(
                events_to_store,
                checkpoint_config.session_id,
                checkpoint_config.actor_id,
            )
        else:
            self.checkpoint_event_client.store_blob_events_batch(
                events_to_store,
                checkpoint_config.session_id,
                checkpoint_config.actor_id,
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
