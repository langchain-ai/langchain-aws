"""
AgentCore Memory Checkpoint Saver implementation.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
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

from langgraph_checkpoint_aws.agentcore.constants import (
    EMPTY_CHANNEL_VALUE,
    InvalidConfigError,
)
from langgraph_checkpoint_aws.agentcore.helpers import (
    AgentCoreEventClient,
    EventProcessor,
    EventSerializer,
)
from langgraph_checkpoint_aws.agentcore.models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)


class AgentCoreMemorySaver(BaseCheckpointSaver[str]):
    """
    AgentCore Memory checkpoint saver.

    This saver persists Checkpoints as serialized blob events in AgentCore Memory.

    Args:
        memory_id: the ID of the memory resource created in AgentCore Memory
        serde: serialization protocol to be used. Defaults to JSONPlusSerializer
    """

    def __init__(
        self,
        memory_id: str,
        *,
        serde: SerializerProtocol | None = None,
        **boto3_kwargs: Any,
    ) -> None:
        super().__init__(serde=serde)

        self.memory_id = memory_id
        self.serializer = EventSerializer(self.serde)
        self.checkpoint_event_client = AgentCoreEventClient(
            memory_id, self.serializer, **boto3_kwargs
        )
        self.processor = EventProcessor()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from Bedrock AgentCore Memory."""

        # TODO: There is room for caching here on the client side

        checkpoint_config = CheckpointerConfig.from_runnable_config(config)

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id, checkpoint_config.actor_id
        )

        checkpoints, writes_by_checkpoint, channel_data = self.processor.process_events(
            events
        )

        if not checkpoints:
            return None

        # Find the specific checkpoint if `checkpoint_id` is provided or return the latest one
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

        checkpoint_config = CheckpointerConfig.from_runnable_config(config)
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id, checkpoint_config.actor_id, limit
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
        checkpoint_config = CheckpointerConfig.from_runnable_config(config)

        # Extract channel values
        checkpoint_copy = checkpoint.copy()
        channel_values: Dict[str, Any] = checkpoint_copy.pop("channel_values", {})

        # Create all events to be stored in a single batch
        events_to_store = []

        # Create channel data events
        for channel, version in new_versions.items():
            channel_event = ChannelDataEvent(
                channel=channel,
                version=version,
                value=channel_values.get(channel, EMPTY_CHANNEL_VALUE),
                thread_id=checkpoint_config.thread_id,
                checkpoint_ns=checkpoint_config.checkpoint_ns,
            )
            events_to_store.append(channel_event)

        checkpoint_event = CheckpointEvent(
            checkpoint_id=checkpoint["id"],
            checkpoint_data=checkpoint_copy,
            metadata=get_checkpoint_metadata(config, metadata),
            parent_checkpoint_id=checkpoint_config.checkpoint_id,
            thread_id=checkpoint_config.thread_id,
            checkpoint_ns=checkpoint_config.checkpoint_ns,
        )
        events_to_store.append(checkpoint_event)

        self.checkpoint_event_client.store_blob_events_batch(
            events_to_store, checkpoint_config.session_id, checkpoint_config.actor_id
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
        checkpoint_config = CheckpointerConfig.from_runnable_config(config)

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

    def delete_thread(self, thread_id: str, actor_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread."""
        self.checkpoint_event_client.delete_events(thread_id, actor_id)

    # ===== Async methods ( TODO: NOT IMPLEMENTED YET ) =====
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str, actor_id: str) -> None:
        return self.delete_thread(thread_id, actor_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
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
