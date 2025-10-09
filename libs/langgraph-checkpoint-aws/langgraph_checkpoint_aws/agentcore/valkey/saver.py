"""
AgentCore Valkey Checkpoint Saver implementation.

Combines AgentCore session management concepts with Valkey storage backend.
"""

from __future__ import annotations

import builtins
import random
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from valkey import Valkey
from valkey.asyncio import Valkey as AsyncValkey
from valkey.connection import ConnectionPool

from ...checkpoint.valkey.utils import set_client_info
from ..constants import InvalidConfigError
from ..helpers import EventSerializer
from .models import (
    StoredChannelData,
    StoredCheckpoint,
    StoredWrite,
    ValkeyCheckpointerConfig,
)


class AgentCoreValkeySaver(BaseCheckpointSaver[str]):
    """
    AgentCore Valkey checkpoint saver.

    This saver combines AgentCore session management concepts with Valkey storage,
    providing high-performance checkpoint persistence with AgentCore-compatible
    session and actor management.

    Args:
        client: The Valkey client instance (sync or async).
        ttl: Time-to-live for stored checkpoints in seconds. Defaults to None (no expiration).
        serde: Serialization protocol to be used. Defaults to JsonPlusSerializer.
        **kwargs: Additional arguments passed to the base class.

    Examples:
        >>> from valkey import Valkey
        >>> from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver
        >>>
        >>> client = Valkey.from_url("valkey://localhost:6379")
        >>> checkpointer = AgentCoreValkeySaver(client, ttl=3600)  # 1 hour TTL
        >>>
        >>> # Use with LangGraph
        >>> config = {
        ...     "configurable": {
        ...         "thread_id": "session-1",
        ...         "actor_id": "agent-1",
        ...         "checkpoint_ns": ""
        ...     }
        ... }
        >>> graph = create_react_agent(model, tools, checkpointer=checkpointer)
        >>> result = graph.invoke({"messages": [...]}, config)
    """

    def __init__(
        self,
        client: Valkey | AsyncValkey,
        *,
        ttl: float | None = None,
        serde: SerializerProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(serde=serde, **kwargs)

        self.client = client
        self.ttl = ttl
        self.jsonplus_serde = JsonPlusSerializer()
        self.event_serializer = EventSerializer(self.serde)
        self.lock = threading.Lock()

        # Set client info for library identification
        if hasattr(client, "aclose") or hasattr(client, "__aenter__"):
            # Async client - will be handled in async methods
            pass
        else:
            # Sync client
            set_client_info(client)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        ttl_seconds: float | None = None,
        pool_size: int = 10,
        **kwargs: Any,
    ) -> Iterator[AgentCoreValkeySaver]:
        """Create a new AgentCoreValkeySaver instance from a connection string.

        Args:
            conn_string: The Valkey connection string.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.
            pool_size: Maximum number of connections in the pool.
            **kwargs: Additional arguments passed to Valkey client.

        Yields:
            AgentCoreValkeySaver: A new AgentCoreValkeySaver instance.

        Examples:
            >>> with AgentCoreValkeySaver.from_conn_string("valkey://localhost:6379") as saver:
            ...     # Use the saver instance
            ...     pass
        """
        pool = ConnectionPool.from_url(conn_string, max_connections=pool_size)
        client = Valkey(connection_pool=pool, **kwargs)
        try:
            yield cls(client, ttl=ttl_seconds)
        finally:
            client.close()

    @classmethod
    @contextmanager
    def from_pool(
        cls,
        pool: ConnectionPool,
        *,
        ttl_seconds: float | None = None,
    ) -> Iterator[AgentCoreValkeySaver]:
        """Create a new AgentCoreValkeySaver instance from a connection pool.

        Args:
            pool: The Valkey connection pool.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.

        Yields:
            AgentCoreValkeySaver: A new AgentCoreValkeySaver instance.
        """
        client = Valkey(connection_pool=pool)
        try:
            yield cls(client, ttl=ttl_seconds)
        finally:
            client.close()

    def _make_checkpoint_key(
        self, config: ValkeyCheckpointerConfig, checkpoint_id: str
    ) -> str:
        """Generate a key for storing checkpoint data."""
        return f"{config.checkpoint_key_prefix}:{checkpoint_id}"

    def _make_writes_key(
        self, config: ValkeyCheckpointerConfig, checkpoint_id: str
    ) -> str:
        """Generate a key for storing writes data."""
        return f"{config.writes_key_prefix}:{checkpoint_id}"

    def _make_channel_key(
        self, config: ValkeyCheckpointerConfig, channel: str, checkpoint_id: str
    ) -> str:
        """Generate a key for storing channel data."""
        return f"{config.channel_key_prefix}:{channel}:{checkpoint_id}"

    def _make_session_checkpoints_key(self, config: ValkeyCheckpointerConfig) -> str:
        """Generate a key for storing session checkpoint list."""
        return f"{config.session_key}:checkpoints"

    def _serialize_checkpoint(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> StoredCheckpoint:
        """Serialize checkpoint data for storage."""
        checkpoint_config = ValkeyCheckpointerConfig.from_runnable_config(config)

        # Serialize checkpoint and metadata
        checkpoint_data = self.event_serializer.serialize_value(checkpoint)
        serialized_metadata = self.event_serializer.serialize_value(
            get_checkpoint_metadata(config, metadata)
        )

        return StoredCheckpoint(
            checkpoint_id=checkpoint["id"],
            session_id=checkpoint_config.session_id,
            thread_id=checkpoint_config.thread_id,
            actor_id=checkpoint_config.actor_id,
            checkpoint_ns=checkpoint_config.checkpoint_ns,
            parent_checkpoint_id=checkpoint_config.checkpoint_id,
            checkpoint_data=checkpoint_data,
            metadata=serialized_metadata,
            created_at=time.time(),
        )

    def _deserialize_checkpoint(
        self,
        stored_checkpoint: StoredCheckpoint,
        writes: builtins.list[StoredWrite],
        channel_data: dict[str, StoredChannelData],
        config: RunnableConfig | None = None,
    ) -> CheckpointTuple:
        """Deserialize checkpoint data from storage."""
        # Deserialize checkpoint and metadata
        checkpoint = self.event_serializer.deserialize_value(
            stored_checkpoint.checkpoint_data
        )
        metadata = self.event_serializer.deserialize_value(stored_checkpoint.metadata)

        # Create parent config if exists
        parent_config: RunnableConfig | None = None
        if stored_checkpoint.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": stored_checkpoint.thread_id,
                    "actor_id": stored_checkpoint.actor_id,
                    "checkpoint_ns": stored_checkpoint.checkpoint_ns,
                    "checkpoint_id": stored_checkpoint.parent_checkpoint_id,
                }
            }

        # Deserialize writes
        pending_writes = []
        for write in writes:
            deserialized_value = self.event_serializer.deserialize_value(write.value)
            pending_writes.append((write.task_id, write.channel, deserialized_value))

        # Use provided config or generate one
        if config is None:
            config = {
                "configurable": {
                    "thread_id": stored_checkpoint.thread_id,
                    "actor_id": stored_checkpoint.actor_id,
                    "checkpoint_ns": stored_checkpoint.checkpoint_ns,
                    "checkpoint_id": stored_checkpoint.checkpoint_id,
                }
            }

        return CheckpointTuple(
            config,
            checkpoint,
            metadata,
            parent_config,
            pending_writes,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from Valkey storage."""
        checkpoint_config = ValkeyCheckpointerConfig.from_runnable_config(config)

        # Get specific checkpoint or latest
        if checkpoint_config.checkpoint_id:
            checkpoint_key = self._make_checkpoint_key(
                checkpoint_config, checkpoint_config.checkpoint_id
            )
            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None
            try:
                stored_checkpoint = StoredCheckpoint.model_validate_json(
                    checkpoint_data
                )
            except Exception as e:
                # Handle corrupted data by raising an exception
                raise ValueError(f"Failed to parse checkpoint data: {e}") from e
        else:
            # Get latest checkpoint from session list
            session_key = self._make_session_checkpoints_key(checkpoint_config)
            latest_checkpoint_id = self.client.lindex(session_key, 0)
            if not latest_checkpoint_id:
                return None

            checkpoint_key = self._make_checkpoint_key(
                checkpoint_config, latest_checkpoint_id.decode()
            )
            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None
            try:
                stored_checkpoint = StoredCheckpoint.model_validate_json(
                    checkpoint_data
                )
            except Exception as e:
                # Handle corrupted data by raising an exception
                raise ValueError(f"Failed to parse checkpoint data: {e}") from e

        # Get writes for this checkpoint
        writes_key = self._make_writes_key(
            checkpoint_config, stored_checkpoint.checkpoint_id
        )
        writes_data = self.client.lrange(writes_key, 0, -1)
        try:
            writes = [StoredWrite.model_validate_json(w) for w in writes_data]
        except Exception as e:
            # Handle corrupted writes data
            raise ValueError(f"Failed to parse writes data: {e}") from e

        # Get channel data (if needed)
        channel_data = {}

        # Create a config that includes the checkpoint_id for the return value
        result_config = {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": stored_checkpoint.checkpoint_id,
            }
        }

        return self._deserialize_checkpoint(
            stored_checkpoint, writes, channel_data, result_config
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Valkey storage."""
        if not config:
            return

        checkpoint_config = ValkeyCheckpointerConfig.from_runnable_config(config)
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        # Get checkpoint list from session
        session_key = self._make_session_checkpoints_key(checkpoint_config)
        checkpoint_ids = self.client.lrange(session_key, 0, -1)

        count = 0
        for checkpoint_id_bytes in checkpoint_ids:
            checkpoint_id = checkpoint_id_bytes.decode()

            # Apply before filter
            if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                continue

            # Apply limit
            if limit is not None and count >= limit:
                break

            # Get checkpoint data
            checkpoint_key = self._make_checkpoint_key(checkpoint_config, checkpoint_id)
            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                continue

            stored_checkpoint = StoredCheckpoint.model_validate_json(checkpoint_data)

            # Apply metadata filter
            if filter:
                metadata = self.event_serializer.deserialize_value(
                    stored_checkpoint.metadata
                )
                if not all(
                    key in metadata and metadata[key] == value
                    for key, value in filter.items()
                ):
                    continue

            # Get writes for this checkpoint
            writes_key = self._make_writes_key(checkpoint_config, checkpoint_id)
            writes_data = self.client.lrange(writes_key, 0, -1)
            writes = [StoredWrite.model_validate_json(w) for w in writes_data]

            # Get channel data
            channel_data = {}

            yield self._deserialize_checkpoint(stored_checkpoint, writes, channel_data)
            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Valkey storage."""
        checkpoint_config = ValkeyCheckpointerConfig.from_runnable_config(config)

        # Serialize checkpoint
        stored_checkpoint = self._serialize_checkpoint(config, checkpoint, metadata)

        # Store checkpoint
        checkpoint_key = self._make_checkpoint_key(checkpoint_config, checkpoint["id"])
        checkpoint_json = stored_checkpoint.model_dump_json()

        if self.ttl:
            self.client.setex(checkpoint_key, int(self.ttl), checkpoint_json)
        else:
            self.client.set(checkpoint_key, checkpoint_json)

        # Add to session checkpoint list (most recent first)
        session_key = self._make_session_checkpoints_key(checkpoint_config)
        self.client.lpush(session_key, checkpoint["id"])
        if self.ttl:
            self.client.expire(session_key, int(self.ttl))

        # Store channel data
        checkpoint_copy = checkpoint.copy()
        channel_values: dict[str, Any] = checkpoint_copy.pop("channel_values", {})

        for channel, version in new_versions.items():
            if channel in channel_values:
                channel_data = StoredChannelData(
                    channel=channel,
                    version=version,
                    value=self.event_serializer.serialize_value(
                        channel_values[channel]
                    ),
                    checkpoint_id=checkpoint["id"],
                    created_at=time.time(),
                )

                channel_key = self._make_channel_key(
                    checkpoint_config, channel, checkpoint["id"]
                )
                if self.ttl:
                    self.client.setex(
                        channel_key, int(self.ttl), channel_data.model_dump_json()
                    )
                else:
                    self.client.set(channel_key, channel_data.model_dump_json())

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
        """Save pending writes to Valkey storage."""
        checkpoint_config = ValkeyCheckpointerConfig.from_runnable_config(config)

        if not checkpoint_config.checkpoint_id:
            raise InvalidConfigError("checkpoint_id is required for put_writes")

        # Store writes
        writes_key = self._make_writes_key(
            checkpoint_config, checkpoint_config.checkpoint_id
        )

        for channel, value in writes:
            stored_write = StoredWrite(
                checkpoint_id=checkpoint_config.checkpoint_id,
                task_id=task_id,
                channel=channel,
                value=self.event_serializer.serialize_value(value),
                task_path=task_path,
                created_at=time.time(),
            )

            self.client.rpush(writes_key, stored_write.model_dump_json())
            if self.ttl:
                self.client.expire(writes_key, int(self.ttl))

    def delete_thread(self, thread_id: str, actor_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread."""
        # Create a temporary config to get keys
        temp_config = ValkeyCheckpointerConfig(
            thread_id=thread_id,
            actor_id=actor_id,
            checkpoint_ns="",
        )

        # Get all checkpoint IDs for this session
        session_key = self._make_session_checkpoints_key(temp_config)
        checkpoint_ids = self.client.lrange(session_key, 0, -1)

        # Delete all checkpoint-related keys
        keys_to_delete = [session_key]

        for checkpoint_id_bytes in checkpoint_ids:
            checkpoint_id = checkpoint_id_bytes.decode()

            # Add checkpoint key
            checkpoint_key = self._make_checkpoint_key(temp_config, checkpoint_id)
            keys_to_delete.append(checkpoint_key)

            # Add writes key
            writes_key = self._make_writes_key(temp_config, checkpoint_id)
            keys_to_delete.append(writes_key)

            # Add channel keys (use pattern matching)
            channel_pattern = f"{temp_config.channel_key_prefix}:*:{checkpoint_id}"
            channel_keys = self.client.keys(channel_pattern)
            keys_to_delete.extend(channel_keys)

        # Delete all keys in batch
        if keys_to_delete:
            self.client.delete(*keys_to_delete)

    # ===== Async methods (delegated to sync for now) =====
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple."""
        # For now, delegate to sync version
        # In a full implementation, this would use async Valkey client
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list."""
        # For now, delegate to sync version
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes."""
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str, actor_id: str) -> None:
        """Async version of delete_thread."""
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
