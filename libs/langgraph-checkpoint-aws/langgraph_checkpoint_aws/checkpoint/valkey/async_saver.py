"""Async Valkey checkpoint saver implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import orjson
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from valkey.asyncio import Valkey as AsyncValkey
from valkey.asyncio.connection import ConnectionPool as AsyncConnectionPool

from .base import BaseValkeyCheckpointSaver
from .utils import aset_client_info


class AsyncValkeyCheckpointSaver(BaseValkeyCheckpointSaver):
    """An async checkpoint saver that stores checkpoints in Valkey (Redis-compatible).

    This class provides asynchronous methods for storing and retrieving checkpoints
    using Valkey as the backend storage.

    Args:
        client: The AsyncValkey client instance.
        ttl: Time-to-live for stored checkpoints in seconds. Defaults to None (no expiration).
        serde: The serializer to use for serializing and deserializing checkpoints.

    Examples:

        >>> from valkey.asyncio import Valkey as AsyncValkey
        >>> from langgraph.checkpoint.valkey.aio import AsyncValkeyCheckpointSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # Create a new AsyncValkeyCheckpointSaver instance
        >>> client = AsyncValkey.from_url("valkey://localhost:6379")
        >>> memory = AsyncValkeyCheckpointSaver(client)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> await graph.aget_state(config)
        >>> result = await graph.ainvoke(3, config)
        >>> await graph.aget_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'}}, parent_config=None)
    """

    def __init__(
        self,
        client: AsyncValkey,
        *,
        ttl: float | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(client, ttl=ttl, serde=serde)
        # Note: aset_client_info cannot be called here since __init__ is not async
        # It should be called in async factory methods like from_conn_string and from_pool

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        ttl_seconds: float | None = None,
        serde: SerializerProtocol | None = None,
        pool_size: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[AsyncValkeyCheckpointSaver]:
        """Create a new AsyncValkeyCheckpointSaver instance from a connection string.

        Args:
            conn_string: The Valkey connection string.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.
            serde: The serializer to use for serializing and deserializing checkpoints.
            pool_size: Maximum number of connections in the pool.
            **kwargs: Additional arguments passed to AsyncValkey client.

        Yields:
            AsyncValkeyCheckpointSaver: A new AsyncValkeyCheckpointSaver instance.

        Examples:

            >>> async with AsyncValkeyCheckpointSaver.from_conn_string("valkey://localhost:6379") as memory:
            ...     # Use the memory instance
            ...     pass
        """
        client = AsyncValkey.from_url(conn_string, max_connections=pool_size, **kwargs)
        try:
            # Set client info for library identification
            await aset_client_info(client)
            yield cls(client, ttl=ttl_seconds, serde=serde)
        finally:
            await client.aclose()

    @classmethod
    @asynccontextmanager
    async def from_pool(
        cls,
        pool: AsyncConnectionPool,
        *,
        ttl_seconds: float | None = None,
    ) -> AsyncIterator[AsyncValkeyCheckpointSaver]:
        """Create a new AsyncValkeyCheckpointSaver instance from a connection pool.

        Args:
            pool: The Valkey async connection pool.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.

        Yields:
            AsyncValkeyCheckpointSaver: A new AsyncValkeyCheckpointSaver instance.

        Examples:

            >>> from valkey.asyncio.connection import ConnectionPool as AsyncConnectionPool
            >>> pool = AsyncConnectionPool.from_url("valkey://localhost:6379")
            >>> async with AsyncValkeyCheckpointSaver.from_pool(pool) as memory:
            ...     # Use the memory instance
            ...     pass
        """
        client = AsyncValkey(connection_pool=pool)
        try:
            # Set client info for library identification
            await aset_client_info(client)
            yield cls(client, ttl=ttl_seconds)
        finally:
            await client.aclose()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Valkey database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")

        if checkpoint_id := get_checkpoint_id(config):
            # Get specific checkpoint
            checkpoint_key = self._make_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )
            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            checkpoint_data = await self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            checkpoint_info = orjson.loads(checkpoint_data)
            writes_data = await self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

        else:
            # Get latest checkpoint
            thread_key = self._make_thread_key(thread_id, checkpoint_ns)
            checkpoint_ids = await self.client.lrange(
                thread_key, 0, 0
            )  # Get most recent

            if not checkpoint_ids:
                return None

            checkpoint_id = checkpoint_ids[0].decode()
            checkpoint_key = self._make_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )
            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            checkpoint_data = await self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            checkpoint_info = orjson.loads(checkpoint_data)
            writes_data = await self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

            # Update config with checkpoint_id
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        return self._deserialize_checkpoint_data(
            checkpoint_info, writes, thread_id, checkpoint_ns, checkpoint_id, config
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Valkey database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An async iterator of checkpoint tuples.
        """
        if not config:
            return

        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        thread_key = self._make_thread_key(thread_id, checkpoint_ns)

        # Get all checkpoint IDs for this thread
        checkpoint_ids = await self.client.lrange(thread_key, 0, -1)

        # Apply before filter
        if before and (before_id := get_checkpoint_id(before)):
            try:
                before_idx = next(
                    i
                    for i, cid in enumerate(checkpoint_ids)
                    if cid.decode() == before_id
                )
                checkpoint_ids = checkpoint_ids[before_idx + 1 :]
            except StopIteration:
                pass

        # Apply limit
        if limit:
            checkpoint_ids = checkpoint_ids[:limit]

        for checkpoint_id_bytes in checkpoint_ids:
            checkpoint_id = checkpoint_id_bytes.decode()
            checkpoint_key = self._make_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )
            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            checkpoint_data = await self.client.get(checkpoint_key)
            if not checkpoint_data:
                continue

            checkpoint_info = orjson.loads(checkpoint_data)

            # Apply metadata filter
            if not self._should_include_checkpoint(checkpoint_info, filter):
                continue

            writes_data = await self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

            yield self._deserialize_checkpoint_data(
                checkpoint_info, writes, thread_id, checkpoint_ns, checkpoint_id
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Valkey database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint data
        checkpoint_info = self._serialize_checkpoint_data(config, checkpoint, metadata)

        # Store checkpoint
        checkpoint_key = self._make_checkpoint_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        thread_key = self._make_thread_key(thread_id, checkpoint_ns)

        pipe = await self.client.pipeline()
        pipe.set(checkpoint_key, orjson.dumps(checkpoint_info))
        if self.ttl:
            pipe.expire(checkpoint_key, int(self.ttl))

        # Add to thread checkpoint list (most recent first)
        pipe.lpush(thread_key, checkpoint_id)
        if self.ttl:
            pipe.expire(thread_key, int(self.ttl))

        await pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the Valkey database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])

        writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

        # Get existing writes
        existing_data = await self.client.get(writes_key)
        existing_writes = orjson.loads(existing_data) if existing_data else []

        # Add new writes
        new_writes = self._serialize_writes_data(writes, task_id)
        existing_writes.extend(new_writes)

        # Store updated writes
        pipe = await self.client.pipeline()
        pipe.set(writes_key, orjson.dumps(existing_writes))
        if self.ttl:
            pipe.expire(writes_key, int(self.ttl))
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        # Find all checkpoint namespaces for this thread
        pattern = f"thread:{thread_id}:*"
        thread_keys = await self.client.keys(pattern)

        all_keys_to_delete = list(thread_keys)

        for thread_key in thread_keys:
            # Get all checkpoint IDs for this thread/namespace
            checkpoint_ids = await self.client.lrange(thread_key, 0, -1)

            # Extract namespace from thread key
            thread_key_str = (
                thread_key.decode() if isinstance(thread_key, bytes) else thread_key
            )
            _, _, checkpoint_ns = thread_key_str.split(":", 2)

            for checkpoint_id_bytes in checkpoint_ids:
                checkpoint_id = checkpoint_id_bytes.decode()
                checkpoint_key = self._make_checkpoint_key(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                writes_key = self._make_writes_key(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                all_keys_to_delete.extend([checkpoint_key, writes_key])

        if all_keys_to_delete:
            await self.client.delete(*all_keys_to_delete)

    # Sync methods that raise NotImplementedError
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Sync method not supported in AsyncValkeyCheckpointSaver."""
        raise NotImplementedError("Use aget_tuple() for async operations")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Sync method not supported in AsyncValkeyCheckpointSaver."""
        raise NotImplementedError("Use alist() for async operations")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Sync method not supported in AsyncValkeyCheckpointSaver."""
        raise NotImplementedError("Use aput() for async operations")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Sync method not supported in AsyncValkeyCheckpointSaver."""
        raise NotImplementedError("Use aput_writes() for async operations")

    def delete_thread(self, thread_id: str) -> None:
        """Sync method not supported in AsyncValkeyCheckpointSaver."""
        raise NotImplementedError("Use adelete_thread() for async operations")


__all__ = ["AsyncValkeyCheckpointSaver"]
