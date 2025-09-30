"""Valkey checkpoint saver implementation."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import orjson
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
)
from valkey import Valkey
from valkey.asyncio import Valkey as AsyncValkey
from valkey.connection import ConnectionPool

from .base import BaseValkeyCheckpointSaver


class ValkeyCheckpointSaver(BaseValkeyCheckpointSaver):
    """A checkpoint saver that stores checkpoints in Valkey (Redis-compatible).

    This class provides both synchronous and asynchronous methods for storing
    and retrieving checkpoints using Valkey as the backend storage.

    Args:
        client: The Valkey client instance.
        ttl: Time-to-live for stored checkpoints in seconds. Defaults to None (no expiration).
        serde: The serializer to use for serializing and deserializing checkpoints.

    Examples:

        >>> from valkey import Valkey
        >>> from langgraph.checkpoint.valkey import ValkeyCheckpointSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # Create a new ValkeyCheckpointSaver instance
        >>> client = Valkey.from_url("valkey://localhost:6379")
        >>> memory = ValkeyCheckpointSaver(client)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'}}, parent_config=None)
    """

    def __init__(
        self,
        client: Valkey | AsyncValkey,
        *,
        ttl: float | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(client, ttl=ttl, serde=serde)
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        ttl_seconds: float | None = None,
        pool_size: int = 10,
        **kwargs: Any,
    ) -> Iterator[ValkeyCheckpointSaver]:
        """Create a new ValkeyCheckpointSaver instance from a connection string.

        Args:
            conn_string: The Valkey connection string.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.
            pool_size: Maximum number of connections in the pool.
            **kwargs: Additional arguments passed to Valkey client.

        Yields:
            ValkeyCheckpointSaver: A new ValkeyCheckpointSaver instance.

        Examples:

            >>> with ValkeyCheckpointSaver.from_conn_string("valkey://localhost:6379") as memory:
            ...     # Use the memory instance
            ...     pass
        """
        # Create connection pool first, then client
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
    ) -> Iterator[ValkeyCheckpointSaver]:
        """Create a new ValkeyCheckpointSaver instance from a connection pool.

        Args:
            pool: The Valkey connection pool.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.

        Yields:
            ValkeyCheckpointSaver: A new ValkeyCheckpointSaver instance.

        Examples:

            >>> from valkey.connection import ConnectionPool
            >>> pool = ConnectionPool.from_url("valkey://localhost:6379")
            >>> with ValkeyCheckpointSaver.from_pool(pool) as memory:
            ...     # Use the memory instance
            ...     pass
        """
        client = Valkey.from_pool(connection_pool=pool)
        try:
            yield cls(client, ttl=ttl_seconds)
        finally:
            client.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

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

            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            checkpoint_info = orjson.loads(checkpoint_data)
            writes_data = self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

        else:
            # Get latest checkpoint
            thread_key = self._make_thread_key(thread_id, checkpoint_ns)
            checkpoint_ids = self.client.lrange(thread_key, 0, 0)  # Get most recent

            if not checkpoint_ids:
                return None

            checkpoint_id = checkpoint_ids[0].decode()
            checkpoint_key = self._make_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )
            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            checkpoint_info = orjson.loads(checkpoint_data)
            writes_data = self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

            # Update config with checkpoint_id
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        # Use base class method to deserialize
        return self._deserialize_checkpoint_data(
            checkpoint_info, writes, thread_id, checkpoint_ns, checkpoint_id, config
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Valkey database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        if not config:
            return

        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        thread_key = self._make_thread_key(thread_id, checkpoint_ns)

        # Get all checkpoint IDs for this thread
        checkpoint_ids = self.client.lrange(thread_key, 0, -1)

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

            checkpoint_data = self.client.get(checkpoint_key)
            if not checkpoint_data:
                continue

            checkpoint_info = orjson.loads(checkpoint_data)

            # Apply metadata filter using base class method
            if not self._should_include_checkpoint(checkpoint_info, filter):
                continue

            writes_data = self.client.get(writes_key)
            writes = orjson.loads(writes_data) if writes_data else []

            # Use base class method to deserialize
            checkpoint_tuple = self._deserialize_checkpoint_data(
                checkpoint_info, writes, thread_id, checkpoint_ns, checkpoint_id
            )
            yield checkpoint_tuple

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

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

        # Use base class method to serialize checkpoint data
        checkpoint_info = self._serialize_checkpoint_data(config, checkpoint, metadata)

        # Store checkpoint
        checkpoint_key = self._make_checkpoint_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        thread_key = self._make_thread_key(thread_id, checkpoint_ns)

        pipe = self.client.pipeline()
        pipe.set(checkpoint_key, orjson.dumps(checkpoint_info))
        if self.ttl:
            pipe.expire(checkpoint_key, int(self.ttl))

        # Add to thread checkpoint list (most recent first)
        pipe.lpush(thread_key, checkpoint_id)
        if self.ttl:
            pipe.expire(thread_key, int(self.ttl))

        pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

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
        existing_data = self.client.get(writes_key)
        existing_writes = orjson.loads(existing_data) if existing_data else []

        # Use base class method to serialize new writes
        new_writes = self._serialize_writes_data(writes, task_id)
        existing_writes.extend(new_writes)

        # Store updated writes
        pipe = self.client.pipeline()
        pipe.set(writes_key, orjson.dumps(existing_writes))
        if self.ttl:
            pipe.expire(writes_key, int(self.ttl))
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        # Find all checkpoint namespaces for this thread
        pattern = f"thread:{thread_id}:*"
        thread_keys = self.client.keys(pattern)

        all_keys_to_delete = list(thread_keys)

        for thread_key in thread_keys:
            # Get all checkpoint IDs for this thread/namespace
            checkpoint_ids = self.client.lrange(thread_key, 0, -1)

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
            self.client.delete(*all_keys_to_delete)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        Note:
            This async method is not supported by the ValkeyCheckpointSaver class.
            Use get_tuple() instead, or consider using AsyncValkeyCheckpointSaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyCheckpointSaver does not support async methods. "
            "Consider using AsyncValkeyCheckpointSaver instead.\n"
            "from langgraph.checkpoint.valkey.aio import AsyncValkeyCheckpointSaver\n"
            "See the documentation for more information."
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

        Note:
            This async method is not supported by the ValkeyCheckpointSaver class.
            Use list() instead, or consider using AsyncValkeyCheckpointSaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyCheckpointSaver does not support async methods. "
            "Consider using AsyncValkeyCheckpointSaver instead.\n"
            "from langgraph.checkpoint.valkey.aio import AsyncValkeyCheckpointSaver\n"
            "See the documentation for more information."
        )
        yield  # This line is needed to make this an async generator

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        Note:
            This async method is not supported by the ValkeyCheckpointSaver class.
            Use put() instead, or consider using AsyncValkeyCheckpointSaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeyCheckpointSaver does not support async methods. "
            "Consider using AsyncValkeyCheckpointSaver instead.\n"
            "from langgraph.checkpoint.valkey.aio import AsyncValkeyCheckpointSaver\n"
            "See the documentation for more information."
        )
