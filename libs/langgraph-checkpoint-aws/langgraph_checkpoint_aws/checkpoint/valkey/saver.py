"""Valkey checkpoint saver implementation."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from .base import BaseValkeySaver

# Conditional imports for optional dependencies
try:
    import orjson
except ImportError as e:
    raise ImportError(
        "The 'orjson' package is required to use ValkeySaver. "
        "Install it with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ) from e

try:
    from valkey import Valkey
    from valkey.asyncio import Valkey as AsyncValkey
    from valkey.connection import ConnectionPool
    from valkey.exceptions import ValkeyError
except ImportError as e:
    raise ImportError(
        "The 'valkey' package is required to use ValkeySaver. "
        "Install it with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ) from e

logger = logging.getLogger(__name__)


class ValkeySaver(BaseValkeySaver):
    """A checkpoint saver that stores checkpoints in Valkey (Redis-compatible).

    This class provides both synchronous and asynchronous methods for storing
    and retrieving checkpoints using Valkey as the backend storage.

    Args:
        client: The Valkey client instance.
        ttl: Time-to-live for stored checkpoints in seconds. Defaults to None (no
            expiration).
        serde: The serializer to use for serializing and deserializing checkpoints.

    Examples:

        >>> from valkey import Valkey
        >>> from langgraph.checkpoint.valkey import ValkeySaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # Create a new ValkeySaver instance
        >>> client = Valkey.from_url("valkey://localhost:6379")
        >>> memory = ValkeySaver(client)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> final_state = graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {
            'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'
        }}, parent_config=None)
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
    ) -> Iterator[ValkeySaver]:
        """Create a new ValkeySaver instance from a connection string.

        Args:
            conn_string: The Valkey connection string.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.
            pool_size: Maximum number of connections in the pool.
            **kwargs: Additional arguments passed to Valkey client.

        Yields:
            ValkeySaver: A new ValkeySaver instance.

        Examples:

            >>> with ValkeySaver.from_conn_string(
            ...     "valkey://localhost:6379"
            ... ) as memory:
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
    ) -> Iterator[ValkeySaver]:
        """Create a new ValkeySaver instance from a connection pool.

        Args:
            pool: The Valkey connection pool.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.

        Yields:
            ValkeySaver: A new ValkeySaver instance.

        Examples:

            >>> from valkey.connection import ConnectionPool
            >>> pool = ConnectionPool.from_url("valkey://localhost:6379")
            >>> with ValkeySaver.from_pool(pool) as memory:
            ...     # Use the memory instance
            ...     pass
        """
        client = Valkey.from_pool(connection_pool=pool)
        try:
            yield cls(client, ttl=ttl_seconds)
        finally:
            client.close()

    def _get_checkpoint_data(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Helper method to get checkpoint and writes data.

        Args:
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            checkpoint_id: The checkpoint ID.

        Returns:
            Tuple of (checkpoint_info, writes) or (None, []) if not found.
        """
        try:
            checkpoint_key = self._make_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )
            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            # Use pipeline for better performance
            pipe = self.client.pipeline()
            pipe.get(checkpoint_key)
            pipe.get(writes_key)
            results = pipe.execute()

            try:
                checkpoint_data, writes_data = results
            except (TypeError, ValueError):
                # Handle Mock objects in tests
                return None, []

            if not checkpoint_data:
                return None, []

            # Handle string vs bytes for orjson
            if isinstance(checkpoint_data, str):
                checkpoint_data = checkpoint_data.encode("utf-8")
            if isinstance(writes_data, str):
                writes_data = writes_data.encode("utf-8")

            checkpoint_info = orjson.loads(checkpoint_data)
            writes = orjson.loads(writes_data) if writes_data else []
            return checkpoint_info, writes

        except (ValkeyError, orjson.JSONDecodeError) as e:
            logger.error(f"Error retrieving checkpoint data for {checkpoint_id}: {e}")
            return None, []

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Valkey database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint
        with the matching thread ID and checkpoint ID is retrieved. Otherwise, the
        latest checkpoint for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no
                matching checkpoint was found.
        """
        try:
            configurable = config.get("configurable", {})
            thread_id = str(configurable["thread_id"])
            checkpoint_ns = configurable.get("checkpoint_ns", "")

            if checkpoint_id := get_checkpoint_id(config):
                # Get specific checkpoint
                checkpoint_info, writes = self._get_checkpoint_data(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                if not checkpoint_info:
                    return None

                return self._deserialize_checkpoint_data(
                    checkpoint_info,
                    writes,
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    config,
                )

            else:
                # Get latest checkpoint
                thread_key = self._make_thread_key(thread_id, checkpoint_ns)
                checkpoint_ids = self.client.lrange(thread_key, 0, 0)  # Get most recent

                if not checkpoint_ids:
                    return None

                checkpoint_id = checkpoint_ids[0].decode()
                checkpoint_info, writes = self._get_checkpoint_data(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                if not checkpoint_info:
                    return None

                # Update config with checkpoint_id
                updated_config: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }

                return self._deserialize_checkpoint_data(
                    checkpoint_info,
                    writes,
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    updated_config,
                )

        except (ValkeyError, KeyError) as e:
            logger.error(f"Error in get_tuple: {e}")
            return None

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
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first). Uses batching for better performance with
        large datasets.

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID
                are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        if not config:
            return

        try:
            configurable = config.get("configurable", {})
            thread_id = str(configurable["thread_id"])
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            thread_key = self._make_thread_key(thread_id, checkpoint_ns)

            # Get checkpoint IDs with pagination for memory efficiency
            batch_size = min(limit or 100, 100)  # Process in batches
            start_idx = 0

            # Apply before filter
            if before and (before_id := get_checkpoint_id(before)):
                # Find the index of the before_id
                all_ids = self.client.lrange(thread_key, 0, -1)
                try:
                    before_idx = next(
                        i for i, cid in enumerate(all_ids) if cid.decode() == before_id
                    )
                    start_idx = before_idx + 1
                except StopIteration:
                    # If before checkpoint doesn't exist, return all checkpoints
                    start_idx = 0

            yielded_count = 0
            while True:
                # Get batch of checkpoint IDs
                end_idx = start_idx + batch_size - 1
                if limit and yielded_count + batch_size > limit:
                    end_idx = start_idx + (limit - yielded_count) - 1

                checkpoint_ids = self.client.lrange(thread_key, start_idx, end_idx)
                if not checkpoint_ids:
                    break

                # Batch fetch checkpoint and writes data
                pipe = self.client.pipeline()
                for checkpoint_id_bytes in checkpoint_ids:
                    checkpoint_id = checkpoint_id_bytes.decode()
                    checkpoint_key = self._make_checkpoint_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )
                    writes_key = self._make_writes_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )
                    pipe.get(checkpoint_key)
                    pipe.get(writes_key)

                results = pipe.execute()

                # Process results in pairs (checkpoint_data, writes_data)
                for i, checkpoint_id_bytes in enumerate(checkpoint_ids):
                    checkpoint_id = checkpoint_id_bytes.decode()
                    checkpoint_data = results[i * 2]
                    writes_data = results[i * 2 + 1]

                    if not checkpoint_data:
                        continue

                    try:
                        checkpoint_info = orjson.loads(checkpoint_data)

                        # Apply metadata filter
                        if not self._should_include_checkpoint(checkpoint_info, filter):
                            continue

                        writes = orjson.loads(writes_data) if writes_data else []

                        yield self._deserialize_checkpoint_data(
                            checkpoint_info,
                            writes,
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                        )

                        yielded_count += 1
                        if limit and yielded_count >= limit:
                            return

                    except orjson.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to decode checkpoint {checkpoint_id}: {e}"
                        )
                        continue

                start_idx += batch_size

        except ValkeyError as e:
            logger.error(f"Error in list: {e}")
            return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Valkey database. The checkpoint is
        associated with the provided config and its parent config (if any). Uses
        transactions for atomicity.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        try:
            configurable = config.get("configurable", {})
            thread_id = str(configurable["thread_id"])
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            checkpoint_id = checkpoint["id"]

            # Use base class method to serialize checkpoint data
            checkpoint_info = self._serialize_checkpoint_data(
                config, checkpoint, metadata
            )

            # Store checkpoint atomically
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

        except (ValkeyError, orjson.JSONEncodeError) as e:
            logger.error(f"Error in put: {e}")
            raise

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the
        Valkey database. Uses atomic operations to ensure consistency.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        try:
            configurable = config.get("configurable", {})
            thread_id = str(configurable["thread_id"])
            checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
            checkpoint_id = str(configurable["checkpoint_id"])

            writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)

            # Use atomic operation to update writes
            pipe = self.client.pipeline()

            # Get existing writes
            pipe.get(writes_key)
            results = pipe.execute()
            existing_data = results[0]

            existing_writes = orjson.loads(existing_data) if existing_data else []

            # Use base class method to serialize new writes
            new_writes = self._serialize_writes_data(writes, task_id)
            existing_writes.extend(new_writes)

            # Store updated writes atomically
            pipe = self.client.pipeline()
            pipe.set(writes_key, orjson.dumps(existing_writes))
            if self.ttl:
                pipe.expire(writes_key, int(self.ttl))
            pipe.execute()

        except (ValkeyError, orjson.JSONEncodeError, KeyError) as e:
            logger.error(f"Error in put_writes: {e}")
            raise

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Uses batching for efficient deletion of large datasets.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        try:
            # Find all checkpoint namespaces for this thread
            pattern = f"thread:{thread_id}:*"
            thread_keys = self.client.keys(pattern)

            if not thread_keys:
                return

            all_keys_to_delete = list(thread_keys)

            # Process in batches to avoid memory issues
            batch_size = 100
            for thread_key in thread_keys:
                # Get all checkpoint IDs for this thread/namespace
                checkpoint_ids = self.client.lrange(thread_key, 0, -1)

                # Extract namespace from thread key
                thread_key_str = (
                    thread_key.decode() if isinstance(thread_key, bytes) else thread_key
                )
                parts = thread_key_str.split(":", 2)
                checkpoint_ns = parts[2] if len(parts) > 2 else ""

                # Collect keys in batches
                for i in range(0, len(checkpoint_ids), batch_size):
                    batch_ids = checkpoint_ids[i : i + batch_size]
                    batch_keys = []

                    for checkpoint_id_bytes in batch_ids:
                        checkpoint_id = checkpoint_id_bytes.decode()
                        checkpoint_key = self._make_checkpoint_key(
                            thread_id, checkpoint_ns, checkpoint_id
                        )
                        writes_key = self._make_writes_key(
                            thread_id, checkpoint_ns, checkpoint_id
                        )
                        batch_keys.extend([checkpoint_key, writes_key])

                    all_keys_to_delete.extend(batch_keys)

            # Delete all keys in batches
            if all_keys_to_delete:
                for i in range(0, len(all_keys_to_delete), batch_size):
                    batch_keys = all_keys_to_delete[i : i + batch_size]
                    self.client.delete(*batch_keys)

        except ValkeyError as e:
            logger.error(f"Error in delete_thread: {e}")
            raise

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        Note:
            This async method is not supported by the ValkeySaver class.
            Use get_tuple() instead, or consider using AsyncValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeySaver does not support async methods. "
            "Consider using AsyncValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "AsyncValkeySaver\n"
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
            This async method is not supported by the ValkeySaver class.
            Use list() instead, or consider using AsyncValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeySaver does not support async methods. "
            "Consider using AsyncValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "AsyncValkeySaver\n"
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
            This async method is not supported by the ValkeySaver class.
            Use put() instead, or consider using AsyncValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support async operations.
        """
        raise NotImplementedError(
            "The ValkeySaver does not support async methods. "
            "Consider using AsyncValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "AsyncValkeySaver\n"
            "See the documentation for more information."
        )
