"""Async Valkey checkpoint saver implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
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
from .utils import aset_client_info

# Conditional imports for optional dependencies
try:
    import orjson
except ImportError as e:
    raise ImportError(
        "The 'orjson' package is required to use AsyncValkeySaver. "
        "Install it with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ) from e

try:
    from valkey.asyncio import Valkey as AsyncValkey
    from valkey.asyncio.connection import ConnectionPool as AsyncConnectionPool
    from valkey.exceptions import ValkeyError
except ImportError as e:
    raise ImportError(
        "The 'valkey' package is required to use AsyncValkeySaver. "
        "Install it with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ) from e

logger = logging.getLogger(__name__)


class AsyncValkeySaver(BaseValkeySaver):
    """An async checkpoint saver that stores checkpoints in Valkey (Redis-compatible).

    This class provides asynchronous methods for storing and retrieving checkpoints
    using Valkey as the backend storage.

    Args:
        client: The AsyncValkey client instance.
        ttl: Time-to-live for stored checkpoints in seconds.
            Defaults to None (no expiration).
        serde: The serializer to use for serializing and deserializing checkpoints.

    Examples:

        >>> from langgraph_checkpoint_aws.checkpoint.valkey import (
        ...     AsyncValkeySaver,
        ... )
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # Create a new AsyncValkeySaver instance using context manager
        >>> async with AsyncValkeySaver.from_conn_string(
        ...     "valkey://localhost:6379"
        ... ) as memory:
        >>>     graph = builder.compile(checkpointer=memory)
        >>>     config = {"configurable": {"thread_id": "1"}}
        >>>     state = await graph.aget_state(config)
        >>>     result = await graph.ainvoke(3, config)
        >>>     final_state = await graph.aget_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {
            'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'
        }}, parent_config=None)

    Note:
        The example output shows the state snapshot with a long config line that
        exceeds normal formatting limits for demonstration purposes.
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
        # It should be called in async factory methods like from_conn_string
        # and from_pool

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
    ) -> AsyncIterator[AsyncValkeySaver]:
        """Create a new AsyncValkeySaver instance from a connection string.

        Args:
            conn_string: The Valkey connection string.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.
            serde: The serializer to use for serializing and deserializing checkpoints.
            pool_size: Maximum number of connections in the pool.
            **kwargs: Additional arguments passed to AsyncValkey client.

        Yields:
            AsyncValkeySaver: A new AsyncValkeySaver instance.

        Examples:

            >>> async with AsyncValkeySaver.from_conn_string(
            ...     "valkey://localhost:6379"
            ... ) as memory:
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
    ) -> AsyncIterator[AsyncValkeySaver]:
        """Create a new AsyncValkeySaver instance from a connection pool.

        Args:
            pool: The Valkey async connection pool.
            ttl_seconds: Time-to-live for stored checkpoints in seconds.

        Yields:
            AsyncValkeySaver: A new AsyncValkeySaver instance.

        Examples:

            >>> from valkey.asyncio.connection import (
            ...     ConnectionPool as AsyncConnectionPool,
            ... )
            >>> pool = AsyncConnectionPool.from_url("valkey://localhost:6379")
            >>> async with AsyncValkeySaver.from_pool(pool) as memory:
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

    async def _get_checkpoint_data(
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
            results = await pipe.execute()

            # Ensure we have exactly 2 results
            if not results or len(results) != 2:
                logger.warning(
                    f"Unexpected pipeline results for {checkpoint_id}: {results}"
                )
                return None, []

            checkpoint_data, writes_data = results
            if not checkpoint_data:
                return None, []

            checkpoint_info = orjson.loads(checkpoint_data)
            if writes_data:
                if isinstance(writes_data, str):
                    writes_data = writes_data.encode("utf-8")
                writes = orjson.loads(writes_data)
            else:
                writes = []
            return checkpoint_info, writes

        except (
            ValkeyError,
            orjson.JSONDecodeError,
            ValueError,
            ConnectionError,
            asyncio.TimeoutError,
        ) as e:
            logger.error(f"Error retrieving checkpoint data for {checkpoint_id}: {e}")
            return None, []

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

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
                checkpoint_info, writes = await self._get_checkpoint_data(
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
                checkpoint_ids = await self.client.lrange(
                    thread_key, 0, 0
                )  # Get most recent

                if not checkpoint_ids:
                    return None

                checkpoint_id = checkpoint_ids[0].decode()
                checkpoint_info, writes = await self._get_checkpoint_data(
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
            logger.error(f"Error in aget_tuple: {e}")
            return None

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
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).
        Uses batching for better performance with large datasets.

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID
                are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An async iterator of checkpoint tuples.
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
                all_ids = await self.client.lrange(thread_key, 0, -1)
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

                checkpoint_ids = await self.client.lrange(
                    thread_key, start_idx, end_idx
                )
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

                results = await pipe.execute()

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
            logger.error(f"Error in alist: {e}")
            return

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

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

            # Serialize checkpoint data
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

            await pipe.execute()

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        except (ValkeyError, orjson.JSONEncodeError) as e:
            logger.error(f"Error in aput: {e}")
            raise

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the
        Valkey database.
        Uses atomic operations to ensure consistency.

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

            # Get existing writes first
            existing_data = await self.client.get(writes_key)

            existing_writes = []
            if existing_data:
                try:
                    # Handle string vs bytes for orjson
                    if isinstance(existing_data, str):
                        existing_data = existing_data.encode("utf-8")
                    elif not isinstance(existing_data, (bytes, bytearray, memoryview)):
                        # Handle other types (like Mock objects) by converting to
                        # JSON string first
                        try:
                            existing_data = orjson.dumps(existing_data)
                        except (TypeError, ValueError):
                            existing_data = b"[]"  # Default to empty array

                    parsed_data = orjson.loads(existing_data)
                    # Ensure we have a list
                    if isinstance(parsed_data, list):
                        existing_writes = parsed_data
                    else:
                        existing_writes = []
                except (orjson.JSONDecodeError, TypeError, ValueError):
                    existing_writes = []

            # Add new writes
            new_writes = self._serialize_writes_data(writes, task_id)
            existing_writes.extend(new_writes)

            # Store updated writes atomically
            pipe = self.client.pipeline()
            pipe.set(writes_key, orjson.dumps(existing_writes))
            if self.ttl:
                pipe.expire(writes_key, int(self.ttl))
            await pipe.execute()

        except (ValkeyError, orjson.JSONEncodeError, KeyError) as e:
            logger.error(f"Error in aput_writes: {e}")
            raise

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Uses batching for efficient deletion of large datasets.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        try:
            # Find all checkpoint namespaces for this thread
            pattern = f"thread:{thread_id}:*"
            thread_keys = await self.client.keys(pattern)

            if not thread_keys:
                return

            all_keys_to_delete = list(thread_keys)

            # Process in batches to avoid memory issues
            batch_size = 100
            for thread_key in thread_keys:
                # Get all checkpoint IDs for this thread/namespace
                checkpoint_ids = await self.client.lrange(thread_key, 0, -1)

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
                    await self.client.delete(*batch_keys)

        except ValkeyError as e:
            logger.error(f"Error in adelete_thread: {e}")
            raise

    # Sync methods that raise NotImplementedError
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database synchronously.

        Note:
            This sync method is not supported by the AsyncValkeySaver class.
            Use aget_tuple() instead, or consider using ValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeySaver does not support sync methods. "
            "Consider using ValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "ValkeySaver\n"
            "See the documentation for more information."
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database synchronously.

        Note:
            This sync method is not supported by the AsyncValkeySaver class.
            Use alist() instead, or consider using ValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeySaver does not support sync methods. "
            "Consider using ValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "ValkeySaver\n"
            "See the documentation for more information."
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database synchronously.

        Note:
            This sync method is not supported by the AsyncValkeySaver class.
            Use aput() instead, or consider using ValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeySaver does not support sync methods. "
            "Consider using ValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "ValkeySaver\n"
            "See the documentation for more information."
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint synchronously.

        Note:
            This sync method is not supported by the AsyncValkeySaver class.
            Use aput_writes() instead, or consider using ValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeySaver does not support sync methods. "
            "Consider using ValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "ValkeySaver\n"
            "See the documentation for more information."
        )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID synchronously.

        Note:
            This sync method is not supported by the AsyncValkeySaver class.
            Use adelete_thread() instead, or consider using ValkeySaver.

        Raises:
            NotImplementedError: Always, as this class doesn't support sync operations.
        """
        raise NotImplementedError(
            "The AsyncValkeySaver does not support sync methods. "
            "Consider using ValkeySaver instead.\n"
            "from langgraph_checkpoint_aws.checkpoint.valkey import "
            "ValkeySaver\n"
            "See the documentation for more information."
        )


__all__ = ["AsyncValkeySaver"]
