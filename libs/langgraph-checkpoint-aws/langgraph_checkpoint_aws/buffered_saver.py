import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class BufferedCheckpointSaver(BaseCheckpointSaver):
    """
    A wrapper for checkpoint savers that defers persistence until flushed.
    Supports both sync and async execution.
    Not thread-safe.

    This wrapper can wrap any BaseCheckpointSaver implementation (DynamoDBSaver,
    AgentCoreMemorySaver, ValkeySaver, etc.) to reduce API calls by buffering
    checkpoint operations and only persisting the state when explicitly flushed.

    The checkpointer can be accessed from the RunnableConfig in any node, allowing
    users to flush whenever they want during graph execution.

    Args:
        saver: The underlying checkpoint saver

    Example:
        ```python
        # Option 1: Context manager (auto-flushing on context exit)
        # Recommendation: Use this for basic use cases

        ## Asynchronous version:
        saver = AgentCoreMemorySaver(memory_id, region_name="us-west-2")
        buffered = BufferedCheckpointSaver(saver)
        graph = create_react_agent(model=model, tools=tools, checkpointer=buffered)
        async with buffered.aflush_on_exit():
            response = await graph.ainvoke({"messages": [...]}, config)

        ## Synchronous version:
        saver = AgentCoreMemorySaver(memory_id, region_name="us-west-2")
        buffered = BufferedCheckpointSaver(saver)
        graph = create_react_agent(model=model, tools=tools, checkpointer=buffered)
        with buffered.flush_on_exit():
            response = graph.invoke({"messages": [...]}, config)

        # Option 2: Flush manually within a node
        # Recommendation: Use this for fine-grained control over persistence timing

        ## Asynchronous version:
        async def my_node(state, config: RunnableConfig):
            # Do some work ...
            buffered = config["configurable"].get("checkpointer")
            if buffered:
                await buffered.aflush()
            # Do some work ...
            return state

        compiled_graph = ... # Build your compiled graph

        async with buffered.aflush_on_exit():
            response = await compiled_graph.ainvoke({...}, config)

        ## Synchronous version:
        def my_node(state, config: RunnableConfig):
            # Do some work ...
            buffered = config["configurable"].get("checkpointer")
            if buffered:
                buffered.flush()
            # Do some work ...
            return state

        compiled_graph = ... # Build your compiled graph

        with buffered.flush_on_exit():
            response = compiled_graph.invoke({...}, config)
        ```
    """

    def __init__(
        self,
        saver: BaseCheckpointSaver,
    ) -> None:
        """
        Initialize the BufferedCheckpointSaver.

        Args:
            saver: The underlying checkpoint saver.
        """
        super().__init__(serde=saver.serde)

        self._saver = saver
        self._last_config: RunnableConfig | None = None

        self._pending_checkpoint: (
            tuple[RunnableConfig, Checkpoint, CheckpointMetadata, ChannelVersions]
            | None
        ) = None
        self._pending_writes: list[
            tuple[RunnableConfig, Sequence[tuple[str, Any]], str, str]
        ] = []

    @property
    def saver(self) -> BaseCheckpointSaver:
        """Return the underlying checkpoint saver."""
        return self._saver

    @property
    def has_buffered_checkpoint(self) -> bool:
        """Return True if there is a buffered checkpoint."""
        return self._pending_checkpoint is not None

    @property
    def has_buffered_writes(self) -> bool:
        """Return True if there are buffered writes."""
        return len(self._pending_writes) > 0

    @property
    def is_empty(self) -> bool:
        """Return True if there is no buffered data (checkpoint or writes)."""
        return not self.has_buffered_checkpoint and not self.has_buffered_writes

    @contextmanager
    def flush_on_exit(self):
        """
        Context manager for flushing buffered checkpoint and writes on exit,
        including on exception.

        Example:
            ```python
            checkpointer = BufferedCheckpointSaver(base_checkpointer)

            with checkpointer.flush_on_exit():
                response = graph.invoke({"messages": [...]}, config)
            # Checkpoint and writes are automatically flushed on exit
            ```
        """
        try:
            yield self
        finally:
            self.flush()

    @asynccontextmanager
    async def aflush_on_exit(self):
        """
        Async context manager for flushing buffered checkpoint and writes on exit,
        including on exception.

        Example:
            ```python
            checkpointer = BufferedCheckpointSaver(base_checkpointer)

            async with checkpointer.aflush_on_exit():
                response = await graph.ainvoke({"messages": [...]}, config)
            # Checkpoint and writes are automatically flushed on exit
            ```
        """
        try:
            yield self
        finally:
            await self.aflush()

    def flush(self) -> RunnableConfig | None:
        """
        Persist all buffered checkpoints and writes to the underlying storage.

        Returns:
            The config from the last persisted checkpoint, or None if nothing to flush.
        """
        result = None

        if self._pending_checkpoint is not None:
            config, checkpoint, metadata, new_versions = self._pending_checkpoint
            result = self._saver.put(config, checkpoint, metadata, new_versions)
            self._pending_checkpoint = None

        for config, writes, task_id, task_path in self._pending_writes:
            self._saver.put_writes(config, writes, task_id, task_path)
        self._pending_writes = []

        return result

    async def aflush(self) -> RunnableConfig | None:
        """
        Async version of flush.
        """
        result = None

        # Persist the checkpoint first
        if self._pending_checkpoint is not None:
            config, checkpoint, metadata, new_versions = self._pending_checkpoint
            result = await self._saver.aput(config, checkpoint, metadata, new_versions)
            self._pending_checkpoint = None

        # Then persist all pending writes
        for config, writes, task_id, task_path in self._pending_writes:
            await self._saver.aput_writes(config, writes, task_id, task_path)
        self._pending_writes = []

        return result

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Buffer a checkpoint.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store
            metadata: Additional metadata for the checkpoint
            new_versions: New channel versions as of this write (not used in
                DynamoDB implementation)

        Returns:
            Updated configuration after storing the checkpoint
        """
        self._pending_checkpoint = (config, checkpoint, metadata, new_versions)
        self._last_config = self._update_runnable_config(config, checkpoint)
        return self._last_config

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put."""
        # NOTE: No need for async execution as this is in-memory
        return self.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Buffer writes.

        Args:
            config: Configuration for the writes.
            writes: The writes to store
            task_id: The task id
            task_path: The task path
        """
        self._pending_writes.append((config, writes, task_id, task_path))

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes."""
        # NOTE: No need for async execution as this is in-memory
        self.put_writes(config, writes, task_id, task_path)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint, checking buffer first."""
        if self._pending_checkpoint:
            buffered_config, checkpoint, metadata, _ = self._pending_checkpoint
            buffered_configurable = buffered_config.get("configurable", {})
            requested_configurable = config.get("configurable", {})

            buffered_thread_id = buffered_configurable.get("thread_id")
            buffered_checkpoint_ns = buffered_configurable.get("checkpoint_ns")
            buffered_checkpoint_id = checkpoint.get("id")

            requested_thread_id = requested_configurable.get("thread_id")
            requested_checkpoint_ns = requested_configurable.get("checkpoint_ns")
            requested_checkpoint_id = requested_configurable.get("checkpoint_id")

            if (
                requested_thread_id == buffered_thread_id
                and requested_checkpoint_ns == buffered_checkpoint_ns
                and (
                    not requested_checkpoint_id
                    or requested_checkpoint_id == buffered_checkpoint_id
                )
            ):
                pending_writes: list[tuple[str, str, Any]] = []
                for write_config, writes, task_id, _ in self._pending_writes:
                    write_configurable = write_config.get("configurable", {})
                    write_thread_id = write_configurable.get("thread_id")
                    write_checkpoint_ns = write_configurable.get("checkpoint_ns")
                    write_checkpoint_id = write_configurable.get("checkpoint_id")

                    if (
                        write_thread_id == buffered_thread_id
                        and write_checkpoint_ns == buffered_checkpoint_ns
                        and write_checkpoint_id == buffered_checkpoint_id
                    ):
                        for channel, value in writes:
                            pending_writes.append((task_id, channel, value))

                return CheckpointTuple(
                    config=self._update_runnable_config(buffered_config, checkpoint),
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=self._update_parent_runnable_config(buffered_config),
                    pending_writes=pending_writes,
                )

        return self._saver.get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple."""
        if self._pending_checkpoint:
            result = self.get_tuple(config)
            if result:
                return result
        return await self._saver.aget_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints.

        Args:
            config: Configuration for the checkpoints.
            filter: Additional filtering criteria for metadata.
            before: Only checkpoints before the specified checkpoint ID are returned.
            limit: The maximum number of checkpoints to return.

        Returns:
            Iterator of checkpoint tuples.
        """
        return self._saver.list(config, filter=filter, before=before, limit=limit)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list."""
        async for item in self._saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    def get_next_version(self, current: str | int | None, channel: None = None) -> str:
        """Generate next version string. Delegates to the underlying saver."""
        return self._saver.get_next_version(current, channel)

    def clear(self) -> None:
        """Discard all buffered checkpoints and writes without persisting."""
        self._pending_checkpoint = None
        self._pending_writes = []
        self._last_config = None

    def _update_runnable_config(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        """Update the runnable config with the checkpoint id."""
        configurable = config.get("configurable", {})

        result_configurable: dict[str, Any] = {
            **configurable,
            "thread_id": configurable.get("thread_id"),
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint["id"],
        }

        return {"configurable": result_configurable}

    def _update_parent_runnable_config(
        self, config: RunnableConfig
    ) -> RunnableConfig | None:
        """Update the parent runnable config with the parent checkpoint id."""
        configurable = config.get("configurable", {})
        parent_id = configurable.get("checkpoint_id")
        if not parent_id:
            return None

        result_configurable: dict[str, Any] = {
            **configurable,
            "thread_id": configurable.get("thread_id"),
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": parent_id,
        }

        return {"configurable": result_configurable}
