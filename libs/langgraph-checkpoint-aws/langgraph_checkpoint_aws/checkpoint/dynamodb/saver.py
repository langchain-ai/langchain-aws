"""DynamoDB checkpoint saver for LangGraph."""

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import boto3
from botocore.config import Config
from langchain_core.runnables import RunnableConfig, run_in_executor
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
    )

from typing import cast

from .serialization import CheckpointSerializer
from .storage_strategy import StorageStrategy
from .unified_repository import UnifiedRepository
from .utils import create_dynamodb_client, create_s3_client

logger = logging.getLogger(__name__)


class S3OffloadConfigDict(TypedDict, total=False):
    """Type alias for S3 offload configuration
    Required when user needs to configure Saver, to offload large
    Checkpoints above threshold (350KB) to S3.
    """

    bucket_name: str
    endpoint_url: str | None


class DynamoDBSaver(BaseCheckpointSaver):
    """Saves and retrieves checkpoints using Amazon DynamoDB with S3 offloading.

    Args:
        table_name: Name of the DynamoDB table for checkpoints
        session: Pre-configured boto3 session instance for custom credentials
        region_name: AWS region name, needs to be used for DynamoDB & S3
        endpoint_url: Custom endpoint URL for the DynamoDB service
        boto_config: Botocore config object
        ttl_seconds: Optional TTL in seconds for automatic cleanup
        enable_checkpoint_compression: If True, compresses the checkpoint while
            saving
        s3_offload_config: Optional dict for S3 offload configuration.
            If provided, data >350KB will be stored in S3.
            Required key: "bucket_name" (str), to be used for checkpoint offloading
            Optional keys:
                "endpoint_url" (str) Custom endpoint URL for the S3 service
            Example:
            {
                "bucket_name": "my-checkpoint-bucket",
                "endpoint_url": "http://localhost:4566"
            }
    """

    def __init__(
        self,
        table_name: str,
        session: boto3.Session | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        boto_config: Config | None = None,
        ttl_seconds: int | None = None,
        enable_checkpoint_compression: bool = False,
        s3_offload_config: S3OffloadConfigDict | None = None,
    ) -> None:
        super().__init__()

        # Initialize DynamoDB client
        self.client = create_dynamodb_client(
            session=session,
            region_name=region_name,
            endpoint_url=endpoint_url,
            boto_config=boto_config,
        )

        # S3 configuration if provided
        s3_client_instance = None
        s3_bucket_name = None

        if s3_offload_config:
            # Validate required bucket_name
            if "bucket_name" not in s3_offload_config:
                raise ValueError(
                    "s3_offload_config must contain 'bucket_name' key. "
                    "Example: {'bucket_name': 'my-checkpoint-bucket'}"
                )

            s3_bucket_name = s3_offload_config["bucket_name"]

            s3_client_instance = create_s3_client(
                session=session,
                region_name=region_name,
                endpoint_url=s3_offload_config.get("endpoint_url"),
                boto_config=boto_config,
            )

        # Checkpoint table name for Dynamodb
        self.table_name = table_name

        # Initialize serializer with compression
        self.serializer = CheckpointSerializer(
            self.serde, enable_checkpoint_compression
        )

        # Initialize storage strategy using S3 and Dynamodb
        self.storage = StorageStrategy(
            dynamodb_client=self.client,
            table_name=self.table_name,
            s3_client=s3_client_instance,
            s3_bucket=s3_bucket_name,
            ttl_seconds=ttl_seconds,
        )

        # Initialize repository
        self.repo = UnifiedRepository(
            dynamodb_client=self.client,
            table_name=self.table_name,
            serializer=self.serializer,
            storage_strategy=self.storage,
            ttl_seconds=ttl_seconds,
        )

    def _get_checkpoint_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite]:
        """Retrieve pending writes for a given checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Namespace that groups related checkpoints
            checkpoint_id: Unique identifier for the checkpoint

        Returns:
            List of (task_id, channel, value) tuples
        """
        return self.repo.get_pending_writes(thread_id, checkpoint_ns, checkpoint_id)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple based on config.

        Args:
            config: Configuration containing thread_id, checkpoint_ns, and
                optionally checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        checkpoint_data = self.repo.get_checkpoint(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
        )

        if not checkpoint_data:
            return None

        # Build checkpoint config
        checkpoint_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                "checkpoint_id": checkpoint_data["checkpoint_id"],
            }
        }

        # Build parent config
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                    "checkpoint_id": checkpoint_data["parent_checkpoint_id"],
                }
            }
            if checkpoint_data["parent_checkpoint_id"]
            else None
        )

        # Get all the pending writes for given checkpointId
        pending_writes = self._get_checkpoint_pending_writes(
            thread_id,
            checkpoint_data["checkpoint_ns"],
            checkpoint_data["checkpoint_id"],
        )

        # Return tuple with pending writes
        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint_data["checkpoint"],
            metadata=checkpoint_data["metadata"],
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously retrieve a checkpoint tuple from DynamoDB.

        Args:
            config: Configuration containing thread_id, checkpoint_ns, and
                optionally checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise
        """
        return await run_in_executor(None, self.get_tuple, config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: "ChannelVersions",
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store
            metadata: Additional metadata for the checkpoint
            new_versions: New channel versions as of this write

        Returns:
            Updated configuration after storing the checkpoint

        Raises:
            ValueError: If required config fields are missing
            ClientError: If DynamoDB operation fails
        """

        thread_id = config["configurable"].get("thread_id")
        if thread_id is None:
            raise ValueError("Thread_id must be configured")

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        metadata_to_store = dict(metadata)
        config_metadata = config.get("metadata", {})
        if isinstance(config_metadata, dict):
            metadata_to_store.update(config_metadata)

        result = self.repo.put_checkpoint(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint=checkpoint,
            metadata=cast("CheckpointMetadata", cast(object, metadata_to_store)),
            parent_checkpoint_id=parent_checkpoint_id,
        )

        # Return stored checkpoint details
        return {
            "configurable": {
                "thread_id": result["thread_id"],
                "checkpoint_ns": result["checkpoint_ns"],
                "checkpoint_id": result["checkpoint_id"],
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: "ChannelVersions",
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store
            metadata: Additional metadata for the checkpoint
            new_versions: New channel versions as of this write (not used in
                DynamoDB implementation)

        Returns:
            Updated configuration after storing the checkpoint

        Raises:
            ValueError: If required config fields are missing
            ClientError: If DynamoDB operation fails
        """
        return await run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint
            writes: List of (channel, value) tuples to store
            task_id: Identifier for the task creating the writes
            task_path: Path of the task creating the writes (for nested task tracking)

        Raises:
            ClientError: If DynamoDB operation fails
        """
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if thread_id is None or checkpoint_id is None:
            raise ValueError("Runnable config must contain thread_id and checkpoint_id")

        self.repo.put_writes(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            writes=writes,
            task_id=task_id,
            task_path=task_path,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint
            writes: List of (channel, value) tuples to store
            task_id: Identifier for the task creating the writes
            task_path: Path of the task creating the writes (for nested task tracking)

        Raises:
            ValueError: If writes contain invalid tuples or required parameters
                are missing
            ClientError: If DynamoDB operation fails
        """
        return await run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the data store.

        Args:
            config: The config to use for listing the checkpoints.
                Must contain thread_id.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified
                checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

        Raises:
            ValueError: If config is None or doesn't contain thread_id.

        Examples:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoints = list(saver.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]
        """
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        if thread_id is None:
            raise ValueError("Runnable config must contain thread_id")

        checkpoint_ns = (
            config.get("configurable", {}).get("checkpoint_ns") if config else None
        )

        before_checkpoint_id = None
        if before is not None:
            before_checkpoint_id = before.get("configurable", {}).get("checkpoint_id")

        items_yielded = 0

        # Get list of checkpoints
        for checkpoint_data in self.repo.list_checkpoints(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            before_checkpoint_id=before_checkpoint_id,
            limit=limit,  # filter is applied get all and filter at client side,
            filter=filter,
        ):
            checkpoint_config: RunnableConfig = {
                "configurable": {
                    "thread_id": checkpoint_data["thread_id"],
                    "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                    "checkpoint_id": checkpoint_data["checkpoint_id"],
                }
            }

            parent_config: RunnableConfig | None = (
                {
                    "configurable": {
                        "thread_id": checkpoint_data["thread_id"],
                        "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                        "checkpoint_id": checkpoint_data["parent_checkpoint_id"],
                    }
                }
                if checkpoint_data["parent_checkpoint_id"]
                else None
            )

            # Get pending writes for checkpoint
            pending_writes = self._get_checkpoint_pending_writes(
                checkpoint_data["thread_id"],
                checkpoint_data["checkpoint_ns"],
                checkpoint_data["checkpoint_id"],
            )

            yield CheckpointTuple(
                config=checkpoint_config,
                checkpoint=checkpoint_data["checkpoint"],
                metadata=checkpoint_data["metadata"],
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

            items_yielded += 1

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config: Configuration object passed to `self.list`.
            filter: Optional filter dictionary.
            before: Optional parameter to limit results before a given checkpoint.
            limit: Optional maximum number of results to return.

        Returns:
            AsyncIterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.

        Raises:
            ClientError: If DynamoDB operation fails.
        """
        # Fetch all the checkpoints their id and ns
        checkpoint_info = self.repo.get_thread_checkpoint_info(thread_id)

        if not checkpoint_info:
            logger.info(f"No checkpoints found for thread_id={thread_id}")
            return

        # Delete writes first as they are child to checkpoints
        self.repo.delete_thread_writes(thread_id, checkpoint_info)

        # Delete checkpoint as writes are deleted for the checkpoints above.
        self.repo.delete_thread_checkpoints(thread_id, checkpoint_info)
        logger.info(f"Successfully deleted all data for thread_id={thread_id}")

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all checkpoints and writes associated
        with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.

        Raises:
            ClientError: If DynamoDB operation fails.
        """
        return await run_in_executor(None, self.delete_thread, thread_id)
