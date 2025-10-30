"""Unified repository for checkpoint and writes operations."""

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError
from langgraph.checkpoint.base import WRITES_IDX_MAP, Checkpoint, CheckpointMetadata

from .serialization import CheckpointSerializer, CompressionType
from .storage_strategy import StorageLocation, StorageStrategy

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_dynamodb.type_defs import (
        KeysAndAttributesTypeDef,
        WriteRequestTypeDef,
    )
else:
    DynamoDBClient = object

logger = logging.getLogger(__name__)


# ========== Key Generation Functions ==========


def _checkpoint_pk(thread_id: str) -> str:
    """Generate partition key for checkpoint metadata."""
    return f"CHECKPOINT_{thread_id}"


def _checkpoint_sk(checkpoint_ns: str, checkpoint_id: str) -> str:
    """Generate sort key for checkpoint metadata."""
    return f"{checkpoint_ns}#{checkpoint_id}"


def _checkpoint_ref_pk(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Generate partition key for checkpoint payload as reference item."""
    return f"CHUNK_{thread_id}#{checkpoint_ns}#{checkpoint_id}"


def _checkpoint_s3_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Generate S3 key for checkpoint payload."""
    return f"{thread_id}/checkpoints/{checkpoint_ns}/{checkpoint_id}"


def _writes_pk(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Generate partition key for writes metadata."""
    return f"WRITES_{thread_id}#{checkpoint_ns}#{checkpoint_id}"


def _writes_sk(task_id: str, idx: int) -> str:
    """Generate sort key for writes metadata."""
    return f"{task_id}#{idx}"


def _writes_ref_pk(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: str
) -> str:
    """Generate partition key for write payload as reference item."""
    return f"CHUNK_{thread_id}#{checkpoint_ns}#{checkpoint_id}#{task_id}#{idx}"


def _writes_s3_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: str
) -> str:
    """Generate S3 key for write payload."""
    return (
        f"{thread_id}/checkpoints/{checkpoint_ns}/{checkpoint_id}/"
        f"writes/{task_id}/{idx}"
    )


class UnifiedRepository:
    """
    Unified repository handling checkpoints and writes with
    s3 offloading capability.
    """

    def __init__(
        self,
        dynamodb_client: "DynamoDBClient",
        table_name: str,
        serializer: CheckpointSerializer,
        storage_strategy: StorageStrategy,
        ttl_seconds: int | None = None,
    ):
        """Initialize unified repository.

        Args:
            dynamodb_client: DynamoDB client for table operations
            table_name: Name of the checkpoint table
            serializer: Checkpoint serializer for data serialization
            storage_strategy: Storage strategy for strategic offloading
            ttl_seconds: Optional TTL in seconds for automatic cleanup
        """
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.serializer = serializer
        self.storage = storage_strategy
        self.ttl_seconds = ttl_seconds

    def _determine_storage_location(self, payload_data: bytes) -> str:
        """Determine storage location based on payload size.

        Args:
            payload_data: Serialized payload data

        Returns:
            Storage location: "S3" or "DYNAMODB"
        """
        return "S3" if self.storage.should_offload_to_s3(payload_data) else "DYNAMODB"

    # ========== Checkpoint Operations ==========

    def _find_checkpoints_with_valid_payload(
        self,
        query_params: dict,
        limit: int | None = None,
        filter_fn: Callable[[dict, bytes], bool] | None = None,
    ) -> Iterator[tuple[dict, bytes]]:
        """Find checkpoints with valid payload up to limit.

        Args:
            limit: Maximum number of valid checkpoints to return (None = no limit)
            filter_fn: Optional callback to filter (base_item, payload_data) tuples

        Returns:
            List of (base_item, payload_data) tuples
        """

        # Only use limit if no filter function is provided
        use_limit = filter_fn is None and limit is not None
        if use_limit:
            query_params["Limit"] = limit

        results_count = 0
        first_attempt = True

        while True:
            response = self.dynamodb_client.query(**query_params)
            items = response.get("Items", [])
            if not items:
                break

            for base_item in items:
                if limit is not None and results_count >= limit:
                    return

                payload = self.storage.retrieve_data(
                    base_item["ref_key"]["S"],
                    cast(StorageLocation, base_item["ref_loc"]["S"]),
                )
                if payload:
                    # Apply custom filter if provided
                    if filter_fn is None or filter_fn(base_item, payload):
                        yield base_item, payload
                        results_count += 1
                        if limit is not None and results_count >= limit:
                            return

            # No more pages available
            if "LastEvaluatedKey" not in response:
                break

            # First attempt with limit didn't yield enough valid results, remove limit
            # for subsequent queries
            if (
                first_attempt
                and use_limit
                and limit is not None
                and results_count < limit
            ):
                query_params.pop("Limit", None)
                first_attempt = False

            query_params["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    def get_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        checkpoint_id: str | None = None,
    ) -> dict | None:
        """Retrieve a checkpoint from storage.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace (default: "")
            checkpoint_id: Optional specific checkpoint ID (if None, gets latest)

        Returns:
            Dictionary containing checkpoint data, metadata, and identifiers,
            or None if not found

        Raises:
            ClientError: If retrieval operation fails
        """
        try:
            pk = _checkpoint_pk(thread_id)

            # Get checkpoint metadata from base table
            projection = (
                "id, ns, #t, ref_key, ref_loc, compression, parent_checkpoint_id"
            )
            expr_attr_names = {"#t": "type"}

            if checkpoint_id:
                # Get specific checkpoint by provided checkpoint_id

                sk = _checkpoint_sk(checkpoint_ns, checkpoint_id)
                response = self.dynamodb_client.get_item(
                    TableName=self.table_name,
                    Key={"PK": {"S": pk}, "SK": {"S": sk}},
                    ProjectionExpression=projection,
                    ExpressionAttributeNames=expr_attr_names,
                )
                if "Item" not in response:
                    return None

                base_item = response["Item"]
                payload_data = self.storage.retrieve_data(
                    base_item["ref_key"]["S"],
                    cast(StorageLocation, base_item["ref_loc"]["S"]),
                )
                if not payload_data:
                    logger.error(
                        f"Payload not found for checkpoint: "
                        f"checkpoint_ns={base_item['ns']['S']}, "
                        f"checkpoint_id={base_item['id']['S']}, "
                    )
                    return None
            else:
                # Latest checkpoint - use optimized method
                query_params = {
                    "TableName": self.table_name,
                    "KeyConditionExpression": (
                        "PK = :pk AND begins_with(SK, :ns_prefix)"
                    ),
                    "ExpressionAttributeValues": {
                        ":pk": {"S": pk},
                        ":ns_prefix": {"S": f"{checkpoint_ns}#"},
                    },
                    "ProjectionExpression": projection,
                    "ExpressionAttributeNames": expr_attr_names,
                    "ScanIndexForward": False,
                }

                # Get first result from generator
                for (
                    base_item,  # noqa: B007
                    payload_data,  # noqa: B007
                ) in self._find_checkpoints_with_valid_payload(query_params, limit=1):
                    break
                else:
                    return None

            # Extract metadata
            checkpoint_id = base_item["id"]["S"]
            checkpoint_ns = base_item["ns"]["S"]
            payload_type = base_item["type"]["S"]
            compression_str = base_item.get("compression", {}).get("S")
            compression_type = (
                CompressionType(compression_str) if compression_str else None
            )

            # Deserialize payload
            checkpoint_value = self.serializer.deserialize(
                payload_type, payload_data, compression_type
            )

            # Return complete checkpoint data
            return {
                "checkpoint": checkpoint_value["checkpoint"],
                "metadata": checkpoint_value["metadata"],
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
                "thread_id": thread_id,
                "parent_checkpoint_id": base_item.get("parent_checkpoint_id", {}).get(
                    "S"
                ),
            }

        except ClientError as err:
            logger.error(
                f"Error retrieving checkpoint: thread_id={thread_id}, "
                f"checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}, "
                f"error={err.response['Error']['Code']}"
            )
            raise err

    def put_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_checkpoint_id: str | None = None,
    ) -> dict:
        """Store a checkpoint with intelligent offloading.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint: Checkpoint data to store
            metadata: Checkpoint metadata
            parent_checkpoint_id: Optional parent checkpoint ID for lineage

        Returns:
            Dictionary with thread_id, checkpoint_ns, and checkpoint_id

        Raises:
            ClientError: If storage operation fails
        """
        checkpoint_id = checkpoint["id"]

        try:
            # Serialize checkpoint and metadata together
            checkpoint_value = {"checkpoint": checkpoint, "metadata": metadata}
            payload_type, payload_data, compression_type = self.serializer.serialize(
                checkpoint_value
            )

            # Generate keys for both storage backends
            chunk_key = _checkpoint_ref_pk(thread_id, checkpoint_ns, checkpoint_id)
            s3_key = _checkpoint_s3_key(thread_id, checkpoint_ns, checkpoint_id)

            # Build metadata item for base table
            pk = _checkpoint_pk(thread_id)
            sk = _checkpoint_sk(checkpoint_ns, checkpoint_id)

            ref_loc = self._determine_storage_location(payload_data)
            ref_key = s3_key if ref_loc == "S3" else chunk_key

            base_item: dict[str, Any] = {
                "PK": {"S": pk},
                "SK": {"S": sk},
                "id": {"S": checkpoint_id},
                "ns": {"S": checkpoint_ns},
                "type": {"S": payload_type},
                "ref_loc": {"S": ref_loc},
                "ref_key": {"S": ref_key},
            }

            if parent_checkpoint_id:
                base_item["parent_checkpoint_id"] = {"S": parent_checkpoint_id}

            if compression_type:
                base_item["compression"] = {"S": compression_type.value}

            if self.ttl_seconds:
                base_item["ttl"] = {"N": str(int(time.time() + self.ttl_seconds))}

            # Store metadata to base table
            self.dynamodb_client.put_item(TableName=self.table_name, Item=base_item)

            # Store payload based on configured store backends
            self.storage.store_data(
                chunk_key, s3_key, payload_data, allow_overwrite=True
            )

            logger.debug(
                f"Stored checkpoint: thread_id={thread_id}, "
                f"checkpoint_id={checkpoint_id}, checkpoint_ns={checkpoint_ns}, "
                f"ref_loc={ref_loc}, size={len(payload_data) / 1024:.1f}KB"
            )

            return {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }

        except ClientError as err:
            logger.error(
                f"Failed to save checkpoint: thread_id={thread_id}, "
                f"checkpoint_id={checkpoint_id}, error={err.response['Error']['Code']}"
            )
            raise err

    def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str | None = None,
        before_checkpoint_id: str | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[dict]:
        """List checkpoints from storage with optional filtering.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Optional namespace filter
            before_checkpoint_id: Optional filter for checkpoints before this ID
            limit: Optional maximum number of checkpoints to return

        Yields:
            Dictionary containing checkpoint data, metadata, and identifiers

        Raises:
            ClientError: If query or retrieval operations fail
        """
        try:
            pk = _checkpoint_pk(thread_id)

            # Build query parameters
            query_params = {
                "TableName": self.table_name,
                "ProjectionExpression": (
                    "id, ns, #t, ref_key, ref_loc, compression, parent_checkpoint_id"
                ),
                "ExpressionAttributeNames": {"#t": "type"},
                "ScanIndexForward": False,
            }

            # Key condition and attribute values based on checkpoint_ns presence
            if checkpoint_ns is not None:
                query_params["KeyConditionExpression"] = (
                    "PK = :pk AND begins_with(SK, :ns_prefix)"
                )
                query_params["ExpressionAttributeValues"] = {
                    ":pk": {"S": pk},
                    ":ns_prefix": {"S": f"{checkpoint_ns}#"},
                }
            else:
                query_params["KeyConditionExpression"] = "PK = :pk"
                query_params["ExpressionAttributeValues"] = {":pk": {"S": pk}}

            # Add filter expression for before_checkpoint_id
            if before_checkpoint_id:
                query_params["FilterExpression"] = "id < :before_checkpoint_id"
                if "ExpressionAttributeValues" not in query_params:
                    query_params["ExpressionAttributeValues"] = {}
                attr_values = cast(dict, query_params["ExpressionAttributeValues"])
                attr_values[":before_checkpoint_id"] = {"S": before_checkpoint_id}

            # Metadata filter function
            metadata_filter = None
            if filter:

                def metadata_filter(base_item: dict, payload_data: bytes) -> bool:
                    payload_type = base_item["type"]["S"]
                    compression_str = base_item.get("compression", {}).get("S")
                    compression_type = (
                        CompressionType(compression_str) if compression_str else None
                    )

                    checkpoint_value = self.serializer.deserialize(
                        payload_type, payload_data, compression_type
                    )
                    metadata = checkpoint_value["metadata"]
                    return all(metadata.get(k) == v for k, v in filter.items())

            # Yield deserialized results
            for base_item, payload_data in self._find_checkpoints_with_valid_payload(
                query_params, limit, metadata_filter
            ):
                payload_type = base_item["type"]["S"]
                compression_str = base_item.get("compression", {}).get("S")
                compression_type = (
                    CompressionType(compression_str) if compression_str else None
                )

                checkpoint_value = self.serializer.deserialize(
                    payload_type, payload_data, compression_type
                )

                yield {
                    "checkpoint": checkpoint_value["checkpoint"],
                    "metadata": checkpoint_value["metadata"],
                    "checkpoint_id": base_item["id"]["S"],
                    "checkpoint_ns": base_item["ns"]["S"],
                    "thread_id": thread_id,
                    "parent_checkpoint_id": base_item.get(
                        "parent_checkpoint_id", {}
                    ).get("S"),
                }

        except ClientError as err:
            logger.error(
                f"Error listing checkpoints: thread_id={thread_id}, "
                f"error={err.response['Error']['Code']}"
            )
            raise err

    # ========== Writes Operations ==========

    def get_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Get pending writes for a checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier

        Returns:
            List of (task_id, channel, value) tuples representing pending writes

        Raises:
            ClientError: If query or retrieval operations fail
        """
        pk = _writes_pk(thread_id, checkpoint_ns, checkpoint_id)
        writes = []

        try:
            # Only fetch needed attributes
            params = {
                "TableName": self.table_name,
                "KeyConditionExpression": "PK = :pk",
                "ExpressionAttributeValues": {":pk": {"S": pk}},
                "ProjectionExpression": (
                    "task_id, channel, ref_key, ref_loc, #t, compression"
                ),
                "ExpressionAttributeNames": {"#t": "type"},  # 'type' is reserved
            }

            # Paginated query
            while True:
                response = self.dynamodb_client.query(**params)  # type: ignore[arg-type]

                for item in response["Items"]:
                    task_id = item["task_id"]["S"]
                    channel = item["channel"]["S"]
                    ref_key = item["ref_key"]["S"]
                    ref_loc = item["ref_loc"]["S"]
                    payload_type = item["type"]["S"]
                    compression_str = item.get("compression", {}).get("S")
                    compression_type = (
                        CompressionType(compression_str) if compression_str else None
                    )

                    # Retrieve value payload
                    logger.debug(
                        f"🔍 RETRIEVING write payload: "
                        f"checkpoint_id={checkpoint_id}, task_id={task_id}, "
                        f"ref_loc={ref_loc}, ref_key={ref_key}"
                    )
                    payload_data = self.storage.retrieve_data(
                        ref_key, cast(StorageLocation, ref_loc)
                    )

                    # Ignore writes which doesn't have payload, consider this as
                    # orphan write
                    if payload_data is None:
                        logger.warning(
                            f"Value payload NOT FOUND for write: "
                            f"checkpoint_id={checkpoint_id}, task_id={task_id}, "
                            f"ref_loc={ref_loc}, ref_key={ref_key}"
                        )
                        continue

                    logger.debug(f"Retrieved {len(payload_data)} bytes for write")

                    # Deserialize value
                    value = self.serializer.deserialize(
                        payload_type, payload_data, compression_type
                    )

                    writes.append((task_id, channel, value))

                # Check for more pages
                if "LastEvaluatedKey" not in response:
                    break

                params["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            return writes

        except ClientError as err:
            logger.error(
                f"Error retrieving pending writes: checkpoint_id={checkpoint_id}, "
                f"error={err.response['Error']['Code']}"
            )
            raise err

    def put_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store writes with intelligent offloading and parallel execution.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            writes: Sequence of (channel, value) tuples to store
            task_id: Task identifier creating the writes
            task_path: Optional task path for nested task tracking

        Raises:
            ClientError: If storage operations fail
        """
        if not writes:
            return

        # Check if all writes are special (can be overwritten)
        all_special = all(w[0] in WRITES_IDX_MAP for w in writes)

        # Build all write items with payload storage
        for base_item, ref_item in self._build_write_items(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            writes,
            task_id,
            task_path,
            allow_overwrites=all_special,
        ):
            # Store metadata to base table
            self.put_single_write_item(
                checkpoint_id, base_item, allow_overwrites=all_special
            )

            # Store Chunks for the record
            self.storage.store_data(
                chunk_key=ref_item["chunk_key"],
                s3_key=ref_item["s3_key"],
                serialized_data=ref_item["value_data"],
                allow_overwrite=all_special,
            )

    def _build_write_items(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
        allow_overwrites: bool = False,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Build write items with intelligent offloading.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            writes: Sequence of (channel, value) tuples
            task_id: Task identifier
            task_path: Task path for nested tracking
            allow_overwrites: If True, always overwrite payloads in chunk/S3.
                If False, skip payload storage if it already exists.

        Returns:
            Iterator of (base_item, ref_item) tuples for sequential processing
        """
        pk = _writes_pk(thread_id, checkpoint_ns, checkpoint_id)

        for idx, (channel, value) in enumerate(writes):
            # Serialize value
            value_type, value_data, compression_type = self.serializer.serialize(value)

            # Resolve index (handle special channels)
            resolved_idx = str(int(WRITES_IDX_MAP.get(channel, idx)))

            # Generate keys for both storage backends
            chunk_key = _writes_ref_pk(
                thread_id, checkpoint_ns, checkpoint_id, task_id, resolved_idx
            )
            s3_key = _writes_s3_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, resolved_idx
            )

            # Store value payload using intelligent offloading
            # Pass allow_overwrites to ensure consistent behavior across
            # metadata and payload
            ref_loc = self._determine_storage_location(value_data)
            ref_key = s3_key if ref_loc == "S3" else chunk_key

            ref_item: dict[str, Any] = {
                "chunk_key": chunk_key,
                "s3_key": s3_key,
                "value_data": value_data,
            }

            logger.info(
                f"📝 PREPARED write payload: checkpoint_id={checkpoint_id}, "
                f"task_id={task_id}, idx={resolved_idx}, ref_loc={ref_loc}, "
                f"ref_key={ref_key}, size={len(value_data) / 1024:.1f}KB, "
                f"chunk_key={chunk_key}, allow_overwrites={allow_overwrites}"
            )

            # Build metadata item for base table
            sk = _writes_sk(task_id, int(resolved_idx))

            base_item: dict[str, Any] = {
                "PK": {"S": pk},
                "SK": {"S": sk},
                "task_id": {"S": task_id},
                "channel": {"S": channel},
                "idx": {"N": resolved_idx},
                "checkpoint_id": {"S": checkpoint_id},
                "type": {"S": value_type},
                "ref_loc": {"S": ref_loc},
                "ref_key": {"S": ref_key},
            }

            if task_path:
                base_item["task_path"] = {"S": task_path}

            if compression_type:
                base_item["compression"] = {"S": compression_type.value}

            if self.ttl_seconds:
                base_item["ttl"] = {"N": str(int(time.time() + self.ttl_seconds))}

            yield base_item, ref_item

    def put_single_write_item(
        self, checkpoint_id: str, item: dict[str, Any], allow_overwrites: bool = False
    ) -> None:
        """Write a single item to base table."""
        # Optimization: Build parameters once, similar to storage_strategy pattern
        put_params = {
            "TableName": self.table_name,
            "Item": item,
        }

        # Add conditional expression to prevent overwrites
        if not allow_overwrites:
            put_params["ConditionExpression"] = "attribute_not_exists(PK)"

        try:
            self.dynamodb_client.put_item(**put_params)  # type: ignore[arg-type]
        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            # ConditionalCheckFailedException means item already exists
            if error_code == "ConditionalCheckFailedException" and not allow_overwrites:
                # Silently ignore - item already exists
                return

            # Other errors should be raised
            logger.error(
                f"Error storing write: checkpoint_id={checkpoint_id}, "
                f"error={error_code}"
            )
            raise

    # ========== Deletion Operations ==========

    def get_thread_checkpoint_info(self, thread_id: str) -> list[tuple[str, str]]:
        """Get checkpoint info for a thread (for deletion).

        Args:
            thread_id: Thread identifier

        Returns:
            List of (checkpoint_ns, checkpoint_id) tuples

        Raises:
            ClientError: If query operation fails
        """
        try:
            pk = _checkpoint_pk(thread_id)
            items = []

            query_params: dict[str, Any] = {
                "TableName": self.table_name,
                "KeyConditionExpression": "PK = :pk",
                "ExpressionAttributeValues": {":pk": {"S": pk}},
                "ProjectionExpression": "SK",
            }

            # Paginated query
            while True:
                response = self.dynamodb_client.query(**query_params)
                items.extend(response.get("Items", []))

                if "LastEvaluatedKey" not in response:
                    break

                query_params["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            logger.info(f"Found {len(items)} checkpoints for thread_id={thread_id}")

            # Parse SK to extract namespace and checkpoint_id
            return [
                (parts[0], parts[1])
                for item in items
                for parts in [item["SK"]["S"].split("#", 1)]
            ]

        except ClientError as err:
            logger.error(
                f"Error getting checkpoint info: thread_id={thread_id}, "
                f"error={err.response['Error']['Code']}"
            )
            raise err

    def delete_thread_checkpoints(
        self, thread_id: str, checkpoint_info: list[tuple[str, str]]
    ) -> None:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            checkpoint_info: List of (checkpoint_ns, checkpoint_id) tuples

        Raises:
            ClientError: If deletion operations fail
        """
        if not checkpoint_info:
            return

        try:
            self._delete_checkpoint_items_parallel(checkpoint_info, thread_id)
        except ClientError as err:
            logger.error(
                f"Error deleting checkpoints: thread_id={thread_id}, "
                f"error={err.response['Error']['Code']}"
            )
            raise err

    def delete_thread_writes(
        self, thread_id: str, checkpoint_info: list[tuple[str, str]]
    ) -> None:
        """Delete writes for specific checkpoints.

        Args:
            thread_id: Thread identifier
            checkpoint_info: List of (checkpoint_ns, checkpoint_id) tuples

        Raises:
            ClientError: If deletion operations fail
        """
        if not checkpoint_info:
            return

        try:
            items_to_delete = []

            # Collect all write items for all checkpoints
            for checkpoint_ns, checkpoint_id in checkpoint_info:
                pk = _writes_pk(thread_id, checkpoint_ns, checkpoint_id)

                # Fetch needed attributes for deletion
                query_params: dict[str, Any] = {
                    "TableName": self.table_name,
                    "KeyConditionExpression": "PK = :pk",
                    "ExpressionAttributeValues": {":pk": {"S": pk}},
                    "ProjectionExpression": "PK, SK, ref_key, ref_loc",
                }

                # Paginated query
                while True:
                    response = self.dynamodb_client.query(**query_params)
                    items_to_delete.extend(response.get("Items", []))

                    if "LastEvaluatedKey" not in response:
                        break

                    query_params["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            self._delete_write_items_batch(items_to_delete)

        except ClientError as err:
            logger.error(
                f"Error deleting writes: error={err.response['Error']['Code']}"
            )
            raise

    def _delete_checkpoint_items_parallel(
        self, items: list[tuple[str, str]], thread_id: str
    ) -> None:
        """Delete checkpoint items using batch operations.

        Args:
            items: List of (checkpoint_ns, checkpoint_id) tuples
            thread_id: Thread identifier

        Raises:
            ClientError: If deletion operations fail
        """
        if not items:
            return

        # Step 1: Batch get checkpoint metadata to find payload locations
        keys_to_get = [
            {
                "PK": {"S": _checkpoint_pk(thread_id)},
                "SK": {"S": _checkpoint_sk(checkpoint_ns, checkpoint_id)},
            }
            for checkpoint_ns, checkpoint_id in items
        ]

        payload_items: list[tuple[str, StorageLocation]] = []
        metadata_keys: list[dict[str, Any]] = []

        # batch_get_item limit is 100 items per request
        batch_size = 100

        for i in range(0, len(keys_to_get), batch_size):
            batch_keys = keys_to_get[i : i + batch_size]

            # Build request with proper types
            keys_and_attrs: KeysAndAttributesTypeDef = {
                "Keys": batch_keys,
                "ProjectionExpression": "PK, SK, ref_key, ref_loc",
            }
            request_items: dict[str, KeysAndAttributesTypeDef] = {
                self.table_name: keys_and_attrs
            }

            try:
                # Queue-based processing: continue until no UnprocessedKeys
                while request_items:
                    response = self.dynamodb_client.batch_get_item(
                        RequestItems=request_items
                    )

                    # Process returned items
                    for item in response.get("Responses", {}).get(self.table_name, []):
                        ref_key = item.get("ref_key", {}).get("S")
                        ref_loc = item.get("ref_loc", {}).get("S")

                        # Collect payload for batch deletion
                        if ref_key and ref_loc:
                            payload_items.append(
                                (ref_key, cast(StorageLocation, ref_loc))
                            )

                        # Collect metadata key for batch deletion
                        metadata_keys.append({"PK": item["PK"], "SK": item["SK"]})

                    # Check for unprocessed keys and continue loop
                    unprocessed = response.get("UnprocessedKeys", {})
                    request_items = (
                        {
                            k: cast("KeysAndAttributesTypeDef", v)
                            for k, v in unprocessed.items()
                        }
                        if unprocessed
                        else {}
                    )
                    if request_items:
                        table_request = request_items.get(self.table_name)
                        num_unprocessed = (
                            len(table_request.get("Keys", [])) if table_request else 0
                        )
                        logger.debug(f"Retrying {num_unprocessed} unprocessed keys")

            except ClientError as err:
                logger.error(
                    f"Failed to batch get checkpoint metadata: "
                    f"error={err.response['Error']['Code']}"
                )
                raise

        # Step 2: Batch delete payloads from storage backend (DynamoDB/S3)
        if payload_items:
            result = self.storage.batch_delete_data(payload_items)
            if result["failed"]:
                logger.warning(
                    f"Failed to delete {len(result['failed'])} checkpoint payloads"
                )

        # Step 3: Batch delete metadata from base table
        if metadata_keys:
            self._batch_delete_metadata(metadata_keys)

    def _delete_write_items_batch(self, items: list[dict]) -> None:
        """Delete write items using batch operations.

        Collects all write payloads, then uses batch delete for payloads
        and metadata separately.

        Args:
            items: List of write metadata items

        Raises:
            ClientError: If deletion operations fail
        """
        if not items:
            return

        # Step 1: Collect payload locations and metadata keys
        payload_items: list[tuple[str, StorageLocation]] = []
        metadata_keys: list[dict[str, Any]] = []

        for item in items:
            ref_key = item.get("ref_key", {}).get("S")
            ref_loc = item.get("ref_loc", {}).get("S")

            # Collect payload for batch deletion
            if ref_key and ref_loc:
                payload_items.append((ref_key, cast(StorageLocation, ref_loc)))

            # Collect metadata key for batch deletion
            metadata_keys.append({"PK": item["PK"], "SK": item["SK"]})

        # Step 2: Batch delete payloads from storage backend (DynamoDB/S3)
        if payload_items:
            result = self.storage.batch_delete_data(payload_items)
            if result["failed"]:
                logger.warning(
                    f"Failed to delete {len(result['failed'])} write payloads"
                )

        # Step 3: Batch delete metadata from base table
        if metadata_keys:
            self._batch_delete_metadata(metadata_keys)

    def _batch_delete_metadata(self, keys: list[dict[str, Any]]) -> None:
        """Batch delete metadata items from base table.

        Uses batch_write_item API with automatic batching (max 25 items per request).

        Args:
            keys: List of DynamoDB key dictionaries (PK, SK)

        Raises:
            ClientError: If deletion operations fail
        """
        if not keys:
            return

        # DynamoDB batch_write_item limit is 25 items per request
        batch_size = 25

        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]

            # Build delete requests with proper types
            delete_requests: list[WriteRequestTypeDef] = [
                {"DeleteRequest": {"Key": key}} for key in batch
            ]

            request_items: dict[str, list[WriteRequestTypeDef]] = {
                self.table_name: delete_requests
            }

            try:
                # Queue-based processing: continue until no UnprocessedItems
                while request_items:
                    response = self.dynamodb_client.batch_write_item(
                        RequestItems=request_items
                    )

                    # Check for unprocessed items and continue loop
                    unprocessed = response.get("UnprocessedItems", {})
                    request_items = (
                        {
                            k: cast("list[WriteRequestTypeDef]", list(v))
                            for k, v in unprocessed.items()
                        }
                        if unprocessed
                        else {}
                    )
                    if request_items:
                        # Backoff before retry
                        time.sleep(0.1)
                        num_unprocessed = len(request_items.get(self.table_name, []))
                        logger.debug(
                            f"Retrying {num_unprocessed} unprocessed metadata items"
                        )
                    else:
                        logger.debug(
                            f"Successfully deleted {len(batch)} metadata items"
                        )

            except ClientError as err:
                logger.error(
                    f"Batch delete metadata failed: "
                    f"error={err.response['Error']['Code']}, "
                    f"batch_size={len(batch)}"
                )
                raise
