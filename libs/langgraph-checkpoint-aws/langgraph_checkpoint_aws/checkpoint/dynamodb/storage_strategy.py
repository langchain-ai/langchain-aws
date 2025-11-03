"""Storage strategy for handling data persistence with S3 offloading."""

import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_dynamodb.type_defs import (
        WriteRequestTypeDef,
    )
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import DeleteTypeDef, ObjectIdentifierTypeDef

logger = logging.getLogger(__name__)

# Storage thresholds for Offloading to S3
S3_OFFLOAD_THRESHOLD = 350 * 1024  # 350 KB

# Storage location types
StorageLocation = Literal["DYNAMODB", "S3"]


class StorageStrategy:
    """Determines and executes storage strategy (DynamoDB vs S3) based on data size.
    The 350KB threshold provides a safety margin below DynamoDB's 400KB item limit.
    """

    def __init__(
        self,
        dynamodb_client: "DynamoDBClient",
        table_name: str,
        s3_client: "S3Client | None" = None,
        s3_bucket: str | None = None,
        ttl_seconds: int | None = None,
    ):
        """Initialize storage strategy.

        Args:
            dynamodb_client: DynamoDB client for chunk table operations
            table_name: Name of the DynamoDB for payload storage
            s3_client: Optional S3 client for large data offloading
            s3_bucket: Optional S3 bucket name for offloading
            ttl_seconds: Optional TTL in seconds for automatic cleanup
        """
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.ttl_seconds = ttl_seconds
        self.s3_enabled = s3_client is not None and s3_bucket is not None

        # Configure S3 lifecycle policy if TTL is set
        if self.s3_enabled and self.ttl_seconds:
            self._ensure_s3_lifecycle_policy()

    def should_offload_to_s3(self, serialized_data: bytes) -> bool:
        """Check if data should be offloaded to S3 based on size and configuration.

        Args:
            serialized_data: Serialized data bytes

        Returns:
            True if data exceeds threshold and S3 is configured, False otherwise
        """
        if not self.s3_enabled:
            return False

        data_size = len(serialized_data)

        if data_size > S3_OFFLOAD_THRESHOLD:
            logger.debug(
                f"Data size {data_size / 1024:.1f}KB exceeds threshold "
                f"{S3_OFFLOAD_THRESHOLD / 1024:.0f}KB - will offload to S3"
            )
            return True

        logger.debug(
            f"Data size {data_size / 1024:.1f}KB below threshold "
            f"{S3_OFFLOAD_THRESHOLD / 1024:.0f}KB - will store in DynamoDB"
        )
        return False

    def store_data(
        self,
        chunk_key: str,
        s3_key: str,
        serialized_data: bytes,
        allow_overwrite: bool = False,
    ) -> StorageLocation:
        """Store data using appropriate strategy based on size.

        Args:
            chunk_key: Primary key for DynamoDB chunk table
            s3_key: S3 key to use if offloading to S3
            serialized_data: Serialized data bytes to store
            allow_overwrite: If True, always store data (overwrite if exists).
                If False, only store if data doesn't already exist (default: False).

        Returns:
            StorageLocation indicating where data was stored ("DYNAMODB" or "S3")

        Raises:
            ClientError: If storage operation fails
        """
        if self.should_offload_to_s3(serialized_data):
            self._store_to_s3(s3_key, serialized_data, allow_overwrite)
            logger.debug(f"Stored {len(serialized_data) / 1024:.1f}KB to S3: {s3_key}")
            return "S3"

        self._store_to_dynamodb(chunk_key, serialized_data, allow_overwrite)
        logger.debug(
            f"Stored {len(serialized_data) / 1024:.1f}KB to DynamoDB "
            f"chunk table: {chunk_key}"
        )
        return "DYNAMODB"

    def retrieve_data(
        self, ref_key: str, ref_location: StorageLocation
    ) -> bytes | None:
        """Retrieve data from the appropriate storage backend.

        Args:
            ref_key: Reference key (chunk table PK or S3 key)
            ref_location: Storage location ("DYNAMODB" or "S3")

        Returns:
            Retrieved data bytes, or None if not found

        Raises:
            ClientError: If retrieval operation fails (except for not found)
        """
        try:
            if ref_location == "DYNAMODB":
                return self._retrieve_from_dynamodb(ref_key)

            if ref_location == "S3":
                return self._retrieve_from_s3(ref_key)

            logger.error(f"Invalid storage location: {ref_location}")
            return None

        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code in ("NoSuchKey", "ResourceNotFoundException"):
                logger.debug(f"Data not found: location={ref_location}, key={ref_key}")
                return None

            logger.error(
                f"Failed to retrieve data: location={ref_location}, "
                f"key={ref_key}, error={error_code}, "
                f"message={err.response['Error'].get('Message', 'No message')}"
            )
            raise

    def batch_delete_data(
        self, items: list[tuple[str, StorageLocation]]
    ) -> dict[str, list[str]]:
        """Delete multiple data items using batch operations.

        Groups items by storage location and uses batch APIs:
        - DynamoDB: batch_write_item (max 25 items per batch)
        - S3: delete_objects (max 1000 objects per batch)

        Args:
            items: List of (ref_key, ref_location) tuples to delete

        Returns:
            Dictionary with 'failed' key containing list of failed ref_keys
        """
        if not items:
            return {"failed": []}

        # Group items by storage location
        dynamodb_keys: list[str] = []
        s3_keys: list[str] = []

        for ref_key, ref_location in items:
            if ref_location == "DYNAMODB":
                dynamodb_keys.append(ref_key)
            elif ref_location == "S3":
                s3_keys.append(ref_key)
            else:
                logger.warning(f"Invalid storage location: {ref_location}")

        failed_keys: list[str] = []

        # Batch delete from DynamoDB
        if dynamodb_keys:
            failed_keys.extend(self._batch_delete_from_dynamodb(dynamodb_keys))

        # Batch delete from S3
        if s3_keys:
            failed_keys.extend(self._batch_delete_from_s3(s3_keys))

        if failed_keys:
            logger.warning(f"Failed to delete {len(failed_keys)} items: {failed_keys}")

        return {"failed": failed_keys}

    # ========== DynamoDB Operations ==========

    def _store_to_dynamodb(
        self, chunk_key: str, data: bytes, allow_overwrite: bool = False
    ) -> None:
        """Store data to DynamoDB chunk table.

        Args:
            chunk_key: Primary key for the chunk table
            data: Serialized data bytes
            allow_overwrite: If True, always store (overwrite if exists).
                If False, only store if item doesn't exist (default: False).

        Raises:
            ClientError: If DynamoDB operation fails
        """
        # Build item with required attributes
        item = {
            "PK": {"S": chunk_key},
            "SK": {"S": "CHUNK"},
            "payload": {"B": data},
        }

        # Add TTL if configured
        if self.ttl_seconds:
            item["ttl"] = {"N": str(int(time.time() + self.ttl_seconds))}

        # Build put_item parameters
        put_params: dict[str, Any] = {
            "TableName": self.table_name,
            "Item": item,
        }

        # Add conditional expression to prevent overwrites
        if not allow_overwrite:
            put_params["ConditionExpression"] = "attribute_not_exists(PK)"

        logger.debug(
            f"Storing to chunk table: table={self.table_name}, "
            f"PK={chunk_key}, size={len(data)}B, allow_overwrite={allow_overwrite}"
        )

        try:
            self.dynamodb_client.put_item(**put_params)
            logger.debug(f"Successfully stored to chunk table: PK={chunk_key}")
        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code == "ConditionalCheckFailedException" and not allow_overwrite:
                logger.debug(f"Chunk already exists, skipping: PK={chunk_key}")
                return

            # Other errors should be raised
            logger.error(
                f"Failed to store to chunk table: PK={chunk_key}, error={error_code}"
            )
            raise

    def _retrieve_from_dynamodb(self, chunk_key: str) -> bytes | None:
        """Retrieve data from DynamoDB chunk table.

        Args:
            chunk_key: Primary key for the chunk table

        Returns:
            Retrieved data bytes, or None if not found

        Raises:
            ClientError: If DynamoDB operation fails
        """
        logger.debug(
            f"Retrieving from chunk table: table={self.table_name}, PK={chunk_key}"
        )

        response = self.dynamodb_client.get_item(
            TableName=self.table_name,
            Key={"PK": {"S": chunk_key}, "SK": {"S": "CHUNK"}},
            ProjectionExpression="payload",
        )

        if "Item" not in response:
            logger.debug(f"Item not found in chunk table: PK={chunk_key}")
            return None

        payload = response["Item"]["payload"]["B"]
        logger.debug(
            f"Retrieved from chunk table: PK={chunk_key}, size={len(payload)}B"
        )
        return payload

    def _batch_delete_from_dynamodb(self, chunk_keys: list[str]) -> list[str]:
        """Batch delete data from DynamoDB chunk table.

        Uses batch_write_item API with automatic batching (max 25 items per request).

        Args:
            chunk_keys: List of primary keys for the chunk table

        Returns:
            List of chunk_keys that failed to delete after retries
        """
        if not chunk_keys:
            return []

        failed_keys: list[str] = []

        # DynamoDB batch_write_item limit is 25 items per request
        batch_size = 25

        for i in range(0, len(chunk_keys), batch_size):
            batch = chunk_keys[i : i + batch_size]

            # Build delete requests with proper types
            delete_requests: list[WriteRequestTypeDef] = [
                {
                    "DeleteRequest": {
                        "Key": {
                            "PK": {"S": key},
                            "SK": {"S": "CHUNK"},
                        }
                    }
                }
                for key in batch
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
                        num_unprocessed = len(request_items.get(self.table_name, []))
                        logger.debug(f"Retrying {num_unprocessed} unprocessed items")
                    else:
                        logger.debug(
                            f"Successfully deleted {len(batch)} items from DynamoDB"
                        )

            except ClientError as err:
                error_code = err.response["Error"]["Code"]
                logger.error(
                    f"Batch delete failed: error={error_code}, batch_size={len(batch)}"
                )
                # Add all items in this batch to failed list
                failed_keys.extend(batch)

        return failed_keys

    # ========== S3 Operations ==========

    # Needs PutLifeCyclePolicy
    def _ensure_s3_lifecycle_policy(self) -> None:
        """Configure S3 lifecycle policy for TTL-based expiration if not exists.

        Creates a tag-filtered lifecycle rule that expires objects based on their
        ttl-days tag. This allows different objects to have different TTLs.
        Rule ID includes TTL days for uniqueness when sharing buckets.
        """
        if not self.s3_enabled or not self.ttl_seconds or self.s3_client is None:
            return

        try:
            # Calculate expiration days (always round up, minimum 1 day)
            # S3 Lifecycle expiration: convert seconds to days (ceil, min 1 day)
            expiration_days = max(1, (self.ttl_seconds + 86399) // 86400)
            rule_id = f"ttl-expiration-{expiration_days}d"

            # Get existing rules
            assert self.s3_bucket is not None  # Already checked by s3_enabled
            try:
                response = self.s3_client.get_bucket_lifecycle_configuration(
                    Bucket=self.s3_bucket
                )
                existing_rules = response.get("Rules", [])

                # Skip if rule exists
                if any(rule.get("ID") == rule_id for rule in existing_rules):
                    logger.debug(
                        f"S3 lifecycle rule '{rule_id}' exists in {self.s3_bucket}"
                    )
                    return

            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
                    raise
                existing_rules = []

            # Add new rule with tag filter for this specific TTL
            existing_rules.append(
                {
                    "ID": rule_id,
                    "Status": "Enabled",
                    "Filter": {
                        "Tag": {"Key": "ttl-days", "Value": str(expiration_days)}
                    },
                    "Expiration": {"Days": expiration_days},
                }
            )

            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.s3_bucket, LifecycleConfiguration={"Rules": existing_rules}
            )

            logger.info(
                f"Added S3 lifecycle rule '{rule_id}' to {self.s3_bucket}: "
                f"expire after {expiration_days} days"
            )

        except ClientError as e:
            logger.warning(
                f"Failed to configure S3 lifecycle: {e.response['Error']['Code']}"
            )

    def _store_to_s3(
        self, s3_key: str, data: bytes, allow_overwrite: bool = False
    ) -> None:
        """Store data to S3 with TTL metadata.

        Args:
            s3_key: S3 object key
            data: Serialized data bytes
            allow_overwrite: If True, always store (overwrite if exists).
                If False, only store if object doesn't exist (default: False).

        Raises:
            ClientError: If S3 operation fails
        """
        if not self.s3_enabled:
            raise ValueError("S3 is not configured but offloading was attempted")

        # Build put parameters
        put_params: dict[str, Any] = {
            "Bucket": self.s3_bucket,
            "Key": s3_key,
            "Body": data,
            "Metadata": {},
        }

        # Add conditional write to prevent overwrites
        if not allow_overwrite:
            put_params["IfNoneMatch"] = "*"

        # Add TTL tag if configured - used by lifecycle policy for expiration
        if self.ttl_seconds:
            # Calculate expiration days (always round up, minimum 1 day)
            # S3 Lifecycle expiration: convert seconds to days (ceil, min 1 day)
            expiration_days = max(1, (self.ttl_seconds + 86399) // 86400)
            put_params["Tagging"] = f"ttl-days={expiration_days}"

        try:
            if self.s3_client is not None:
                self.s3_client.put_object(**put_params)
            logger.debug(
                f"S3 PUT: {s3_key} ({len(data) / 1024:.1f}KB)"
                + (f" with TTL {self.ttl_seconds}s" if self.ttl_seconds else "")
            )
        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code == "PreconditionFailed" and not allow_overwrite:
                logger.debug(f"S3 object already exists, skipping: {s3_key}")
                return

            logger.error(
                f"Failed to store to S3: key={s3_key}, "
                f"error={error_code}, "
                f"message={err.response['Error'].get('Message', 'No message')}"
            )
            raise

    def _retrieve_from_s3(self, s3_key: str) -> bytes | None:
        """Retrieve data from S3.

        Args:
            s3_key: S3 object key

        Returns:
            Retrieved data bytes

        Raises:
            ClientError: If S3 operation fails
        """
        if not self.s3_enabled or self.s3_client is None:
            raise ValueError("S3 is not configured but retrieval was attempted")

        assert self.s3_bucket is not None  # Already checked by s3_enabled
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        data = response["Body"].read()

        logger.debug(f"Retrieved {len(data) / 1024:.1f}KB from S3: {s3_key}")
        return data

    def _batch_delete_from_s3(self, s3_keys: list[str]) -> list[str]:
        """Batch delete objects from S3.

        Uses delete_objects API with automatic batching (max 1000 objects per request).

        Args:
            s3_keys: List of S3 object keys to delete

        Returns:
            List of s3_keys that failed to delete
        """
        if not s3_keys:
            return []

        if not self.s3_enabled or self.s3_client is None:
            logger.warning(
                f"S3 not enabled, cannot delete {len(s3_keys)} objects: "
                f"{s3_keys[:5]}..."
            )
            return s3_keys

        failed_keys: list[str] = []
        # S3 delete_objects limit is 1000 objects per request
        batch_size = 1000

        for i in range(0, len(s3_keys), batch_size):
            batch = s3_keys[i : i + batch_size]

            # Build delete request with proper types
            objects_to_delete: list[ObjectIdentifierTypeDef] = [
                {"Key": key} for key in batch
            ]
            delete_request: DeleteTypeDef = {"Objects": objects_to_delete}

            assert self.s3_bucket is not None  # Already checked by s3_enabled
            try:
                response = self.s3_client.delete_objects(
                    Bucket=self.s3_bucket,
                    Delete=delete_request,
                )

                # Check for errors in response
                errors = response.get("Errors", [])
                if errors:
                    for error in errors:
                        failed_key = error.get("Key")
                        error_code = error.get("Code")
                        error_message = error.get("Message", "No message")

                        logger.error(
                            f"Failed to delete S3 object: key={failed_key}, "
                            f"error={error_code}, message={error_message}"
                        )

                        if failed_key:
                            failed_keys.append(failed_key)

                # Log successful deletions
                deleted = response.get("Deleted", [])
                if deleted:
                    logger.debug(f"Successfully deleted {len(deleted)} objects from S3")

            except ClientError as err:
                error_code = err.response["Error"]["Code"]
                logger.error(
                    f"S3 batch delete failed: error={error_code}, "
                    f"batch_size={len(batch)}"
                )

                # Add all items in this batch to failed list
                failed_keys.extend(batch)

        return failed_keys
