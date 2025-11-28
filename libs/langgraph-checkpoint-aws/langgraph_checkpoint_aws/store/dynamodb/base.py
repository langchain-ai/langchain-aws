"""DynamoDB store implementation for LangGraph.

This module provides a DynamoDB-backed store implementation that extends
the BaseStore class from LangGraph. It offers persistent storage with
hierarchical namespaces and key-value operations without vector search
capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, cast

import boto3
import orjson
from botocore.exceptions import ClientError
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

from .exceptions import DynamoDBConnectionError, TableCreationError, ValidationError

logger = logging.getLogger(__name__)


class DynamoDBStore(BaseStore):
    """DynamoDB-backed store implementation for LangGraph.

    This store provides persistent key-value storage using AWS DynamoDB.
    It supports hierarchical namespaces, TTL (time-to-live) for automatic
    item expiration, and basic filtering capabilities.

    The store uses a single DynamoDB table with the following schema:
    - PK (Partition Key): Namespace (joined with ':')
    - SK (Sort Key): Item key
    - value: The stored dictionary
    - created_at: ISO format timestamp
    - updated_at: ISO format timestamp
    - expires_at: Unix timestamp for TTL (optional)

    Examples:
        Basic usage:
        ```python
        from langgraph_checkpoint_aws import DynamoDBStore

        store = DynamoDBStore(table_name="my-store-table")
        store.setup()  # Create table if it doesn't exist

        # Store and retrieve data
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        print(item.value)  # {"theme": "dark"}
        ```

        Using context manager:
        ```python
        from langgraph_checkpoint_aws import DynamoDBStore

        with DynamoDBStore.from_conn_string("my-store-table") as store:
            store.setup()
            store.put(("docs",), "doc1", {"text": "Hello"})
            items = store.search(("docs",))
        ```

        With TTL configuration:
        ```python
        store = DynamoDBStore(
            table_name="my-store-table",
            ttl={
                "default_ttl": 60,  # 60 minutes default TTL
                "refresh_on_read": True,  # Refresh TTL on reads
            }
        )
        store.setup()

        # Item will expire after 60 minutes
        store.put(("temp",), "data", {"value": 123})
        ```

    Note:
        Make sure to call `setup()` before first use to create the necessary
        DynamoDB table if it doesn't exist.

    Warning:
        DynamoDB charges are based on read/write capacity and storage.
        Consider using on-demand billing for unpredictable workloads or
        provisioned capacity for consistent traffic patterns.
    """

    MIGRATIONS: list[str] = []
    supports_ttl = True

    def __init__(
        self,
        table_name: str,
        *,
        region_name: str | None = None,
        boto3_session: boto3.Session | None = None,
        ttl: TTLConfig | None = None,
        max_read_capacity_units: int | None = None,
        max_write_capacity_units: int | None = None,
    ) -> None:
        """Initialize DynamoDB store.

        Args:
            table_name: Name of the DynamoDB table to use.
            region_name: AWS region name. If not provided, uses default from AWS config.
            boto3_session: Optional boto3 session to use. If not provided, creates a new one.
            ttl: Optional TTL configuration for automatic item expiration.
            max_read_capacity_units: Maximum read capacity units for on-demand mode.
                Only used when creating a new table. Default is 10.
            max_write_capacity_units: Maximum write capacity units for on-demand mode.
                Only used when creating a new table. Default is 10.
        """
        super().__init__()
        self.table_name = table_name
        self.ttl_config = ttl
        self.max_read_capacity_units = max_read_capacity_units or 10
        self.max_write_capacity_units = max_write_capacity_units or 10

        # Initialize boto3 session and resources
        if boto3_session:
            self.session = boto3_session
        else:
            self.session = boto3.Session(region_name=region_name)

        try:
            self.dynamodb = self.session.resource("dynamodb")
            self.table = self.dynamodb.Table(table_name)
        except Exception as e:
            raise DynamoDBConnectionError(
                f"Failed to initialize DynamoDB connection: {e}"
            ) from e

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        table_name: str,
        *,
        region_name: str | None = None,
        ttl: TTLConfig | None = None,
        max_read_capacity_units: int | None = None,
        max_write_capacity_units: int | None = None,
    ) -> Iterator[DynamoDBStore]:
        """Create a DynamoDB store instance using a context manager.

        Args:
            table_name: Name of the DynamoDB table to use.
            region_name: AWS region name. If not provided, uses default from AWS config.
            ttl: Optional TTL configuration for automatic item expiration.
            max_read_capacity_units: Maximum read capacity units for on-demand mode.
            max_write_capacity_units: Maximum write capacity units for on-demand mode.

        Yields:
            DynamoDBStore: A DynamoDB store instance.

        Example:
            ```python
            with DynamoDBStore.from_conn_string("my-table") as store:
                store.setup()
                store.put(("docs",), "doc1", {"text": "Hello"})
            ```
        """
        store = cls(
            table_name=table_name,
            region_name=region_name,
            ttl=ttl,
            max_read_capacity_units=max_read_capacity_units,
            max_write_capacity_units=max_write_capacity_units,
        )
        try:
            yield store
        finally:
            # No cleanup needed for DynamoDB client
            pass

    def setup(self) -> None:
        """Set up the DynamoDB table.

        This method creates the DynamoDB table if it doesn't already exist.
        It configures the table with:
        - On-demand billing mode
        - Primary key: PK (partition key) and SK (sort key)
        - TTL enabled on expires_at attribute (if TTL config provided)

        This should be called before first use of the store.

        Raises:
            TableCreationError: If table creation fails.
        """
        try:
            # Try to load the table to check if it exists
            self.table.load()
            logger.info(f"DynamoDB table '{self.table_name}' already exists.")

            # Enable TTL if configured and not already enabled
            if self.ttl_config:
                self._enable_ttl()

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Table doesn't exist, create it
                logger.info(f"Creating DynamoDB table '{self.table_name}'...")
                self._create_table()
            else:
                raise TableCreationError(
                    f"Failed to check/create table '{self.table_name}': {e}"
                ) from e

    def _create_table(self) -> None:
        """Create the DynamoDB table with appropriate configuration."""
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},  # Partition key
                    {"AttributeName": "SK", "KeyType": "RANGE"},  # Sort key
                ],
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
                OnDemandThroughput={
                    "MaxReadRequestUnits": self.max_read_capacity_units,
                    "MaxWriteRequestUnits": self.max_write_capacity_units,
                },
            )
            # Wait for table to be created
            table.wait_until_exists()
            self.table = table
            logger.info(f"DynamoDB table '{self.table_name}' created successfully.")

            # Enable TTL if configured
            if self.ttl_config:
                self._enable_ttl()

        except Exception as e:
            raise TableCreationError(
                f"Failed to create table '{self.table_name}': {e}"
            ) from e

    def _enable_ttl(self) -> None:
        """Enable TTL on the DynamoDB table."""
        try:
            client = self.session.client("dynamodb")
            client.update_time_to_live(
                TableName=self.table_name,
                TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
            )
            logger.info(f"TTL enabled on table '{self.table_name}'.")
        except ClientError as e:
            # TTL might already be enabled or enabling, log but don't fail
            logger.warning(f"Could not enable TTL on table '{self.table_name}': {e}")

    def _construct_composite_key(
        self, namespace: tuple[str, ...], key: str
    ) -> tuple[str, str]:
        """Construct DynamoDB composite key from namespace and key.

        Args:
            namespace: Hierarchical namespace tuple.
            key: Item key.

        Returns:
            Tuple of (PK, SK) for DynamoDB.
        """
        namespace_str = ":".join(namespace)
        return (namespace_str, key)

    def _deconstruct_namespace(self, namespace: str) -> tuple[str, ...]:
        """Deconstruct namespace string back to tuple.

        Args:
            namespace: Namespace string (e.g., "users:123").

        Returns:
            Namespace tuple (e.g., ("users", "123")).
        """
        if not namespace:
            return ()
        if ":" in namespace:
            return tuple(namespace.split(":"))
        return (namespace,)

    def _map_to_item(self, result_dict: dict[str, Any], item_type: type = Item) -> Item:
        """Map DynamoDB item to store Item.

        Args:
            result_dict: DynamoDB item dictionary.
            item_type: Type of item to create (Item or SearchItem).

        Returns:
            Item or SearchItem instance.
        """
        namespace = self._deconstruct_namespace(result_dict["PK"])
        key = result_dict["SK"]
        value = result_dict["value"]

        # Parse timestamps
        created_at = datetime.fromisoformat(result_dict["created_at"])
        updated_at = datetime.fromisoformat(result_dict["updated_at"])

        return item_type(
            value=value,
            key=key,
            namespace=namespace,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _calculate_expiry(self, ttl_minutes: float | None) -> int | None:
        """Calculate Unix timestamp for TTL expiry.

        Args:
            ttl_minutes: TTL in minutes.

        Returns:
            Unix timestamp for expiry, or None if no TTL.
        """
        if ttl_minutes is None:
            return None
        # DynamoDB TTL expects Unix timestamp in seconds
        expiry_seconds = int(datetime.now(timezone.utc).timestamp() + (ttl_minutes * 60))
        return expiry_seconds

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations in a batch.

        Args:
            ops: Iterable of operations (GetOp, PutOp, SearchOp, ListNamespacesOp).

        Returns:
            List of results corresponding to each operation.
        """
        results: list[Result] = []

        for op in ops:
            if isinstance(op, GetOp):
                result = self._batch_get_op(op)
            elif isinstance(op, PutOp):
                result = self._batch_put_op(op)
            elif isinstance(op, SearchOp):
                result = self._batch_search_op(op)
            elif isinstance(op, ListNamespacesOp):
                result = self._batch_list_namespaces_op(op)
            else:
                raise NotImplementedError(f"Operation type {type(op)} not supported")
            results.append(result)

        return results

    def _batch_get_op(self, op: GetOp) -> Item | None:
        """Execute a GetOp operation.

        Args:
            op: GetOp operation.

        Returns:
            Item if found, None otherwise.
        """
        composite_key = self._construct_composite_key(op.namespace, op.key)
        try:
            response = self.table.get_item(Key={"PK": composite_key[0], "SK": composite_key[1]})
            item = response.get("Item")
            if item:
                # Refresh TTL if configured
                if op.refresh_ttl and self.ttl_config:
                    self._refresh_ttl(composite_key[0], composite_key[1])
                return self._map_to_item(item)
            return None
        except Exception as e:
            logger.error(f"Error getting item {op.namespace}/{op.key}: {e}")
            return None

    def _batch_put_op(self, op: PutOp) -> None:
        """Execute a PutOp operation.

        Args:
            op: PutOp operation.
        """
        if op.value is None:
            # Delete operation
            self._delete_item(op.namespace, op.key)
        else:
            # Put operation
            self._put_item(op.namespace, op.key, op.value, op.ttl)
        return None

    def _batch_search_op(self, op: SearchOp) -> list[SearchItem]:
        """Execute a SearchOp operation.

        Args:
            op: SearchOp operation.

        Returns:
            List of SearchItem instances.
        """
        namespace_str = ":".join(op.namespace_prefix)

        try:
            # Query items with the namespace prefix
            response = self.table.query(
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": namespace_str},
                Limit=op.limit,
            )

            items = response.get("Items", [])

            # Apply filter if provided
            if op.filter:
                items = self._apply_filter(items, op.filter)

            # Apply offset
            if op.offset > 0:
                items = items[op.offset :]

            # Convert to SearchItem instances
            results = [self._map_to_item(item, SearchItem) for item in items]

            # Refresh TTL if configured
            if op.refresh_ttl and self.ttl_config:
                for item in items:
                    self._refresh_ttl(item["PK"], item["SK"])

            return results

        except Exception as e:
            logger.error(f"Error searching namespace {op.namespace_prefix}: {e}")
            return []

    def _batch_list_namespaces_op(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Execute a ListNamespacesOp operation.

        Args:
            op: ListNamespacesOp operation.

        Returns:
            List of namespace tuples.
        """
        try:
            # Scan the table to get all unique namespaces
            response = self.table.scan(
                ProjectionExpression="PK",
            )

            namespaces_set = set()
            for item in response.get("Items", []):
                namespace = self._deconstruct_namespace(item["PK"])
                namespaces_set.add(namespace)

            # Handle pagination if more items exist
            while "LastEvaluatedKey" in response:
                response = self.table.scan(
                    ProjectionExpression="PK",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for item in response.get("Items", []):
                    namespace = self._deconstruct_namespace(item["PK"])
                    namespaces_set.add(namespace)

            # Filter namespaces based on match conditions
            namespaces = list(namespaces_set)
            filtered = self._filter_namespaces(namespaces, op)

            # Apply limit and offset
            start = op.offset
            end = start + op.limit
            return filtered[start:end]

        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return []

    def _filter_namespaces(
        self, namespaces: list[tuple[str, ...]], op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        """Filter namespaces based on operation criteria.

        Args:
            namespaces: List of namespace tuples.
            op: ListNamespacesOp with filter criteria.

        Returns:
            Filtered list of namespaces.
        """
        filtered = namespaces

        # Apply match conditions (prefix/suffix)
        for condition in op.match_conditions:
            if condition.match_type == "prefix":
                prefix = condition.path
                filtered = [ns for ns in filtered if ns[: len(prefix)] == prefix]
            elif condition.match_type == "suffix":
                suffix = condition.path
                filtered = [ns for ns in filtered if ns[-len(suffix) :] == suffix]

        # Apply max_depth
        if op.max_depth is not None:
            filtered = [ns[: op.max_depth] for ns in filtered]
            # Remove duplicates after truncation
            filtered = list(dict.fromkeys(filtered))

        return sorted(filtered)

    def _apply_filter(
        self, items: list[dict[str, Any]], filter_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply filter to items.

        Args:
            items: List of DynamoDB items.
            filter_dict: Filter criteria.

        Returns:
            Filtered list of items.
        """
        filtered_items = []
        for item in items:
            value = item.get("value", {})
            if self._matches_filter(value, filter_dict):
                filtered_items.append(item)
        return filtered_items

    def _matches_filter(self, value: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if value matches filter criteria.

        Args:
            value: Item value dictionary.
            filter_dict: Filter criteria.

        Returns:
            True if value matches filter, False otherwise.
        """
        for key, expected in filter_dict.items():
            if key not in value:
                return False
            if value[key] != expected:
                return False
        return True

    def _put_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        ttl: float | None,
    ) -> None:
        """Put an item into DynamoDB.

        Args:
            namespace: Namespace tuple.
            key: Item key.
            value: Item value dictionary.
            ttl: TTL in minutes (optional).
        """
        composite_key = self._construct_composite_key(namespace, key)
        current_time = datetime.now(timezone.utc).isoformat()

        # Check if item exists to preserve created_at
        existing_item = None
        try:
            response = self.table.get_item(Key={"PK": composite_key[0], "SK": composite_key[1]})
            existing_item = response.get("Item")
        except Exception:
            pass

        item: dict[str, Any] = {
            "PK": composite_key[0],
            "SK": composite_key[1],
            "value": value,
            "created_at": existing_item["created_at"] if existing_item else current_time,
            "updated_at": current_time,
        }

        # Add TTL if configured
        if ttl is not None:
            expires_at = self._calculate_expiry(ttl)
            if expires_at:
                item["expires_at"] = expires_at

        try:
            self.table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Error putting item {namespace}/{key}: {e}")
            raise

    def _delete_item(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from DynamoDB.

        Args:
            namespace: Namespace tuple.
            key: Item key.
        """
        composite_key = self._construct_composite_key(namespace, key)
        try:
            self.table.delete_item(Key={"PK": composite_key[0], "SK": composite_key[1]})
        except Exception as e:
            logger.error(f"Error deleting item {namespace}/{key}: {e}")
            raise

    def _refresh_ttl(self, pk: str, sk: str) -> None:
        """Refresh TTL for an item.

        Args:
            pk: Partition key.
            sk: Sort key.
        """
        if not self.ttl_config or not self.ttl_config.get("refresh_on_read"):
            return

        default_ttl = self.ttl_config.get("default_ttl")
        if default_ttl is None:
            return

        expires_at = self._calculate_expiry(default_ttl)
        if expires_at is None:
            return

        try:
            self.table.update_item(
                Key={"PK": pk, "SK": sk},
                UpdateExpression="SET expires_at = :expires_at, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ":expires_at": expires_at,
                    ":updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"Error refreshing TTL for {pk}/{sk}: {e}")

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async batch operations are not supported.
        
        DynamoDBStore only supports synchronous operations.
        
        Raises:
            NotImplementedError: Always raised for this synchronous store.
        """
        raise NotImplementedError(
            "Async batch operations are not supported by DynamoDBStore. "
            "This is a synchronous store implementation."
        )
