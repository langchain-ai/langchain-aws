"""DynamoDB store implementation for LangGraph.

This module provides a DynamoDB-backed store implementation that extends
the BaseStore class from LangGraph. It offers persistent storage with
hierarchical namespaces, key-value operations, and optional vector search
via DynamoDB Vector Index for semantic long-term memory.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, TypeVar

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.config import Config
from langchain_core.runnables import run_in_executor
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)

from langgraph_checkpoint_aws.checkpoint.dynamodb.utils import create_dynamodb_client

from .exceptions import DynamoDBConnectionError, ValidationError
from .search import DynamoDBSearchMixin
from .table import DynamoDBTableSetupMixin
from .vector import (
    _DEFAULT_FIELDS,
    _VECTOR_ATTR,
    Embeddings,
    _CallableEmbeddingsAdapter,
)

_ItemT = TypeVar("_ItemT", bound=Item)

logger = logging.getLogger(__name__)


class DynamoDBStore(DynamoDBTableSetupMixin, DynamoDBSearchMixin, BaseStore):
    """DynamoDB-backed store implementation for LangGraph.

    This store provides persistent key-value storage using AWS DynamoDB.
    It supports hierarchical namespaces, TTL (time-to-live) for automatic
    item expiration, basic filtering, and optional vector semantic search
    via DynamoDB Vector Index.

    Note: like a global secondary index, the vector index is eventually
    consistent. A ``search()`` issued immediately after ``put()`` may not
    include the just-written item until the index catches up.

    The store uses a single DynamoDB table with the following schema:
    - PK (Partition Key): Namespace (joined with ':')
    - SK (Sort Key): Item key
    - value: The stored dictionary
    - embedding: Vector embedding (when index is configured)
    - created_at: ISO format timestamp
    - updated_at: ISO format timestamp
    - expires_at: Unix timestamp for TTL (optional)

    Examples:
        Basic usage (no vector search):
        ```python
        from langgraph_checkpoint_aws import DynamoDBStore

        store = DynamoDBStore(table_name="my-store-table")
        store.setup()  # Create table if it doesn't exist

        # Store and retrieve data
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        print(item.value)  # {"theme": "dark"}
        ```

        With vector search (long-term semantic memory):
        ```python
        from langgraph_checkpoint_aws import DynamoDBStore
        from langchain_aws import BedrockEmbeddings

        store = DynamoDBStore(
            table_name="my-memory-table",
            index={
                "embed": BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
                "dims": 256,
                "fields": ["text"],  # which value fields to embed
            },
        )
        store.setup()

        # Store a memory
        store.put(("user", "alice"), "mem1", {"text": "Alice prefers dark mode"})

        # Semantic search across memories
        results = store.search(("user", "alice"), query="UI preferences")
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
        ```

    Note:
        Make sure to call `setup()` before first use to create the necessary
        DynamoDB table and vector index (if configured).

    Warning:
        DynamoDB charges are based on read/write capacity and storage.
        Consider using on-demand billing for unpredictable workloads or
        provisioned capacity for consistent traffic patterns.
    """

    supports_ttl = True

    def __init__(
        self,
        table_name: str,
        *,
        region_name: str | None = None,
        boto3_session: boto3.Session | None = None,
        endpoint_url: str | None = None,
        boto_config: Config | None = None,
        ttl: TTLConfig | None = None,
        index: dict[str, Any] | None = None,
        max_read_capacity_units: int | None = None,
        max_write_capacity_units: int | None = None,
    ) -> None:
        """Initialize DynamoDB store.

        Args:
            table_name: Name of the DynamoDB table to use.
            region_name: AWS region name. If not provided along with boto3_session,
                AWS_DEFAULT_REGION or AWS_REGION environment variable must be set.
            boto3_session: Optional boto3 session to use. If not provided, creates
                a new one using region_name or AWS environment variables.
            endpoint_url: Custom endpoint URL for the DynamoDB service.
            boto_config: Botocore config object.
            ttl: Optional TTL configuration for automatic item expiration.
            index: Optional vector index configuration for semantic search.
                Keys:
                - "embed": Embeddings object (LangChain compatible) or callable.
                    Must implement embed_documents() and embed_query().
                - "dims": int — vector dimensions (must match embedding model output).
                - "fields": list[str] — JSON paths within the value to embed.
                    Default ["$"] embeds the entire value as JSON text.
                - "distance_function": str — "COSINE" (default), "EUCLIDEAN",
                    or "DOT_PRODUCT".
            max_read_capacity_units: Maximum read capacity units for on-demand mode.
                Only used when creating a new table. Default is 10.
            max_write_capacity_units: Maximum write capacity units for on-demand mode.
                Only used when creating a new table. Default is 10.

        Raises:
            ValidationError: If neither boto3_session nor region_name is provided
                and AWS region environment variables are not set.
        """
        super().__init__()

        # Validate that either boto3_session, region_name, or AWS env vars are set
        if boto3_session is None and region_name is None:
            if not os.environ.get("AWS_DEFAULT_REGION") and not os.environ.get(
                "AWS_REGION"
            ):
                msg = (
                    "Either 'boto3_session' or 'region_name' must be provided, "
                    "or AWS_DEFAULT_REGION/AWS_REGION environment variable must be "
                    "set. "
                    "Example: DynamoDBStore(table_name='my-table', "
                    "region_name='us-east-1')"
                )
                raise ValidationError(msg)

        self.table_name = table_name
        self.ttl_config = ttl
        self.max_read_capacity_units = max_read_capacity_units
        self.max_write_capacity_units = max_write_capacity_units
        self._type_serializer = TypeSerializer()
        self._type_deserializer = TypeDeserializer()

        # Vector index configuration
        self._index_config = index
        self._embeddings: Embeddings | None = None
        self._dims: int | None = None
        self._embed_fields: list[str] = _DEFAULT_FIELDS
        self._distance_function: str = "COSINE"

        if index:
            embed = index.get("embed")
            if embed is None:
                raise ValidationError(
                    "index config requires 'embed' key with an embeddings object. "
                    "Example: index={'embed': BedrockEmbeddings(...), 'dims': 256}"
                )
            # Support LangGraph's ensure_embeddings pattern
            if isinstance(embed, Embeddings):
                self._embeddings = embed
            elif callable(embed):
                # Wrap callable in a simple adapter
                self._embeddings = _CallableEmbeddingsAdapter(embed)
            else:
                try:
                    from langgraph.store.base.embed import ensure_embeddings

                    self._embeddings = ensure_embeddings(embed)
                except Exception as err:
                    raise ValidationError(
                        f"Could not initialize embeddings from: {embed!r}. "
                        "Provide a LangChain Embeddings object or callable."
                    ) from err

            self._dims = index.get("dims")
            if self._dims is None:
                raise ValidationError(
                    "index config requires 'dims' key specifying vector dimensions. "
                    "Example: index={'embed': ..., 'dims': 256}"
                )
            self._embed_fields = index.get("fields", _DEFAULT_FIELDS)
            self._distance_function = index.get("distance_function", "COSINE")

        # Initialize DynamoDB client using shared utility
        try:
            self.client = create_dynamodb_client(
                session=boto3_session,
                region_name=region_name,
                endpoint_url=endpoint_url,
                boto_config=boto_config,
            )
        except Exception as e:
            raise DynamoDBConnectionError(
                f"Failed to initialize DynamoDB connection: {e}"
            ) from e

    @classmethod
    @contextmanager
    def from_table_name(
        cls,
        table_name: str,
        *,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        boto_config: Config | None = None,
        ttl: TTLConfig | None = None,
        index: dict[str, Any] | None = None,
        max_read_capacity_units: int | None = None,
        max_write_capacity_units: int | None = None,
    ) -> Iterator[DynamoDBStore]:
        """Create a DynamoDB store instance using a context manager.

        Args:
            table_name: Name of the DynamoDB table to use.
            region_name: AWS region name.
            endpoint_url: Custom endpoint URL for the DynamoDB service.
            boto_config: Botocore config object.
            ttl: Optional TTL configuration for automatic item expiration.
            index: Optional vector index configuration for semantic search.
            max_read_capacity_units: Maximum read capacity units.
            max_write_capacity_units: Maximum write capacity units.

        Yields:
            DynamoDBStore: A DynamoDB store instance.
        """
        store = cls(
            table_name=table_name,
            region_name=region_name,
            endpoint_url=endpoint_url,
            boto_config=boto_config,
            ttl=ttl,
            index=index,
            max_read_capacity_units=max_read_capacity_units,
            max_write_capacity_units=max_write_capacity_units,
        )
        try:
            yield store
        finally:
            pass

    def _deserialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a DynamoDB client item to Python native types.

        Args:
            item: DynamoDB item with type annotations
                (e.g., {"PK": {"S": "value"}}).

        Returns:
            Dictionary with Python native values
                (e.g., {"PK": "value"}).
        """
        return {k: self._type_deserializer.deserialize(v) for k, v in item.items()}

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

    def _map_to_item(
        self,
        result_dict: dict[str, Any],
        item_type: type[_ItemT] = Item,  # type: ignore[assignment]
    ) -> _ItemT:
        """Map deserialized DynamoDB item to store Item.

        Args:
            result_dict: Deserialized DynamoDB item dictionary
                (Python native types).
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

    @staticmethod
    def _parse_ts(value: Any) -> datetime:
        """Parse an ISO timestamp, tolerating missing/empty values.

        The vector-search projection may omit a timestamp; fall back to epoch
        rather than raising ValueError on an empty string.
        """
        if not value:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.fromtimestamp(0, tz=timezone.utc)

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
        expiry_seconds = int(
            datetime.now(timezone.utc).timestamp() + (ttl_minutes * 60)
        )
        return expiry_seconds

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations in a batch.

        Args:
            ops: Iterable of operations (GetOp, PutOp, SearchOp,
                ListNamespacesOp).

        Returns:
            List of results corresponding to each operation.
        """
        results: list[Result] = []

        for op in ops:
            result: Result
            if isinstance(op, GetOp):
                result = self._batch_get_op(op)
            elif isinstance(op, PutOp):
                self._batch_put_op(op)
                result = None
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
            response = self.client.get_item(
                TableName=self.table_name,
                Key={
                    "PK": {"S": composite_key[0]},
                    "SK": {"S": composite_key[1]},
                },
            )
            raw_item = response.get("Item")
            if raw_item:
                item = self._deserialize_item(raw_item)
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

    def _apply_filter(
        self,
        items: list[dict[str, Any]],
        filter_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply filter to items.

        Args:
            items: List of deserialized DynamoDB items.
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

    def _matches_filter(
        self,
        value: dict[str, Any],
        filter_dict: dict[str, Any],
    ) -> bool:
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
        """Put an item into DynamoDB, with optional embedding generation.

        Args:
            namespace: Namespace tuple.
            key: Item key.
            value: Item value dictionary.
            ttl: TTL in minutes (optional).
        """
        composite_key = self._construct_composite_key(namespace, key)
        current_time = datetime.now(timezone.utc).isoformat()

        # Single atomic write. created_at is set only on first insert via
        # if_not_exists, so we avoid the previous read-before-write (which
        # doubled RCU/latency and had a TOCTOU race under concurrent puts).
        set_parts = [
            "#value = :value",
            "updated_at = :updated_at",
            "created_at = if_not_exists(created_at, :created_at)",
        ]
        # A re-put must not resurrect attributes from the previous version of
        # the item. put_item (full replace) cleared them implicitly; with a
        # SET-only update a stale expires_at would silently expire the new
        # value and a stale embedding would rank the item by its old content.
        remove_parts: list[str] = []
        ean: dict[str, str] = {"#value": "value"}
        eav: dict[str, Any] = {
            ":value": self._type_serializer.serialize(value),
            ":updated_at": {"S": current_time},
            ":created_at": {"S": current_time},
        }

        # Generate and store embedding if vector index is configured
        if self.has_vector_index:
            text = self._extract_text_for_embedding(value)
            if text:
                assert self._embeddings is not None
                embedding = self._embeddings.embed_documents([text])[0]
                if self._dims is not None and len(embedding) != self._dims:
                    raise ValidationError(
                        f"Embedding dimension mismatch for {namespace}/{key}: "
                        f"model returned {len(embedding)} dims but the vector "
                        f"index is configured for {self._dims}. Set index "
                        f"'dims' to match the embedding model's output size."
                    )
                if not all(math.isfinite(v) for v in embedding):
                    raise ValidationError(
                        f"Embedding for {namespace}/{key} contains non-finite "
                        "values (nan/inf); the DynamoDB N type rejects them."
                    )
                ean["#embedding"] = _VECTOR_ATTR
                eav[":embedding"] = {"L": [{"N": str(v)} for v in embedding]}
                set_parts.append("#embedding = :embedding")
            else:
                # New value has nothing to embed: drop any embedding from a
                # previous version so semantic search cannot match stale text.
                logger.warning(
                    "No text extracted from %s/%s for fields %s; item stored "
                    "WITHOUT an embedding and will not appear in semantic "
                    "search results.",
                    namespace,
                    key,
                    self._embed_fields,
                )
                ean["#embedding"] = _VECTOR_ATTR
                remove_parts.append("#embedding")

        # TTL: set when requested, otherwise clear any expiry left by a
        # previous put so the new value does not inherit a stale deadline.
        expires_at = self._calculate_expiry(ttl) if ttl is not None else None
        if expires_at:
            eav[":expires_at"] = {"N": str(expires_at)}
            set_parts.append("expires_at = :expires_at")
        else:
            remove_parts.append("expires_at")

        update_expression = "SET " + ", ".join(set_parts)
        if remove_parts:
            update_expression += " REMOVE " + ", ".join(remove_parts)

        try:
            self.client.update_item(
                TableName=self.table_name,
                Key={
                    "PK": {"S": composite_key[0]},
                    "SK": {"S": composite_key[1]},
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=ean,
                ExpressionAttributeValues=eav,
            )
        except Exception as e:
            logger.error(f"Error putting item {namespace}/{key}: {e}")
            raise

    def _extract_text_for_embedding(self, value: dict[str, Any]) -> str:
        """Extract text from value dict based on configured fields.

        Args:
            value: The item value dictionary.

        Returns:
            Text string to embed.
        """
        import json

        if self._embed_fields == _DEFAULT_FIELDS or self._embed_fields == ["$"]:
            # Embed entire value as JSON
            return json.dumps(value, default=str)

        # Extract specific fields
        parts = []
        for field in self._embed_fields:
            if field in value:
                v = value[field]
                # JSON for non-strings, consistent with whole-value mode
                # (str(dict) would embed Python repr syntax).
                parts.append(v if isinstance(v, str) else json.dumps(v, default=str))

        return " ".join(parts)

    def _delete_item(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from DynamoDB.

        Args:
            namespace: Namespace tuple.
            key: Item key.
        """
        composite_key = self._construct_composite_key(namespace, key)
        try:
            self.client.delete_item(
                TableName=self.table_name,
                Key={
                    "PK": {"S": composite_key[0]},
                    "SK": {"S": composite_key[1]},
                },
            )
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
            self.client.update_item(
                TableName=self.table_name,
                Key={"PK": {"S": pk}, "SK": {"S": sk}},
                UpdateExpression=(
                    "SET expires_at = :expires_at, updated_at = :updated_at"
                ),
                ExpressionAttributeValues={
                    ":expires_at": {"N": str(expires_at)},
                    ":updated_at": {"S": datetime.now(timezone.utc).isoformat()},
                },
            )
        except Exception as e:
            logger.warning(f"Error refreshing TTL for {pk}/{sk}: {e}")

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch operations asynchronously in parallel.

        This method leverages run_in_executor to execute synchronous DynamoDB
        operations concurrently in a thread pool. Each operation (GetOp,
        PutOp, SearchOp, ListNamespacesOp) is wrapped in an async function
        and executed in parallel using asyncio.gather, to improve throughput
        for batch operations compared to sequential execution.
        TODO: should perform multiple dynamodb put_item with
        batch_write_item for better performance.

        Args:
            ops: Iterable of operations to execute. Supported operation types
                are GetOp, PutOp, SearchOp, and ListNamespacesOp.

        Returns:
            List of results corresponding to the input operations, in the
            same order as the input operations.

        Raises:
            NotImplementedError: If an unsupported operation type is
                encountered.
        """

        async def execute_op(op: Op) -> Result:
            """Execute a single operation in the executor."""
            if isinstance(op, GetOp):
                return await run_in_executor(None, self._batch_get_op, op)
            if isinstance(op, PutOp):
                return await run_in_executor(None, self._batch_put_op, op)
            if isinstance(op, SearchOp):
                return await run_in_executor(None, self._batch_search_op, op)
            if isinstance(op, ListNamespacesOp):
                return await run_in_executor(None, self._batch_list_namespaces_op, op)
            raise NotImplementedError(  # noqa: EM102
                f"Operation type {type(op)} not supported"
            )

        # Execute all operations in parallel
        results = await asyncio.gather(*[execute_op(op) for op in ops])
        return list(results)
