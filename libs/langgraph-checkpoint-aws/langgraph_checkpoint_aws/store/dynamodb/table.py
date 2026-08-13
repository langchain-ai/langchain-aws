"""Table and vector-index provisioning for the DynamoDB store.

Split out of ``base.py`` so the store class stays focused on BaseStore
operations. ``DynamoDBTableSetupMixin`` carries ``setup()`` and everything it
needs: table creation, vector-index creation and retrofit, the index waiter,
and TTL enablement.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from botocore.exceptions import ClientError

from .exceptions import TableCreationError, ValidationError
from .vector import _VECTOR_ATTR, _VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)


class DynamoDBTableSetupMixin:
    """Table/index provisioning behaviour mixed into ``DynamoDBStore``."""

    # Provided by DynamoDBStore.__init__
    client: Any
    table_name: str
    ttl_config: Any
    max_read_capacity_units: int | None
    max_write_capacity_units: int | None
    _dims: int | None
    _embeddings: Any
    _distance_function: str

    @property
    def has_vector_index(self) -> bool:
        """Whether a vector index is configured on this store."""
        return self._embeddings is not None and self._dims is not None

    def setup(self) -> None:
        """Set up the DynamoDB table and vector index (if configured).

        Creates the DynamoDB table if it doesn't exist, enables TTL if
        configured, and creates a vector index for semantic search if
        index config is provided.

        Raises:
            TableCreationError: If table creation fails.
        """
        try:
            resp = self.client.describe_table(TableName=self.table_name)
            logger.info(f"DynamoDB table '{self.table_name}' already exists.")

            if self.ttl_config:
                self._enable_ttl()

            # Check if vector index needs to be created
            if self.has_vector_index:
                existing_indexes = cast(
                    "list[dict[str, Any]]",
                    resp["Table"].get("VectorIndexes", []),
                )
                existing = next(
                    (
                        i
                        for i in existing_indexes
                        if i.get("IndexName") == _VECTOR_INDEX_NAME
                    ),
                    None,
                )
                if existing is None:
                    self._create_vector_index()
                else:
                    self._validate_existing_vector_index(existing)

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"Creating DynamoDB table '{self.table_name}'...")
                self._create_table()
            else:
                raise TableCreationError(
                    f"Failed to check/create table '{self.table_name}': {e}"
                ) from e

    def _create_table(self) -> None:
        """Create the DynamoDB table with appropriate configuration."""
        try:
            create_params: dict[str, Any] = {
                "TableName": self.table_name,
                "KeySchema": [
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                "AttributeDefinitions": [
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                "BillingMode": "PAY_PER_REQUEST",
            }

            # Only cap on-demand throughput when the caller explicitly asks.
            # An unconditional low cap (previously 10 RCU/WCU) throttles any
            # real agent-memory workload; leaving it unset means uncapped.
            if (
                self.max_read_capacity_units is not None
                or self.max_write_capacity_units is not None
            ):
                on_demand: dict[str, int] = {}
                if self.max_read_capacity_units is not None:
                    on_demand["MaxReadRequestUnits"] = self.max_read_capacity_units
                if self.max_write_capacity_units is not None:
                    on_demand["MaxWriteRequestUnits"] = self.max_write_capacity_units
                create_params["OnDemandThroughput"] = on_demand

            # Add vector index at table creation time if configured
            if self.has_vector_index:
                create_params["VectorIndexes"] = [
                    {
                        "IndexName": _VECTOR_INDEX_NAME,
                        "VectorAttribute": {"AttributeName": _VECTOR_ATTR},
                        "SearchSchema": [
                            {
                                "AttributeName": "PK",
                                "SearchSchemaElementType": "HASH",
                            },
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                        "Dimensions": self._dims,
                        "DistanceFunction": self._distance_function,
                    }
                ]

            self.client.create_table(**create_params)
            waiter = self.client.get_waiter("table_exists")
            waiter.wait(TableName=self.table_name)
            logger.info(f"DynamoDB table '{self.table_name}' created successfully.")

            # Wait for vector index to become active
            if self.has_vector_index:
                self._wait_for_vector_index()

            if self.ttl_config:
                self._enable_ttl()

        except Exception as e:
            raise TableCreationError(
                f"Failed to create table '{self.table_name}': {e}"
            ) from e

    def _validate_existing_vector_index(self, index: dict[str, Any]) -> None:
        """Check an existing vector index matches this store's configuration.

        ``Dimensions`` and ``DistanceFunction`` are immutable after index
        creation, so a store configured differently from the index it points at
        cannot be reconciled at runtime. Both mismatches are silent by default:
        a wrong dimension count fails later at write time with a service-level
        error that doesn't name the index, and a wrong distance function never
        fails at all. The service scores using the index's metric while
        ``_distance_to_score`` converts using this store's setting, so
        ``search()`` returns wrongly-scaled scores in the right order.

        Raises:
            ValidationError: If dimensions or distance function disagree.
        """
        existing_dims = index.get("Dimensions")
        if existing_dims is not None and existing_dims != self._dims:
            raise ValidationError(
                f"Vector index '{_VECTOR_INDEX_NAME}' on table "
                f"'{self.table_name}' has Dimensions={existing_dims} but this "
                f"store is configured for {self._dims}. Index dimensions "
                "cannot be changed after creation. Set index 'dims' to "
                f"{existing_dims}, use an embedding model with that output "
                "size, or point the store at a different table."
            )

        existing_fn = index.get("DistanceFunction")
        if existing_fn is not None and existing_fn != self._distance_function:
            raise ValidationError(
                f"Vector index '{_VECTOR_INDEX_NAME}' on table "
                f"'{self.table_name}' has DistanceFunction={existing_fn} but "
                f"this store is configured for {self._distance_function}. The "
                "index metric cannot be changed after creation, and searches "
                f"would be scored by {existing_fn} while scores were converted "
                f"for {self._distance_function}. Set index "
                f"'distance_function' to '{existing_fn}' or point the store at "
                "a different table."
            )

    def _create_vector_index(self) -> None:
        """Add a vector index to an existing table via UpdateTable."""
        try:
            # VectorIndexUpdates requires botocore >= 1.43.64; typeshed stubs lag.
            self.client.update_table(  # type: ignore[call-arg]
                TableName=self.table_name,
                VectorIndexUpdates=[
                    {
                        "Create": {
                            "IndexName": _VECTOR_INDEX_NAME,
                            "VectorAttribute": {"AttributeName": _VECTOR_ATTR},
                            "SearchSchema": [
                                {
                                    "AttributeName": "PK",
                                    "SearchSchemaElementType": "HASH",
                                },
                            ],
                            "Projection": {"ProjectionType": "ALL"},
                            "Dimensions": self._dims,
                            "DistanceFunction": self._distance_function,
                        }
                    }
                ],
            )
            self._wait_for_vector_index()
            logger.info(
                f"Vector index '{_VECTOR_INDEX_NAME}' created on "
                f"table '{self.table_name}'."
            )
        except ClientError as e:
            logger.error(f"Failed to create vector index: {e}")
            raise

    def _wait_for_vector_index(self, timeout: int = 600) -> None:
        """Wait for the vector index to become ACTIVE."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.client.describe_table(TableName=self.table_name)
            for idx in cast(
                "list[dict[str, Any]]", resp["Table"].get("VectorIndexes", [])
            ):
                if idx["IndexName"] == _VECTOR_INDEX_NAME:
                    # ACTIVE alone is not "ready": while Backfilling is true
                    # SearchVectors returns an error (there is no BACKFILLING
                    # status value). Require explicit ACTIVE with Backfilling
                    # false/absent; treat a missing status as not-yet-active.
                    if idx.get("IndexStatus") == "ACTIVE" and not idx.get(
                        "Backfilling", False
                    ):
                        return
                    break
            time.sleep(10)
        raise TableCreationError(
            f"Vector index '{_VECTOR_INDEX_NAME}' did not become ACTIVE "
            f"within {timeout}s"
        )

    def _enable_ttl(self) -> None:
        """Enable TTL on the DynamoDB table."""
        try:
            self.client.update_time_to_live(
                TableName=self.table_name,
                TimeToLiveSpecification={
                    "Enabled": True,
                    "AttributeName": "expires_at",
                },
            )
            logger.info(f"TTL enabled on table '{self.table_name}'.")
        except ClientError as e:
            # TTL might already be enabled or enabling, log but don't fail
            logger.warning(f"Could not enable TTL on table '{self.table_name}': {e}")
