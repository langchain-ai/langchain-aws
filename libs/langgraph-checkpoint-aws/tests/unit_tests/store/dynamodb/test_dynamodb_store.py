"""Unit tests for DynamoDB store implementation."""

import asyncio
import os
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
)

from langgraph_checkpoint_aws.store import DynamoDBStore
from langgraph_checkpoint_aws.store.dynamodb.exceptions import (
    ValidationError,
)

# Test constants for async testing
N_ASYNC_CALLS = 5
MOCK_SLEEP_DURATION = 0.1 / N_ASYNC_CALLS
OVERHEAD_DURATION = 0.01
TOTAL_EXPECTED_TIME = MOCK_SLEEP_DURATION + OVERHEAD_DURATION


def _make_dynamo_item(
    pk: str,
    sk: str,
    value: dict,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict:
    """Build a DynamoDB client-format item with type annotations."""
    now = datetime.now(timezone.utc).isoformat()
    serialized_value: dict = {}
    for k, v in value.items():
        if isinstance(v, str):
            serialized_value[k] = {"S": v}
        else:
            serialized_value[k] = {"N": str(v)}
    item: dict = {
        "PK": {"S": pk},
        "SK": {"S": sk},
        "value": {"M": serialized_value},
        "created_at": {"S": created_at or now},
        "updated_at": {"S": updated_at or now},
    }
    return item


@pytest.fixture
def mock_dynamodb_client():
    """Mock DynamoDB client (low-level)."""
    client = Mock()
    client.get_waiter = Mock(return_value=Mock())
    return client


@pytest.fixture
def dynamodb_store(mock_dynamodb_client):
    """Create a DynamoDBStore instance with mocked dependencies."""
    with patch(
        "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
        return_value=mock_dynamodb_client,
    ):
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            store = DynamoDBStore(table_name="test_table")
            return store


class TestDynamoDBStoreInit:
    """Test DynamoDB store initialization."""

    def test_init_basic(self, mock_dynamodb_client):
        """Test basic initialization."""
        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            store = DynamoDBStore(
                table_name="test_table",
                region_name="us-east-1",
            )

            assert store.table_name == "test_table"
            assert store.ttl_config is None
            assert store.max_read_capacity_units == 10
            assert store.max_write_capacity_units == 10

    def test_init_with_ttl(self, mock_dynamodb_client):
        """Test initialization with TTL config."""
        ttl_config = TTLConfig(default_ttl=60, refresh_on_read=True)

        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            store = DynamoDBStore(
                table_name="test_table",
                region_name="us-east-1",
                ttl=ttl_config,
            )

            assert store.ttl_config == ttl_config

    def test_init_with_custom_capacity(self, mock_dynamodb_client):
        """Test initialization with custom capacity units."""
        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            store = DynamoDBStore(
                table_name="test_table",
                region_name="us-east-1",
                max_read_capacity_units=20,
                max_write_capacity_units=30,
            )

            assert store.max_read_capacity_units == 20
            assert store.max_write_capacity_units == 30

    def test_from_table_name(self, mock_dynamodb_client):
        """Test creating store from connection string."""
        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            with DynamoDBStore.from_table_name(
                "test_table", region_name="us-east-1"
            ) as store:
                assert store.table_name == "test_table"

    def test_from_table_name_with_endpoint_url(self, mock_dynamodb_client):
        """Test creating store from connection string with endpoint_url."""
        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            with DynamoDBStore.from_table_name(
                "test_table",
                region_name="us-east-1",
                endpoint_url="http://localhost:8000",
            ) as store:
                assert store.table_name == "test_table"

    def test_init_without_region_or_session(self):
        """Test init fails without region_name nor session."""
        # Clear AWS region environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                DynamoDBStore(table_name="test_table")

            assert "Either 'boto3_session' or 'region_name' must be provided" in str(
                exc_info.value
            )

    def test_init_with_env_region(self, mock_dynamodb_client):
        """Test init succeeds with AWS_DEFAULT_REGION env var."""
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            with patch(
                "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
                return_value=mock_dynamodb_client,
            ):
                store = DynamoDBStore(table_name="test_table")
                assert store.table_name == "test_table"

    def test_init_with_aws_region_env(self, mock_dynamodb_client):
        """Test init succeeds with AWS_REGION env var."""
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            with patch(
                "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
                return_value=mock_dynamodb_client,
            ):
                store = DynamoDBStore(table_name="test_table")
                assert store.table_name == "test_table"

    def test_init_with_boto3_session(self, mock_dynamodb_client):
        """Test init with boto3_session (no region_name required)."""
        mock_session = Mock()
        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=mock_dynamodb_client,
        ):
            store = DynamoDBStore(
                table_name="test_table",
                boto3_session=mock_session,
            )
            assert store.table_name == "test_table"

    def test_init_passes_client_kwargs(self):
        """Test endpoint_url and boto_config are passed correctly."""
        from botocore.config import Config

        mock_session = Mock()
        boto_config = Config(retries={"max_attempts": 3})

        with patch(
            "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
            return_value=Mock(),
        ) as mock_create:
            DynamoDBStore(
                table_name="test_table",
                boto3_session=mock_session,
                endpoint_url="http://localhost:8000",
                boto_config=boto_config,
            )

            mock_create.assert_called_once_with(
                session=mock_session,
                region_name=None,
                endpoint_url="http://localhost:8000",
                boto_config=boto_config,
            )


class TestDynamoDBStoreSetup:
    """Test store setup."""

    def test_setup_table_exists(self, dynamodb_store):
        """Test setup when table already exists."""
        dynamodb_store.client.describe_table.return_value = {
            "Table": {"TableName": "test_table"}
        }

        dynamodb_store.setup()

        dynamodb_store.client.describe_table.assert_called_once_with(
            TableName="test_table"
        )

    def test_setup_table_not_exists(self, dynamodb_store):
        """Test setup when table doesn't exist."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "DescribeTable",
        )
        dynamodb_store.client.describe_table.side_effect = error

        dynamodb_store.client.create_table.return_value = {}
        mock_waiter = Mock()
        dynamodb_store.client.get_waiter.return_value = mock_waiter

        dynamodb_store.setup()

        dynamodb_store.client.create_table.assert_called_once()
        dynamodb_store.client.get_waiter.assert_called_once_with("table_exists")
        mock_waiter.wait.assert_called_once_with(TableName="test_table")


class TestDynamoDBStoreOperations:
    """Test store operations."""

    def test_construct_composite_key(self, dynamodb_store):
        """Test composite key construction."""
        pk, sk = dynamodb_store._construct_composite_key(("users", "123"), "prefs")

        assert pk == "users:123"
        assert sk == "prefs"

    def test_deconstruct_namespace(self, dynamodb_store):
        """Test namespace deconstruction."""
        namespace = dynamodb_store._deconstruct_namespace("users:123")

        assert namespace == ("users", "123")

    def test_deconstruct_namespace_empty(self, dynamodb_store):
        """Test deconstruction of empty namespace."""
        namespace = dynamodb_store._deconstruct_namespace("")

        assert namespace == ()

    def test_deconstruct_namespace_single(self, dynamodb_store):
        """Test deconstruction of single-level namespace."""
        namespace = dynamodb_store._deconstruct_namespace("users")

        assert namespace == ("users",)

    def test_map_to_item(self, dynamodb_store):
        """Test mapping deserialized DynamoDB item to Item."""
        now = datetime.now(timezone.utc)
        result_dict = {
            "PK": "users:123",
            "SK": "prefs",
            "value": {"theme": "dark"},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        item = dynamodb_store._map_to_item(result_dict)

        assert isinstance(item, Item)
        assert item.namespace == ("users", "123")
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}

    def test_deserialize_item(self, dynamodb_store):
        """Test deserializing a DynamoDB client item."""
        now = datetime.now(timezone.utc).isoformat()
        raw_item = {
            "PK": {"S": "users:123"},
            "SK": {"S": "prefs"},
            "value": {"M": {"theme": {"S": "dark"}}},
            "created_at": {"S": now},
            "updated_at": {"S": now},
        }

        deserialized = dynamodb_store._deserialize_item(raw_item)

        assert deserialized["PK"] == "users:123"
        assert deserialized["SK"] == "prefs"
        assert deserialized["value"] == {"theme": "dark"}
        assert deserialized["created_at"] == now

    def test_calculate_expiry(self, dynamodb_store):
        """Test TTL expiry calculation."""
        expiry = dynamodb_store._calculate_expiry(60)

        assert expiry is not None
        assert expiry > datetime.now(timezone.utc).timestamp()

    def test_calculate_expiry_none(self, dynamodb_store):
        """Test TTL expiry calculation with None."""
        expiry = dynamodb_store._calculate_expiry(None)

        assert expiry is None


class TestDynamoDBStoreBatch:
    """Test batch operations."""

    def test_batch_get_op(self, dynamodb_store):
        """Test batch GetOp."""
        now = datetime.now(timezone.utc).isoformat()
        dynamodb_store.client.get_item.return_value = {
            "Item": _make_dynamo_item("users:123", "prefs", {"theme": "dark"}, now, now)
        }

        op = GetOp(
            namespace=("users", "123"),
            key="prefs",
            refresh_ttl=False,
        )
        result = dynamodb_store._batch_get_op(op)

        assert result is not None
        assert result.key == "prefs"
        assert result.value == {"theme": "dark"}

    def test_batch_get_op_not_found(self, dynamodb_store):
        """Test batch GetOp when item not found."""
        dynamodb_store.client.get_item.return_value = {}

        op = GetOp(
            namespace=("users", "123"),
            key="prefs",
            refresh_ttl=False,
        )
        result = dynamodb_store._batch_get_op(op)

        assert result is None

    def test_batch_put_op(self, dynamodb_store):
        """Test batch PutOp."""
        dynamodb_store.client.get_item.return_value = {}
        dynamodb_store.client.put_item.return_value = {}

        op = PutOp(
            namespace=("users", "123"),
            key="prefs",
            value={"theme": "dark"},
            index=None,
            ttl=None,
        )
        result = dynamodb_store._batch_put_op(op)

        assert result is None
        dynamodb_store.client.put_item.assert_called_once()

    def test_batch_put_op_delete(self, dynamodb_store):
        """Test batch PutOp for delete."""
        dynamodb_store.client.delete_item.return_value = {}

        op = PutOp(
            namespace=("users", "123"),
            key="prefs",
            value=None,
            index=None,
            ttl=None,
        )
        result = dynamodb_store._batch_put_op(op)

        assert result is None
        dynamodb_store.client.delete_item.assert_called_once()

    def test_batch_search_op(self, dynamodb_store):
        """Test batch SearchOp."""
        now = datetime.now(timezone.utc).isoformat()
        dynamodb_store.client.query.return_value = {
            "Items": [_make_dynamo_item("docs", "doc1", {"text": "Hello"}, now, now)]
        }

        op = SearchOp(
            namespace_prefix=("docs",),
            filter=None,
            limit=10,
            offset=0,
            query=None,
            refresh_ttl=False,
        )
        results = dynamodb_store._batch_search_op(op)

        assert len(results) == 1
        assert isinstance(results[0], SearchItem)
        assert results[0].key == "doc1"

    def test_batch_search_op_with_filter(self, dynamodb_store):
        """Test batch SearchOp with filter."""
        now = datetime.now(timezone.utc).isoformat()
        dynamodb_store.client.query.return_value = {
            "Items": [
                _make_dynamo_item(
                    "docs",
                    "doc1",
                    {"type": "article", "status": "published"},
                    now,
                    now,
                ),
                _make_dynamo_item(
                    "docs",
                    "doc2",
                    {"type": "article", "status": "draft"},
                    now,
                    now,
                ),
            ]
        }

        op = SearchOp(
            namespace_prefix=("docs",),
            filter={"status": "published"},
            limit=10,
            offset=0,
            query=None,
            refresh_ttl=False,
        )
        results = dynamodb_store._batch_search_op(op)

        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].value["status"] == "published"

    def test_batch_list_namespaces_op(self, dynamodb_store):
        """Test batch ListNamespacesOp."""
        dynamodb_store.client.scan.return_value = {
            "Items": [
                {"PK": {"S": "users:123"}},
                {"PK": {"S": "users:456"}},
                {"PK": {"S": "docs"}},
            ]
        }

        op = ListNamespacesOp(
            match_conditions=tuple(),
            max_depth=None,
            limit=100,
            offset=0,
        )
        results = dynamodb_store._batch_list_namespaces_op(op)

        assert len(results) == 3
        assert ("users", "123") in results
        assert ("users", "456") in results
        assert ("docs",) in results

    def test_batch_list_namespaces_op_with_prefix(self, dynamodb_store):
        """Test batch ListNamespacesOp with prefix filter."""
        dynamodb_store.client.scan.return_value = {
            "Items": [
                {"PK": {"S": "users:123"}},
                {"PK": {"S": "users:456"}},
                {"PK": {"S": "docs"}},
            ]
        }

        op = ListNamespacesOp(
            match_conditions=(MatchCondition(match_type="prefix", path=("users",)),),
            max_depth=None,
            limit=100,
            offset=0,
        )
        results = dynamodb_store._batch_list_namespaces_op(op)

        assert len(results) == 2
        assert ("users", "123") in results
        assert ("users", "456") in results


class TestDynamoDBStoreFiltering:
    """Test filtering functionality."""

    def test_matches_filter_true(self, dynamodb_store):
        """Test filter matching returns True."""
        value = {"type": "article", "status": "published"}
        filter_dict = {"type": "article"}

        result = dynamodb_store._matches_filter(value, filter_dict)

        assert result is True

    def test_matches_filter_false(self, dynamodb_store):
        """Test filter matching returns False."""
        value = {"type": "article", "status": "draft"}
        filter_dict = {"status": "published"}

        result = dynamodb_store._matches_filter(value, filter_dict)

        assert result is False

    def test_matches_filter_missing_key(self, dynamodb_store):
        """Test filter matching with missing key."""
        value = {"type": "article"}
        filter_dict = {"status": "published"}

        result = dynamodb_store._matches_filter(value, filter_dict)

        assert result is False

    def test_apply_filter(self, dynamodb_store):
        """Test applying filter to items."""
        items = [
            {"value": {"type": "article", "status": "published"}},
            {"value": {"type": "article", "status": "draft"}},
            {"value": {"type": "blog", "status": "published"}},
        ]
        filter_dict = {"type": "article", "status": "published"}

        filtered = dynamodb_store._apply_filter(items, filter_dict)

        assert len(filtered) == 1
        assert filtered[0]["value"]["status"] == "published"


class TestDynamoDBStoreAsync:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_abatch_get(self, dynamodb_store):
        """Test abatch with GetOp."""
        op = GetOp(namespace=("users", "1"), key="k1")
        item = Item(
            value={"v": 1},
            key="k1",
            namespace=("users", "1"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        dynamodb_store._batch_get_op = Mock(return_value=item)

        results = await dynamodb_store.abatch([op])

        assert len(results) == 1
        assert results[0] == item
        dynamodb_store._batch_get_op.assert_called_once_with(op)

    @pytest.mark.asyncio
    async def test_abatch_put(self, dynamodb_store):
        """Test abatch with PutOp."""
        op = PutOp(namespace=("users", "2"), key="k2", value={"v": 2})

        dynamodb_store._batch_put_op = Mock(return_value=None)

        results = await dynamodb_store.abatch([op])

        assert len(results) == 1
        assert results[0] is None
        dynamodb_store._batch_put_op.assert_called_once_with(op)

    @pytest.mark.asyncio
    async def test_abatch_search(self, dynamodb_store):
        """Test abatch with SearchOp."""
        op = SearchOp(namespace_prefix=("users",), limit=10)
        search_item = SearchItem(
            value={"v": 1},
            key="k1",
            namespace=("users", "1"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        dynamodb_store._batch_search_op = Mock(return_value=[search_item])

        results = await dynamodb_store.abatch([op])

        assert len(results) == 1
        assert results[0] == [search_item]
        dynamodb_store._batch_search_op.assert_called_once_with(op)

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces(self, dynamodb_store):
        """Test abatch with ListNamespacesOp."""
        op = ListNamespacesOp(limit=10)

        dynamodb_store._batch_list_namespaces_op = Mock(return_value=[("users", "1")])

        results = await dynamodb_store.abatch([op])

        assert len(results) == 1
        assert results[0] == [("users", "1")]
        dynamodb_store._batch_list_namespaces_op.assert_called_once_with(op)

    async def assert_concurrent_calls_are_faster_than_sequential(
        self, n_async_calls: int, func, *args, **kwargs
    ) -> None:
        """Helper to run n async tasks concurrently."""
        tasks = [func(*args, **kwargs) for _ in range(n_async_calls)]
        start_time = time.time()
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        assert concurrent_time < TOTAL_EXPECTED_TIME, (
            f"Concurrent execution took {concurrent_time:.2f}s, "
            f"expected < {TOTAL_EXPECTED_TIME}s"
        )

    @pytest.mark.asyncio
    async def test_abatch_get_concurrency(self, dynamodb_store):
        """Test that abatch GetOp runs concurrently."""

        def delayed_get(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return Item(
                value={"v": 1},
                key="k1",
                namespace=("users", "1"),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        dynamodb_store._batch_get_op = Mock(side_effect=delayed_get)
        ops = [GetOp(namespace=("users", "1"), key="k1")]

        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )

    @pytest.mark.asyncio
    async def test_abatch_put_concurrency(self, dynamodb_store):
        """Test that abatch PutOp runs concurrently."""

        def delayed_put(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return None

        dynamodb_store._batch_put_op = Mock(side_effect=delayed_put)
        ops = [PutOp(namespace=("users", "2"), key="k2", value={"v": 2})]

        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )

    @pytest.mark.asyncio
    async def test_abatch_search_concurrency(self, dynamodb_store):
        """Test that abatch SearchOp runs concurrently."""

        def delayed_search(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return [
                SearchItem(
                    value={"v": 1},
                    key="k1",
                    namespace=("users", "1"),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            ]

        dynamodb_store._batch_search_op = Mock(side_effect=delayed_search)
        ops = [SearchOp(namespace_prefix=("users",), limit=10)]

        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces_concurrency(self, dynamodb_store):
        """Test that abatch ListNamespacesOp runs concurrently."""

        def delayed_list(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return [("users", "1")]

        dynamodb_store._batch_list_namespaces_op = Mock(side_effect=delayed_list)
        ops = [ListNamespacesOp(limit=10)]

        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )
