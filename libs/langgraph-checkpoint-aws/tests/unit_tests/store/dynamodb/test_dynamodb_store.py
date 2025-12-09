"""Unit tests for DynamoDB store implementation."""

import os
import time
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
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
    DynamoDBConnectionError,
    TableCreationError,
    ValidationError,
)


# Test constants for async testing
N_ASYNC_CALLS = 5
MOCK_SLEEP_DURATION = 0.1 / N_ASYNC_CALLS
OVERHEAD_DURATION = 0.01
TOTAL_EXPECTED_TIME = MOCK_SLEEP_DURATION + OVERHEAD_DURATION


@pytest.fixture
def mock_boto3_session():
    """Mock boto3 session."""
    session = Mock()
    return session


@pytest.fixture
def mock_dynamodb_resource():
    """Mock DynamoDB resource."""
    resource = Mock()
    return resource


@pytest.fixture
def mock_dynamodb_table():
    """Mock DynamoDB table."""
    table = Mock()
    table.table_name = "test_table"
    return table


@pytest.fixture
def dynamodb_store(mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
    """Create a DynamoDBStore instance with mocked dependencies."""
    with patch("langgraph_checkpoint_aws.store.dynamodb.base.boto3") as mock_boto3:
        mock_boto3_session.resource.return_value = mock_dynamodb_resource
        mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

        with patch("boto3.Session", return_value=mock_boto3_session):
            # Set AWS region env var to satisfy validation
            with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
                store = DynamoDBStore(table_name="test_table")
                store.table = mock_dynamodb_table
                return store


class TestDynamoDBStoreInit:
    """Test DynamoDB store initialization."""

    def test_init_basic(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test basic initialization."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_boto3_session.resource.return_value = mock_dynamodb_resource
            mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

            store = DynamoDBStore(table_name="test_table", region_name="us-east-1")

            assert store.table_name == "test_table"
            assert store.ttl_config is None
            assert store.max_read_capacity_units == 10
            assert store.max_write_capacity_units == 10

    def test_init_with_ttl(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test initialization with TTL config."""
        ttl_config = TTLConfig(default_ttl=60, refresh_on_read=True)

        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_boto3_session.resource.return_value = mock_dynamodb_resource
            mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

            store = DynamoDBStore(table_name="test_table", region_name="us-east-1", ttl=ttl_config)

            assert store.ttl_config == ttl_config

    def test_init_with_custom_capacity(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test initialization with custom capacity units."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_boto3_session.resource.return_value = mock_dynamodb_resource
            mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

            store = DynamoDBStore(
                table_name="test_table",
                region_name="us-east-1",
                max_read_capacity_units=20,
                max_write_capacity_units=30,
            )

            assert store.max_read_capacity_units == 20
            assert store.max_write_capacity_units == 30

    def test_from_conn_string(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test creating store from connection string."""
        with patch("boto3.Session", return_value=mock_boto3_session):
            mock_boto3_session.resource.return_value = mock_dynamodb_resource
            mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

            with DynamoDBStore.from_conn_string("test_table", region_name="us-east-1") as store:
                assert store.table_name == "test_table"

    def test_init_without_region_or_session(self):
        """Test initialization fails when neither region_name nor session provided."""
        # Clear AWS region environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                DynamoDBStore(table_name="test_table")
            
            assert "Either 'boto3_session' or 'region_name' must be provided" in str(exc_info.value)

    def test_init_with_env_region(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test initialization succeeds with AWS_DEFAULT_REGION env var."""
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            with patch("boto3.Session", return_value=mock_boto3_session):
                mock_boto3_session.resource.return_value = mock_dynamodb_resource
                mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

                store = DynamoDBStore(table_name="test_table")
                assert store.table_name == "test_table"

    def test_init_with_aws_region_env(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test initialization succeeds with AWS_REGION env var."""
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            with patch("boto3.Session", return_value=mock_boto3_session):
                mock_boto3_session.resource.return_value = mock_dynamodb_resource
                mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

                store = DynamoDBStore(table_name="test_table")
                assert store.table_name == "test_table"

    def test_init_with_boto3_session(self, mock_boto3_session, mock_dynamodb_resource, mock_dynamodb_table):
        """Test initialization with boto3_session (no region_name required)."""
        mock_boto3_session.resource.return_value = mock_dynamodb_resource
        mock_dynamodb_resource.Table.return_value = mock_dynamodb_table

        store = DynamoDBStore(table_name="test_table", boto3_session=mock_boto3_session)
        assert store.table_name == "test_table"


class TestDynamoDBStoreSetup:
    """Test store setup."""

    def test_setup_table_exists(self, dynamodb_store):
        """Test setup when table already exists."""
        dynamodb_store.table.load.return_value = None

        dynamodb_store.setup()

        dynamodb_store.table.load.assert_called_once()

    def test_setup_table_not_exists(self, dynamodb_store):
        """Test setup when table doesn't exist."""
        from botocore.exceptions import ClientError

        # Mock table.load() to raise ResourceNotFoundException
        error = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "DescribeTable"
        )
        dynamodb_store.table.load.side_effect = error

        # Mock create_table
        new_table = Mock()
        new_table.wait_until_exists = Mock()
        dynamodb_store.dynamodb.create_table = Mock(return_value=new_table)

        dynamodb_store.setup()

        dynamodb_store.dynamodb.create_table.assert_called_once()
        new_table.wait_until_exists.assert_called_once()


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
        """Test mapping DynamoDB item to Item."""
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
        now = datetime.now(timezone.utc)
        dynamodb_store.table.get_item.return_value = {
            "Item": {
                "PK": "users:123",
                "SK": "prefs",
                "value": {"theme": "dark"},
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
        }

        op = GetOp(namespace=("users", "123"), key="prefs", refresh_ttl=False)
        result = dynamodb_store._batch_get_op(op)

        assert result is not None
        assert result.key == "prefs"
        assert result.value == {"theme": "dark"}

    def test_batch_get_op_not_found(self, dynamodb_store):
        """Test batch GetOp when item not found."""
        dynamodb_store.table.get_item.return_value = {}

        op = GetOp(namespace=("users", "123"), key="prefs", refresh_ttl=False)
        result = dynamodb_store._batch_get_op(op)

        assert result is None

    def test_batch_put_op(self, dynamodb_store):
        """Test batch PutOp."""
        dynamodb_store.table.get_item.return_value = {}
        dynamodb_store.table.put_item.return_value = {}

        op = PutOp(
            namespace=("users", "123"),
            key="prefs",
            value={"theme": "dark"},
            index=None,
            ttl=None,
        )
        result = dynamodb_store._batch_put_op(op)

        assert result is None
        dynamodb_store.table.put_item.assert_called_once()

    def test_batch_put_op_delete(self, dynamodb_store):
        """Test batch PutOp for delete."""
        dynamodb_store.table.delete_item.return_value = {}

        op = PutOp(
            namespace=("users", "123"), key="prefs", value=None, index=None, ttl=None
        )
        result = dynamodb_store._batch_put_op(op)

        assert result is None
        dynamodb_store.table.delete_item.assert_called_once()

    def test_batch_search_op(self, dynamodb_store):
        """Test batch SearchOp."""
        now = datetime.now(timezone.utc)
        dynamodb_store.table.query.return_value = {
            "Items": [
                {
                    "PK": "docs",
                    "SK": "doc1",
                    "value": {"text": "Hello"},
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                }
            ]
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
        now = datetime.now(timezone.utc)
        dynamodb_store.table.query.return_value = {
            "Items": [
                {
                    "PK": "docs",
                    "SK": "doc1",
                    "value": {"type": "article", "status": "published"},
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
                {
                    "PK": "docs",
                    "SK": "doc2",
                    "value": {"type": "article", "status": "draft"},
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
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
        dynamodb_store.table.scan.return_value = {
            "Items": [
                {"PK": "users:123"},
                {"PK": "users:456"},
                {"PK": "docs"},
            ]
        }

        op = ListNamespacesOp(
            match_conditions=tuple(), max_depth=None, limit=100, offset=0
        )
        results = dynamodb_store._batch_list_namespaces_op(op)

        assert len(results) == 3
        assert ("users", "123") in results
        assert ("users", "456") in results
        assert ("docs",) in results

    def test_batch_list_namespaces_op_with_prefix(self, dynamodb_store):
        """Test batch ListNamespacesOp with prefix filter."""
        dynamodb_store.table.scan.return_value = {
            "Items": [
                {"PK": "users:123"},
                {"PK": "users:456"},
                {"PK": "docs"},
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
        """Test that abatch GetOp is non-blocking and runs concurrently."""
        def delayed_get(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return Item(value={"v": 1}, key="k1", namespace=("users", "1"), created_at=datetime.now(), updated_at=datetime.now())

        dynamodb_store._batch_get_op = Mock(side_effect=delayed_get)
        ops = [GetOp(namespace=("users", "1"), key="k1")]
        
        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )

    @pytest.mark.asyncio
    async def test_abatch_put_concurrency(self, dynamodb_store):
        """Test that abatch PutOp is non-blocking and runs concurrently."""
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
        """Test that abatch SearchOp is non-blocking and runs concurrently."""
        def delayed_search(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return [SearchItem(value={"v": 1}, key="k1", namespace=("users", "1"), created_at=datetime.now(), updated_at=datetime.now())]

        dynamodb_store._batch_search_op = Mock(side_effect=delayed_search)
        ops = [SearchOp(namespace_prefix=("users",), limit=10)]
        
        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces_concurrency(self, dynamodb_store):
        """Test that abatch ListNamespacesOp is non-blocking and runs concurrently."""
        def delayed_list(*args, **kwargs):
            time.sleep(MOCK_SLEEP_DURATION)
            return [("users", "1")]

        dynamodb_store._batch_list_namespaces_op = Mock(side_effect=delayed_list)
        ops = [ListNamespacesOp(limit=10)]
        
        await self.assert_concurrent_calls_are_faster_than_sequential(
            5, dynamodb_store.abatch, ops
        )
