"""Integration tests comparing sync and async DynamoDBStore behavior.

This test module validates that DynamoDBStore sync and async operations
return consistent results for the same operations.
"""

import uuid
from collections.abc import Generator

import pytest

from langgraph_checkpoint_aws.store.dynamodb import DynamoDBStore

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BOTO3_AVAILABLE,
    reason="boto3 not available. Install with: pip install boto3",
)


def _is_dynamodb_available() -> bool:
    """Check if a DynamoDB Local server is available for testing.

    Verifies availability by making a real DynamoDB API call (list_tables)
    with dummy credentials, rather than just checking if the port is open.
    """
    if not BOTO3_AVAILABLE:
        return False

    from .conftest import (
        DYNAMODB_AWS_ACCESS_KEY_ID,
        DYNAMODB_AWS_SECRET_ACCESS_KEY,
        DYNAMODB_ENDPOINT_URL,
        DYNAMODB_REGION,
    )

    try:
        client = boto3.client(
            "dynamodb",
            region_name=DYNAMODB_REGION,
            endpoint_url=DYNAMODB_ENDPOINT_URL,
            aws_access_key_id=DYNAMODB_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=DYNAMODB_AWS_SECRET_ACCESS_KEY,
        )
        client.list_tables(Limit=1)
        return True
    except Exception:
        return False


DYNAMODB_AVAILABLE = _is_dynamodb_available()


@pytest.fixture
def sync_store(
    dynamodb_endpoint_url: str,
) -> Generator[DynamoDBStore, None, None]:
    """Create a sync DynamoDBStore instance."""
    table_name = f"test_sync_{uuid.uuid4().hex[:8]}"
    store = DynamoDBStore(
        table_name=table_name,
        region_name="us-east-1",
        endpoint_url=dynamodb_endpoint_url,
    )
    store.setup()
    yield store
    try:
        store.client.delete_table(TableName=table_name)
    except Exception:
        pass


@pytest.fixture
def async_store(
    dynamodb_endpoint_url: str,
) -> Generator[DynamoDBStore, None, None]:
    """Create a DynamoDBStore instance for async testing.

    DynamoDBStore uses the same class for both sync and async operations.
    Async operations are executed via run_in_executor.
    """
    table_name = f"test_async_{uuid.uuid4().hex[:8]}"
    store = DynamoDBStore(
        table_name=table_name,
        region_name="us-east-1",
        endpoint_url=dynamodb_endpoint_url,
    )
    store.setup()
    yield store
    try:
        store.client.delete_table(TableName=table_name)
    except Exception:
        pass


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_sync_async_put_get_parity(
    sync_store: DynamoDBStore,
    async_store: DynamoDBStore,
) -> None:
    """Test that sync and async put/get return consistent results."""
    namespace = ("test", "operations")
    key = "test_key"
    value = {"text": "Test value", "number": 42}

    sync_store.put(namespace, key, value)
    sync_result = sync_store.get(namespace, key)

    await async_store.aput(namespace, key, value)
    async_result = await async_store.aget(namespace, key)

    assert sync_result is not None
    assert async_result is not None

    assert sync_result.value == async_result.value == value
    assert sync_result.key == async_result.key == key
    assert sync_result.namespace == async_result.namespace == namespace


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_sync_async_delete_parity(
    sync_store: DynamoDBStore,
    async_store: DynamoDBStore,
) -> None:
    """Test that sync and async delete operations are consistent."""
    namespace = ("test", "delete")

    sync_store.put(namespace, "key1", {"text": "Delete me"})
    assert sync_store.get(namespace, "key1") is not None
    sync_store.delete(namespace, "key1")
    assert sync_store.get(namespace, "key1") is None

    await async_store.aput(namespace, "key2", {"text": "Delete me"})
    assert await async_store.aget(namespace, "key2") is not None
    await async_store.adelete(namespace, "key2")
    assert await async_store.aget(namespace, "key2") is None


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_sync_async_search_parity(
    sync_store: DynamoDBStore,
    async_store: DynamoDBStore,
) -> None:
    """Test that sync and async search return consistent results."""
    test_docs = [
        (
            ("test", "search"),
            "doc1",
            {"text": "Python programming", "category": "tech"},
        ),
        (
            ("test", "search"),
            "doc2",
            {"text": "Data science", "category": "tech"},
        ),
        (
            ("test", "other"),
            "doc3",
            {"text": "Machine learning", "category": "tech"},
        ),
    ]

    for namespace, key, value in test_docs:
        sync_store.put(namespace, key, value)
        await async_store.aput(namespace, key, value)

    # Basic search
    sync_results = sync_store.search(("test", "search"))
    async_results = await async_store.asearch(("test", "search"))

    assert len(sync_results) == len(async_results) == 2
    sync_keys = {r.key for r in sync_results}
    async_keys = {r.key for r in async_results}
    assert sync_keys == async_keys == {"doc1", "doc2"}

    # Search with filter
    sync_results = sync_store.search(("test", "search"), filter={"category": "tech"})
    async_results = await async_store.asearch(
        ("test", "search"), filter={"category": "tech"}
    )

    assert len(sync_results) == len(async_results) == 2
    sync_keys = {r.key for r in sync_results}
    async_keys = {r.key for r in async_results}
    assert sync_keys == async_keys == {"doc1", "doc2"}

    # Search with limit and offset
    sync_results = sync_store.search(("test", "search"), limit=1, offset=0)
    async_results = await async_store.asearch(("test", "search"), limit=1, offset=0)
    assert len(sync_results) == len(async_results) == 1

    sync_results = sync_store.search(("test", "search"), limit=2, offset=1)
    async_results = await async_store.asearch(("test", "search"), limit=2, offset=1)
    assert len(sync_results) == len(async_results) == 1


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_sync_async_list_namespaces_parity(
    sync_store: DynamoDBStore,
    async_store: DynamoDBStore,
) -> None:
    """Test that sync and async list_namespaces return consistent results."""
    namespaces = [
        ("test", "ns1"),
        ("test", "ns2"),
        ("other", "ns3"),
    ]

    for namespace in namespaces:
        sync_store.put(namespace, "key", {"text": "test"})
        await async_store.aput(namespace, "key", {"text": "test"})

    sync_namespaces = sync_store.list_namespaces()
    async_namespaces = await async_store.alist_namespaces()

    sync_ns_set = set(sync_namespaces)
    async_ns_set = set(async_namespaces)

    assert sync_ns_set == async_ns_set
    assert len(sync_ns_set) == 3

    # With prefix
    sync_namespaces = sync_store.list_namespaces(prefix=("test",))
    async_namespaces = await async_store.alist_namespaces(prefix=("test",))

    sync_ns_set = set(sync_namespaces)
    async_ns_set = set(async_namespaces)

    assert sync_ns_set == async_ns_set
    assert len(sync_ns_set) == 2


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_sync_async_batch_parity(
    sync_store: DynamoDBStore,
    async_store: DynamoDBStore,
) -> None:
    """Test that sync batch and async abatch return consistent results."""
    from langgraph.store.base import GetOp, Item, PutOp, SearchItem, SearchOp

    namespace = ("test", "batch_parity")

    # Put via batch
    put_ops = [
        PutOp(
            namespace=namespace,
            key=f"doc{i}",
            value={"text": f"Document {i}"},
        )
        for i in range(3)
    ]

    sync_store.batch(put_ops)
    await async_store.abatch(put_ops)

    # Get via batch
    get_ops = [GetOp(namespace=namespace, key=f"doc{i}") for i in range(3)]

    sync_results = sync_store.batch(get_ops)
    async_results = await async_store.abatch(get_ops)

    assert len(sync_results) == len(async_results) == 3
    for sync_r, async_r in zip(sync_results, async_results, strict=True):
        assert isinstance(sync_r, Item)
        assert isinstance(async_r, Item)
        assert sync_r.value == async_r.value

    # Search via batch
    search_ops = [
        SearchOp(namespace_prefix=namespace, limit=10),
    ]

    sync_search = sync_store.batch(search_ops)
    async_search = await async_store.abatch(search_ops)

    sync_search_items = sync_search[0]
    async_search_items = async_search[0]
    assert isinstance(sync_search_items, list)
    assert isinstance(async_search_items, list)
    assert len(sync_search_items) == len(async_search_items) == 3
    sync_keys = {r.key for r in sync_search_items if isinstance(r, SearchItem)}
    async_keys = {r.key for r in async_search_items if isinstance(r, SearchItem)}
    assert sync_keys == async_keys
