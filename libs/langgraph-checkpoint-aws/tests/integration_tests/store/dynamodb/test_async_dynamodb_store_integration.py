"""Async integration tests for the DynamoDBStore implementation."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from datetime import datetime

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

if BOTO3_AVAILABLE:
    from langgraph.store.base import Item, SearchItem


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
def table_name() -> str:
    """Generate a unique table name for test isolation."""
    return f"test_async_store_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def store(
    dynamodb_endpoint_url: str, table_name: str
) -> Generator[DynamoDBStore, None, None]:
    """Create a DynamoDBStore instance connected to DynamoDB Local."""
    store = DynamoDBStore(
        table_name=table_name,
        region_name="us-east-1",
        endpoint_url=dynamodb_endpoint_url,
        ttl={"default_ttl": 60, "refresh_on_read": True},
    )
    store.setup()
    yield store
    try:
        store.client.delete_table(TableName=table_name)
    except Exception:
        pass


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_put_and_get(store: DynamoDBStore) -> None:
    """Test basic async put and get operations."""
    namespace = ("test", "docs")
    key = "doc1"
    value = {"text": "Hello world", "tags": ["test"]}

    await store.aput(namespace, key, value)

    item = await store.aget(namespace, key)
    assert item is not None
    assert item.value == value
    assert item.key == key
    assert item.namespace == namespace
    assert isinstance(item.created_at, datetime)
    assert isinstance(item.updated_at, datetime)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_get_nonexistent(store: DynamoDBStore) -> None:
    """Test async get of a non-existent item returns None."""
    item = await store.aget(("test",), "nonexistent")
    assert item is None


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_delete(store: DynamoDBStore) -> None:
    """Test async delete operation."""
    namespace = ("test", "delete")
    key = "doc1"
    value = {"text": "Delete me"}

    await store.aput(namespace, key, value)
    item = await store.aget(namespace, key)
    assert item is not None

    await store.adelete(namespace, key)

    item = await store.aget(namespace, key)
    assert item is None


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_search(store: DynamoDBStore) -> None:
    """Test async search functionality."""
    docs = [
        (("test", "search"), "doc1", {"text": "Document 1", "type": "test"}),
        (("test", "search"), "doc2", {"text": "Document 2", "type": "test"}),
        (("test", "search"), "doc3", {"text": "Document 3", "type": "test"}),
    ]

    for namespace, key, value in docs:
        await store.aput(namespace, key, value)

    results = await store.asearch(("test", "search"))
    assert len(results) == 3
    assert all(isinstance(r, SearchItem) for r in results)
    assert {r.key for r in results} == {"doc1", "doc2", "doc3"}


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_search_with_filter(store: DynamoDBStore) -> None:
    """Test async search with filter."""
    await store.aput(
        ("test", "filter"),
        "doc1",
        {"type": "article", "status": "published"},
    )
    await store.aput(
        ("test", "filter"),
        "doc2",
        {"type": "article", "status": "draft"},
    )
    await store.aput(
        ("test", "filter"),
        "doc3",
        {"type": "blog", "status": "published"},
    )

    results = await store.asearch(("test", "filter"), filter={"status": "published"})
    assert len(results) == 2
    assert all(r.value["status"] == "published" for r in results)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_list_namespaces(store: DynamoDBStore) -> None:
    """Test async namespace listing."""
    test_data = [
        (("test", "a", "1"), "doc1", {"text": "a1"}),
        (("test", "a", "2"), "doc2", {"text": "a2"}),
        (("test", "b", "1"), "doc3", {"text": "b1"}),
    ]

    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    namespaces = await store.alist_namespaces(prefix=("test",))
    assert len(namespaces) == 3
    assert all(ns[0] == "test" for ns in namespaces)

    namespaces = await store.alist_namespaces(prefix=("test",), max_depth=2)
    assert all(len(ns) <= 2 for ns in namespaces)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_batch_operations(store: DynamoDBStore) -> None:
    """Test async batch operations."""
    namespace = ("test", "batch")
    items = [(namespace, f"doc{i}", {"text": f"Document {i}"}) for i in range(3)]

    for ns, key, value in items:
        await store.aput(ns, key, value)

    results = await asyncio.gather(*[store.aget(ns, key) for ns, key, _ in items])

    assert len(results) == 3
    assert all(isinstance(r, Item) for r in results)
    assert all(r is not None for r in results)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_concurrent_puts(store: DynamoDBStore) -> None:
    """Test concurrent async put operations."""
    namespace = ("test", "concurrent")

    async def put_item(i: int) -> None:
        await store.aput(namespace, f"doc{i}", {"index": i})

    await asyncio.gather(*[put_item(i) for i in range(10)])

    for i in range(10):
        item = await store.aget(namespace, f"doc{i}")
        assert item is not None
        assert item.value["index"] == i


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_search_empty_namespace(store: DynamoDBStore) -> None:
    """Test async search on an empty namespace."""
    results = await store.asearch(("nonexistent", "namespace"))
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_complex_values(store: DynamoDBStore) -> None:
    """Test storing and retrieving complex nested values asynchronously."""
    namespace = ("test", "complex")
    key = "nested"
    value = {
        "string": "hello",
        "number": 42,
        "float_val": 3.14,
        "boolean": True,
        "list": [1, 2, 3],
        "nested": {"a": "b", "c": {"d": "e"}},
    }

    await store.aput(namespace, key, value)

    item = await store.aget(namespace, key)
    assert item is not None
    assert item.value["string"] == "hello"
    assert item.value["number"] == 42
    assert item.value["boolean"] is True
    assert item.value["list"] == [1, 2, 3]
    assert item.value["nested"]["c"]["d"] == "e"


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
@pytest.mark.asyncio
async def test_async_multiple_filter_conditions(
    store: DynamoDBStore,
) -> None:
    """Test async filtering by multiple fields simultaneously."""
    await store.aput(
        ("test", "multi"),
        "doc1",
        {"user": "alice", "type": "fact", "importance": "high"},
    )
    await store.aput(
        ("test", "multi"),
        "doc2",
        {"user": "alice", "type": "preference", "importance": "medium"},
    )
    await store.aput(
        ("test", "multi"),
        "doc3",
        {"user": "bob", "type": "fact", "importance": "high"},
    )

    results = await store.asearch(
        ("test", "multi"), filter={"user": "alice", "type": "fact"}
    )

    assert len(results) == 1
    assert results[0].key == "doc1"
    assert results[0].value["user"] == "alice"
    assert results[0].value["type"] == "fact"
