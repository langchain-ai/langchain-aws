"""Integration tests for the DynamoDBStore implementation."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Generator
from datetime import datetime

import pytest

from langgraph_checkpoint_aws.store.dynamodb import DynamoDBStore
from langgraph_checkpoint_aws.store.dynamodb.exceptions import ValidationError

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
    return f"test_store_{uuid.uuid4().hex[:8]}"


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
    # Cleanup: delete the table after test
    try:
        store.client.delete_table(TableName=table_name)
    except Exception:
        pass


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_put_and_get(store: DynamoDBStore) -> None:
    """Test basic put and get operations with DynamoDBStore."""
    namespace = ("test", "docs")
    key = "doc1"
    value = {"text": "Hello world", "tags": ["test"]}

    store.put(namespace, key, value)

    item = store.get(namespace, key)
    assert item is not None
    assert item.value == value
    assert item.key == key
    assert item.namespace == namespace
    assert isinstance(item.created_at, datetime)
    assert isinstance(item.updated_at, datetime)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_put_update_preserves_created_at(store: DynamoDBStore) -> None:
    """Test that updating an item preserves its created_at timestamp."""
    namespace = ("test", "docs")
    key = "doc1"

    store.put(namespace, key, {"version": 1})
    item1 = store.get(namespace, key)
    assert item1 is not None

    time.sleep(0.1)

    store.put(namespace, key, {"version": 2})
    item2 = store.get(namespace, key)
    assert item2 is not None

    assert item2.value == {"version": 2}
    assert item2.created_at == item1.created_at
    assert item2.updated_at >= item1.updated_at


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_get_nonexistent(store: DynamoDBStore) -> None:
    """Test getting a non-existent item returns None."""
    item = store.get(("test",), "nonexistent")
    assert item is None


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_delete(store: DynamoDBStore) -> None:
    """Test delete operation with DynamoDBStore."""
    namespace = ("test", "delete")
    key = "doc1"
    value = {"text": "Delete me"}

    store.put(namespace, key, value)
    item = store.get(namespace, key)
    assert item is not None

    store.delete(namespace, key)

    item = store.get(namespace, key)
    assert item is None


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search(store: DynamoDBStore) -> None:
    """Test search functionality with DynamoDBStore."""
    docs = [
        (("test", "search"), "doc1", {"text": "Document 1", "type": "test"}),
        (("test", "search"), "doc2", {"text": "Document 2", "type": "test"}),
        (("test", "search"), "doc3", {"text": "Document 3", "type": "test"}),
    ]

    for namespace, key, value in docs:
        store.put(namespace, key, value)

    results = store.search(("test", "search"))
    assert len(results) == 3
    assert all(isinstance(r, SearchItem) for r in results)
    assert {r.key for r in results} == {"doc1", "doc2", "doc3"}


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search_with_filter(store: DynamoDBStore) -> None:
    """Test search with filter."""
    store.put(("test", "filter"), "doc1", {"type": "article", "status": "published"})
    store.put(("test", "filter"), "doc2", {"type": "article", "status": "draft"})
    store.put(("test", "filter"), "doc3", {"type": "blog", "status": "published"})

    results = store.search(("test", "filter"), filter={"status": "published"})
    assert len(results) == 2
    assert all(r.value["status"] == "published" for r in results)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search_with_limit(store: DynamoDBStore) -> None:
    """Test search with limit."""
    for i in range(5):
        store.put(("test", "limit"), f"doc{i}", {"text": f"Document {i}"})

    results = store.search(("test", "limit"), limit=3)
    assert len(results) == 3


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search_with_offset(store: DynamoDBStore) -> None:
    """Test search with offset."""
    for i in range(5):
        store.put(("test", "offset"), f"doc{i}", {"text": f"Document {i}"})

    all_results = store.search(("test", "offset"), limit=10)
    offset_results = store.search(("test", "offset"), limit=10, offset=2)

    assert len(offset_results) == len(all_results) - 2


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search_empty_namespace(store: DynamoDBStore) -> None:
    """Test search on an empty namespace returns no results."""
    results = store.search(("nonexistent", "namespace"))
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_list_namespaces(store: DynamoDBStore) -> None:
    """Test namespace listing with DynamoDBStore."""
    test_data = [
        (("test", "a", "1"), "doc1", {"text": "a1"}),
        (("test", "a", "2"), "doc2", {"text": "a2"}),
        (("test", "b", "1"), "doc3", {"text": "b1"}),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    namespaces = store.list_namespaces(prefix=("test",))
    assert len(namespaces) == 3
    assert all(ns[0] == "test" for ns in namespaces)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_list_namespaces_max_depth(store: DynamoDBStore) -> None:
    """Test namespace listing with max_depth."""
    test_data = [
        (("test", "a", "1"), "doc1", {"text": "a1"}),
        (("test", "a", "2"), "doc2", {"text": "a2"}),
        (("test", "b", "1"), "doc3", {"text": "b1"}),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    namespaces = store.list_namespaces(prefix=("test",), max_depth=2)
    assert all(len(ns) <= 2 for ns in namespaces)
    # After truncation and dedup, should have ("test", "a") and ("test", "b")
    assert len(namespaces) == 2


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_list_namespaces_with_suffix(store: DynamoDBStore) -> None:
    """Test namespace listing with suffix filter."""
    test_data = [
        (("test", "a", "end"), "doc1", {"text": "a1"}),
        (("test", "b", "end"), "doc2", {"text": "b1"}),
        (("test", "c", "other"), "doc3", {"text": "c1"}),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    namespaces = store.list_namespaces(suffix=("end",))
    assert len(namespaces) == 2
    assert all(ns[-1] == "end" for ns in namespaces)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_list_namespaces_with_limit_offset(
    store: DynamoDBStore,
) -> None:
    """Test namespace listing with limit and offset."""
    for i in range(5):
        store.put(("test", f"ns{i}"), "doc", {"text": f"ns{i}"})

    all_namespaces = store.list_namespaces(prefix=("test",))
    paginated = store.list_namespaces(prefix=("test",), limit=2, offset=1)

    assert len(paginated) == 2
    assert paginated == all_namespaces[1:3]


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_batch_operations(store: DynamoDBStore) -> None:
    """Test batch operations with DynamoDBStore using the batch() API."""
    from langgraph.store.base import GetOp, PutOp, SearchOp

    namespace = ("test", "batch")

    # Put via batch
    put_ops = [
        PutOp(
            namespace=namespace,
            key=f"doc{i}",
            value={"text": f"Document {i}"},
        )
        for i in range(3)
    ]
    store.batch(put_ops)

    # Get via batch
    get_ops = [GetOp(namespace=namespace, key=f"doc{i}") for i in range(3)]
    results = store.batch(get_ops)

    assert len(results) == 3
    assert all(isinstance(r, Item) for r in results)
    assert all(r is not None for r in results)
    for i, r in enumerate(results):
        assert isinstance(r, Item)
        assert r.value == {"text": f"Document {i}"}

    # Search via batch
    search_ops = [SearchOp(namespace_prefix=namespace, limit=10)]
    search_results = store.batch(search_ops)

    search_items = search_results[0]
    assert isinstance(search_items, list)
    assert len(search_items) == 3
    assert all(isinstance(r, SearchItem) for r in search_items)


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_complex_values(store: DynamoDBStore) -> None:
    """Test storing and retrieving complex nested values."""
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

    store.put(namespace, key, value)

    item = store.get(namespace, key)
    assert item is not None
    assert item.value["string"] == "hello"
    assert item.value["number"] == 42
    assert item.value["boolean"] is True
    assert item.value["list"] == [1, 2, 3]
    assert item.value["nested"]["c"]["d"] == "e"


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_from_table_name(dynamodb_endpoint_url: str, table_name: str) -> None:
    """Test creating store via from_table_name context manager."""
    with DynamoDBStore.from_table_name(
        table_name,
        region_name="us-east-1",
        endpoint_url=dynamodb_endpoint_url,
    ) as store:
        store.setup()

        store.put(("test",), "doc1", {"text": "Hello"})
        item = store.get(("test",), "doc1")
        assert item is not None
        assert item.value == {"text": "Hello"}

    # Cleanup
    try:
        import boto3

        client = boto3.client(
            "dynamodb",
            region_name="us-east-1",
            endpoint_url=dynamodb_endpoint_url,
        )
        client.delete_table(TableName=table_name)
    except Exception:
        pass


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_setup_idempotent(dynamodb_endpoint_url: str, table_name: str) -> None:
    """Test that setup() is idempotent and can be called multiple times."""
    store = DynamoDBStore(
        table_name=table_name,
        region_name="us-east-1",
        endpoint_url=dynamodb_endpoint_url,
    )

    store.setup()
    store.setup()  # Should not raise

    store.put(("test",), "doc1", {"text": "Hello"})
    item = store.get(("test",), "doc1")
    assert item is not None

    try:
        store.client.delete_table(TableName=table_name)
    except Exception:
        pass


def test_sync_init_without_region_raises() -> None:
    """Test that init without region or session raises ValidationError."""
    from unittest.mock import patch

    env_without_region = {
        k: v
        for k, v in os.environ.items()
        if k not in ("AWS_DEFAULT_REGION", "AWS_REGION")
    }
    with patch.dict(os.environ, env_without_region, clear=True):
        with pytest.raises(ValidationError):
            DynamoDBStore(table_name="test_table")


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_multiple_filter_conditions(store: DynamoDBStore) -> None:
    """Test filtering by multiple fields simultaneously."""
    store.put(
        ("test", "multi"),
        "doc1",
        {"user": "alice", "type": "fact", "importance": "high"},
    )
    store.put(
        ("test", "multi"),
        "doc2",
        {"user": "alice", "type": "preference", "importance": "medium"},
    )
    store.put(
        ("test", "multi"),
        "doc3",
        {"user": "bob", "type": "fact", "importance": "high"},
    )

    results = store.search(("test", "multi"), filter={"user": "alice", "type": "fact"})

    assert len(results) == 1
    assert results[0].key == "doc1"
    assert results[0].value["user"] == "alice"
    assert results[0].value["type"] == "fact"


@pytest.mark.skipif(not DYNAMODB_AVAILABLE, reason="DynamoDB Local not available")
def test_sync_search_performance(store: DynamoDBStore) -> None:
    """Test that search completes within a reasonable time."""
    for i in range(20):
        store.put(
            ("test", "perf"),
            f"doc_{i}",
            {
                "user_id": f"user_{i % 5}",
                "type": "fact" if i % 2 == 0 else "preference",
                "content": f"Test document number {i}",
            },
        )

    start_time = time.time()
    results = store.search(("test", "perf"), filter={"type": "fact"}, limit=10)
    duration = time.time() - start_time

    assert duration < 5.0, f"Search took too long: {duration:.2f} seconds"
    assert len(results) > 0
    assert all(r.value["type"] == "fact" for r in results)
