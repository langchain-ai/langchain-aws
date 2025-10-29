"""Async tests for the Valkey store implementation."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from langgraph_checkpoint_aws import (
    AsyncValkeyStore,
    ValkeyIndexConfig,
    ValkeyValidationError,
)

# Check for optional dependencies
try:
    from valkey import Valkey

    VALKEY_AVAILABLE = True
except ImportError:
    Valkey = None  # type: ignore[assignment, misc]
    VALKEY_AVAILABLE = False

# Skip all tests if valkey dependencies are not available
pytestmark = pytest.mark.skipif(
    not VALKEY_AVAILABLE,
    reason=(
        "valkey dependencies not available. "
        "Install with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ),
)

# Import after optional dependency check
if VALKEY_AVAILABLE:
    import asyncio
    import os
    from datetime import datetime

    from langgraph.store.base import Item, SearchItem


def _is_valkey_server_available() -> bool:
    """Check if a Valkey server is available for testing."""
    if not VALKEY_AVAILABLE or Valkey is None:
        return False

    try:
        valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")
        client = Valkey.from_url(valkey_url)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


VALKEY_SERVER_AVAILABLE = _is_valkey_server_available()


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey server URL from environment or use default."""
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest.fixture
def async_store(valkey_url: str) -> AsyncValkeyStore:
    """Create an AsyncValkeyStore instance."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)
    return AsyncValkeyStore(client, ttl={"default_ttl": 60.0, "refresh_on_read": True})


@pytest_asyncio.fixture
async def async_store_with_index(
    valkey_url: str,
) -> AsyncGenerator[AsyncValkeyStore, None]:
    """Create an AsyncValkeyStore instance with vector indexing."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)

    # Create proper ValkeyIndexConfig
    index_config: ValkeyIndexConfig = {
        "dims": 4,
        "embed": lambda texts: [[1.0, 2.0, 3.0, 4.0] for _ in texts],
        "fields": ["text"],
        "collection_name": "test_index",
    }

    store = AsyncValkeyStore(client, index=index_config)
    await store.setup()
    yield store


@pytest_asyncio.fixture
async def clean_async_store(
    async_store: AsyncValkeyStore,
) -> AsyncGenerator[AsyncValkeyStore, None]:
    """Provide a clean async store and cleanup after tests."""
    yield async_store
    # Clear all test data - use flushdb to clear the database
    # flushdb() is synchronous in valkey-py, not async
    try:
        async_store.client.flushdb()
    except Exception:
        # Ignore cleanup errors (e.g., if server is not running)
        pass


# AsyncValkeyStore tests
@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_put_and_get(clean_async_store: AsyncValkeyStore) -> None:
    """Test basic put and get operations with AsyncValkeyStore."""
    namespace = ("test", "docs")
    key = "doc1"
    value = {"text": "Hello world", "tags": ["test"]}

    # Store item
    await clean_async_store.aput(namespace, key, value)

    # Retrieve item
    item = await clean_async_store.aget(namespace, key)
    assert item is not None
    assert item.value == value
    assert item.key == key
    assert item.namespace == namespace
    assert isinstance(item.created_at, datetime)
    assert isinstance(item.updated_at, datetime)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_put_and_get_with_ttl(clean_async_store: AsyncValkeyStore) -> None:
    """Test TTL functionality with AsyncValkeyStore."""
    namespace = ("test", "cache")
    key = "temp1"
    value = {"data": "temporary"}
    ttl = 0.1  # 6 seconds

    # Store item with TTL
    await clean_async_store.aput(namespace, key, value, ttl=ttl)

    # Verify item exists
    item = await clean_async_store.aget(namespace, key)
    assert item is not None
    assert item.value == value

    # Wait for TTL to expire
    await asyncio.sleep(ttl * 60 + 1)

    # Verify item is gone
    item = await clean_async_store.aget(namespace, key)
    assert item is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search(async_store_with_index: AsyncValkeyStore) -> None:
    """Test search functionality with AsyncValkeyStore."""
    # Add test data
    docs = [
        ("test", "search", "doc1"),
        ("test", "search", "doc2"),
        ("test", "search", "doc3"),
    ]

    for ns_parts in docs:
        namespace = ns_parts[:-1]
        key = ns_parts[-1]
        await async_store_with_index.aput(
            namespace, key, {"text": f"Document {key}", "type": "test"}
        )

    # Test basic prefix search
    results = await async_store_with_index.asearch(
        ("test", "search"), filter={"type": "test"}
    )
    assert len(results) == 3
    assert all(isinstance(r, SearchItem) for r in results)

    # Test vector similarity search
    results = await async_store_with_index.asearch(
        ("test", "search"), query="test search"
    )
    assert len(results) > 0
    assert all(hasattr(r, "score") for r in results)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_delete(clean_async_store: AsyncValkeyStore) -> None:
    """Test delete operation with AsyncValkeyStore."""
    namespace = ("test", "delete")
    key = "doc1"
    value = {"text": "Delete me"}

    # Store and verify item exists
    await clean_async_store.aput(namespace, key, value)
    item = await clean_async_store.aget(namespace, key)
    assert item is not None

    # Delete item
    await clean_async_store.adelete(namespace, key)

    # Verify item is gone
    item = await clean_async_store.aget(namespace, key)
    assert item is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_list_namespaces(clean_async_store: AsyncValkeyStore) -> None:
    """Test namespace listing with AsyncValkeyStore."""
    # Create test data in different namespaces
    test_data = [
        (("test", "a", "1"), "doc1", {"text": "a1"}),
        (("test", "a", "2"), "doc2", {"text": "a2"}),
        (("test", "b", "1"), "doc3", {"text": "b1"}),
    ]

    for namespace, key, value in test_data:
        await clean_async_store.aput(namespace, key, value)

    # List all namespaces under test
    namespaces = await clean_async_store.alist_namespaces(prefix=("test",))
    assert len(namespaces) > 0
    assert all(ns[0] == "test" for ns in namespaces)

    # Test max depth
    namespaces = await clean_async_store.alist_namespaces(prefix=("test",), max_depth=2)
    assert all(len(ns) <= 2 for ns in namespaces)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_batch_operations(clean_async_store: AsyncValkeyStore) -> None:
    """Test batch operations with AsyncValkeyStore."""
    # Create multiple items
    namespace = ("test", "batch")
    items = [(namespace, f"doc{i}", {"text": f"Document {i}"}) for i in range(3)]

    # Store items
    for ns, key, value in items:
        await clean_async_store.aput(ns, key, value)

    # Retrieve items in batch
    results = await asyncio.gather(
        *[clean_async_store.aget(ns, key) for ns, key, _ in items]
    )

    assert len(results) == 3
    assert all(isinstance(r, Item) for r in results)
    assert all(r is not None for r in results)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_error_handling(clean_async_store: AsyncValkeyStore) -> None:
    """Test error handling with AsyncValkeyStore."""
    # Test invalid namespace
    with pytest.raises(ValkeyValidationError):
        await clean_async_store.aput((), "key", {"test": "value"})

    # Test invalid value type - this should raise a ValkeyValidationError
    # but we need to use Any type to bypass static type checking
    from typing import Any

    invalid_value: Any = "not a dict"
    with pytest.raises(ValkeyValidationError):
        await clean_async_store.aput(("test",), "key", invalid_value)
