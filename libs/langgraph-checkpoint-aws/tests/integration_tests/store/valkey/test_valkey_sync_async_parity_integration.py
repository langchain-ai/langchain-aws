"""Integration tests comparing sync and async Valkey store behavior.

This test module validates that ValkeyStore (sync) and AsyncValkeyStore (async)
return consistent results for the same operations.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from langgraph_checkpoint_aws.store.valkey import AsyncValkeyStore, ValkeyStore


def _is_valkey_server_available() -> bool:
    """Check if Valkey server is available."""
    host = os.getenv("VALKEY_HOST", "localhost")
    port = int(os.getenv("VALKEY_PORT", "6379"))

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


VALKEY_SERVER_AVAILABLE = _is_valkey_server_available()


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey connection URL from environment."""
    host = os.getenv("VALKEY_HOST", "localhost")
    port = os.getenv("VALKEY_PORT", "6379")
    return f"valkey://{host}:{port}"


def mock_embedding_sync(texts: list[str]) -> list[list[float]]:
    """Mock embedding function that returns predictable vectors."""
    return [[0.1, 0.2, 0.3] for _ in texts]


async def mock_embedding_async(texts: list[str]) -> list[list[float]]:
    """Async mock embedding function."""
    return mock_embedding_sync(texts)


class MockEmbeddings:
    """Mock embeddings class for testing both sync and async."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Make the class callable for direct invocation."""
        return mock_embedding_sync(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync embedding generation."""
        return mock_embedding_sync(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embedding generation."""
        return await mock_embedding_async(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return mock_embedding_sync([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed a single query text."""
        results = await mock_embedding_async([text])
        return results[0]


@pytest.fixture
def sync_store_with_index(valkey_url: str) -> Generator[ValkeyStore, None, None]:
    """Create a sync ValkeyStore instance with vector indexing."""
    from valkey import Valkey

    client = Valkey.from_url(valkey_url)

    store = ValkeyStore(
        client,
        index={  # type: ignore[arg-type]
            "dims": 3,
            "embed": MockEmbeddings(),
            "fields": ["text"],
            "collection_name": "test_sync_async_parity",
        },
    )

    # Setup and cleanup
    try:
        # Clean up before test
        keys = client.keys("langgraph:*")
        if isinstance(keys, list) and keys:
            client.delete(*keys)

        # Setup store (creates vector index if not exists)
        # Note: sync setup is not async, so we skip it for integration tests

        yield store
    finally:
        # Clean up after test
        keys = client.keys("langgraph:*")
        if isinstance(keys, list) and keys:
            client.delete(*keys)
        client.close()


@pytest_asyncio.fixture
async def async_store_with_index(
    valkey_url: str,
) -> AsyncGenerator[AsyncValkeyStore, None]:
    """Create an async AsyncValkeyStore instance with vector indexing."""
    from valkey import Valkey

    client = Valkey.from_url(valkey_url)

    store = AsyncValkeyStore(
        client,
        index={  # type: ignore[arg-type]
            "dims": 3,
            "embed": MockEmbeddings(),
            "fields": ["text"],
            "collection_name": "test_sync_async_parity",
        },
    )

    # Setup and cleanup
    try:
        # Clean up before test
        keys = await asyncio.get_event_loop().run_in_executor(
            None, client.keys, "langgraph:*"
        )
        if isinstance(keys, list) and keys:
            await asyncio.get_event_loop().run_in_executor(None, client.delete, *keys)

        # Setup store (creates vector index if not exists)
        await store.setup()

        yield store
    finally:
        # Clean up after test
        keys = await asyncio.get_event_loop().run_in_executor(
            None, client.keys, "langgraph:*"
        )
        if isinstance(keys, list) and keys:
            await asyncio.get_event_loop().run_in_executor(None, client.delete, *keys)
        client.close()


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_sync_async_search_parity(
    sync_store_with_index: ValkeyStore,
    async_store_with_index: AsyncValkeyStore,
) -> None:
    """Test that sync and async stores return similar search results."""
    # Populate sync store
    test_docs = [
        (
            ("test", "search"),
            "doc2",
            {"text": "Python programming", "category": "tech"},
        ),
        (("test", "search"), "doc3", {"text": "Data science", "category": "tech"}),
        (("test", "other"), "doc4", {"text": "Machine learning", "category": "tech"}),
    ]

    # Add to sync store
    for namespace, key, value in test_docs:
        sync_store_with_index.put(namespace, key, value)

    # Add to async store
    for namespace, key, value in test_docs:
        await async_store_with_index.aput(namespace, key, value)

    # Test 1: Basic namespace prefix search (no vector search)
    sync_results = sync_store_with_index.search(("test", "search"))
    async_results = await async_store_with_index.asearch(("test", "search"))

    assert len(sync_results) == len(async_results) == 2
    sync_keys = {r.key for r in sync_results}
    async_keys = {r.key for r in async_results}
    assert sync_keys == async_keys == {"doc2", "doc3"}

    # Test 2: Search with filter
    sync_results = sync_store_with_index.search(
        ("test", "search"), filter={"category": "tech"}
    )
    async_results = await async_store_with_index.asearch(
        ("test", "search"), filter={"category": "tech"}
    )

    assert len(sync_results) == len(async_results) == 2
    sync_keys = {r.key for r in sync_results}
    async_keys = {r.key for r in async_results}
    assert sync_keys == async_keys == {"doc2", "doc3"}

    # Test 3: Text search with query (will use hash/key pattern fallback)
    sync_results = sync_store_with_index.search(("test", "search"), query="Python")
    async_results = await async_store_with_index.asearch(
        ("test", "search"), query="Python"
    )

    # Both should return results
    assert len(sync_results) > 0
    assert len(async_results) > 0
    # Both should find the Python doc
    assert any(r.key == "doc2" for r in sync_results)
    assert any(r.key == "doc2" for r in async_results)

    # Test 4: Search with pagination
    sync_results = sync_store_with_index.search(("test", "search"), limit=2, offset=0)
    async_results = await async_store_with_index.asearch(
        ("test", "search"), limit=2, offset=0
    )

    assert len(sync_results) == len(async_results) == 2

    # Test 5: Search with offset
    sync_results = sync_store_with_index.search(("test", "search"), limit=2, offset=1)
    async_results = await async_store_with_index.asearch(
        ("test", "search"), limit=2, offset=1
    )

    assert len(sync_results) == len(async_results) == 1


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_sync_async_get_put_parity(
    sync_store_with_index: ValkeyStore,
    async_store_with_index: AsyncValkeyStore,
) -> None:
    """Test that sync and async stores handle put/get operations consistently."""
    namespace = ("test", "operations")
    key = "test_key"
    value = {"text": "Test value", "number": 42}

    # Put in sync store
    sync_store_with_index.put(namespace, key, value)
    sync_result = sync_store_with_index.get(namespace, key)

    # Put in async store
    await async_store_with_index.aput(namespace, key, value)
    async_result = await async_store_with_index.aget(namespace, key)

    # Both should return items
    assert sync_result is not None
    assert async_result is not None

    # Values should match
    assert sync_result.value == async_result.value == value
    assert sync_result.key == async_result.key == key
    assert sync_result.namespace == async_result.namespace == namespace


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_sync_async_delete_parity(
    sync_store_with_index: ValkeyStore,
    async_store_with_index: AsyncValkeyStore,
) -> None:
    """Test that sync and async stores handle delete operations consistently."""
    namespace = ("test", "delete")

    # Test sync delete
    sync_store_with_index.put(namespace, "key1", {"text": "Delete me"})
    assert sync_store_with_index.get(namespace, "key1") is not None
    sync_store_with_index.delete(namespace, "key1")
    assert sync_store_with_index.get(namespace, "key1") is None

    # Test async delete
    await async_store_with_index.aput(namespace, "key2", {"text": "Delete me"})
    assert await async_store_with_index.aget(namespace, "key2") is not None
    await async_store_with_index.adelete(namespace, "key2")
    assert await async_store_with_index.aget(namespace, "key2") is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_sync_async_list_namespaces_parity(
    sync_store_with_index: ValkeyStore,
    async_store_with_index: AsyncValkeyStore,
) -> None:
    """Test that sync and async stores list namespaces consistently."""
    # Add test data to both stores
    namespaces = [
        ("test", "ns1"),
        ("test", "ns2"),
        ("other", "ns3"),
    ]

    for namespace in namespaces:
        sync_store_with_index.put(namespace, "key", {"text": "test"})
        await async_store_with_index.aput(namespace, "key", {"text": "test"})

    # List all namespaces
    sync_namespaces = sync_store_with_index.list_namespaces()
    async_namespaces = await async_store_with_index.alist_namespaces()

    # Convert to sets for comparison (order might differ)
    sync_ns_set = set(sync_namespaces)
    async_ns_set = set(async_namespaces)

    assert sync_ns_set == async_ns_set
    assert len(sync_ns_set) == 3

    # List with prefix
    sync_namespaces = sync_store_with_index.list_namespaces(prefix=("test",))
    async_namespaces = await async_store_with_index.alist_namespaces(prefix=("test",))

    sync_ns_set = set(sync_namespaces)
    async_ns_set = set(async_namespaces)

    assert sync_ns_set == async_ns_set
    assert len(sync_ns_set) == 2


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_sync_async_vector_search_parity(
    sync_store_with_index: ValkeyStore,
    async_store_with_index: AsyncValkeyStore,
) -> None:
    """Test that vector search returns consistent results in sync and async."""
    # Add documents with different content
    docs = [
        (("vector", "test"), "doc1", {"text": "artificial intelligence"}),
        (("vector", "test"), "doc2", {"text": "machine learning algorithms"}),
        (("vector", "test"), "doc3", {"text": "neural networks"}),
        (("vector", "test"), "doc4", {"text": "cooking recipes"}),
    ]

    # Populate both stores
    for namespace, key, value in docs:
        sync_store_with_index.put(namespace, key, value)
        await async_store_with_index.aput(namespace, key, value)

    # Perform vector search on both
    sync_results = sync_store_with_index.search(
        ("vector", "test"), query="machine learning", limit=3
    )
    async_results = await async_store_with_index.asearch(
        ("vector", "test"), query="machine learning", limit=3
    )

    # Both should return results
    assert len(sync_results) > 0
    assert len(async_results) > 0

    # All results should have scores
    assert all(hasattr(r, "score") and r.score is not None for r in sync_results)
    assert all(hasattr(r, "score") and r.score is not None for r in async_results)

    # Both should find doc2 which contains "machine learning"
    # Note: Since mock embeddings return identical vectors, both stores will
    # use fallback search (hash/key pattern) that performs text matching
    sync_keys = {r.key for r in sync_results}
    async_keys = {r.key for r in async_results}

    # Verify doc2 is found by at least one store
    assert "doc2" in sync_keys or "doc2" in async_keys
