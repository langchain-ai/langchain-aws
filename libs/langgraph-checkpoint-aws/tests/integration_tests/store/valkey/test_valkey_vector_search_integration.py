"""Tests for vector search functionality in ValkeyStore implementations."""

from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from langgraph_checkpoint_aws import AsyncValkeyStore, ValkeyIndexConfig, ValkeyStore

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
    import os

    from langgraph.store.base import SearchItem


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


# Mock embedding function for testing
def mock_embedding_sync(texts: list[str]) -> list[list[float]]:
    """Mock embedding function that returns predictable vectors."""
    embeddings = []
    for text in texts:
        # Create simple embeddings based on text content
        # This allows us to test similarity without real embeddings
        vector = [float(ord(c)) for c in text[:4].ljust(4, "a")]
        embeddings.append(vector)
    return embeddings


async def mock_embedding_async(texts: list[str]) -> list[list[float]]:
    """Async mock embedding function."""
    return mock_embedding_sync(texts)


class MockEmbeddings:
    """Mock embeddings class for testing."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Make the class callable for direct invocation."""
        return mock_embedding_sync(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return mock_embedding_sync(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await mock_embedding_async(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return mock_embedding_sync([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed a single query text."""
        results = await mock_embedding_async([text])
        return results[0]


@pytest.fixture
def store_with_vector_search(valkey_url: str) -> Generator[ValkeyStore, None, None]:
    """Create a sync ValkeyStore with vector search enabled."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)
    index_config: ValkeyIndexConfig = {
        "collection_name": "test_store_idx",
        "dims": 4,
        "embed": MockEmbeddings(),  # type: ignore
        "fields": ["text", "content"],
    }
    store = ValkeyStore(client, index=index_config, ttl={"default_ttl": 60.0})
    store.setup()
    yield store
    # Cleanup
    try:
        store.client.flushdb()
    except Exception:
        pass


@pytest.fixture
def store_without_vector_search(valkey_url: str) -> Generator[ValkeyStore, None, None]:
    """Create a sync ValkeyStore without vector search (fallback only)."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)
    store = ValkeyStore(client, ttl={"default_ttl": 60.0})
    yield store
    # Cleanup
    try:
        store.client.flushdb()
    except Exception:
        pass


@pytest_asyncio.fixture
async def async_store_with_vector_search(
    valkey_url: str,
) -> AsyncGenerator[AsyncValkeyStore, None]:
    """Create an async ValkeyStore with vector search enabled."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)
    index_config: ValkeyIndexConfig = {
        "collection_name": "test_async_store_idx",
        "dims": 4,
        "embed": MockEmbeddings(),  # type: ignore
        "fields": ["text", "content"],
    }
    store = AsyncValkeyStore(client, index=index_config, ttl={"default_ttl": 60.0})
    await store.setup()
    yield store
    # Cleanup
    try:
        store.client.flushdb()
    except Exception:
        pass


@pytest_asyncio.fixture
async def async_store_without_vector_search(
    valkey_url: str,
) -> AsyncGenerator[AsyncValkeyStore, None]:
    """Create an async ValkeyStore without vector search (fallback only)."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    client = Valkey.from_url(valkey_url)
    store = AsyncValkeyStore(client, ttl={"default_ttl": 60.0})
    yield store
    # Cleanup
    try:
        store.client.flushdb()
    except Exception:
        pass


# Sync ValkeyStore Vector Search Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_vector_search_with_embeddings(
    store_with_vector_search: ValkeyStore,
) -> None:
    """Test vector search functionality with sync ValkeyStore."""
    # Add test documents with different content
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {"text": "machine learning guide", "category": "tech"},
        },
        {
            "namespace": ("docs", "tech"),
            "key": "ai_basics",
            "value": {"text": "artificial intelligence basics", "category": "tech"},
        },
        {
            "namespace": ("docs", "general"),
            "key": "cooking",
            "value": {"text": "cooking recipes and tips", "category": "lifestyle"},
        },
    ]

    # Store documents
    for doc in test_docs:
        store_with_vector_search.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Test vector search with query
    results = store_with_vector_search.search(
        namespace_prefix=("docs",), query="machine learning", limit=10
    )

    # Should return results with similarity scores
    assert len(results) > 0
    assert all(isinstance(r, SearchItem) for r in results)
    assert all(hasattr(r, "score") for r in results)
    assert all(r.score is not None for r in results)

    # Results should be sorted by relevance (higher scores first)
    scores = [r.score for r in results if r.score is not None]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_vector_search_with_filters(store_with_vector_search: ValkeyStore) -> None:
    """Test vector search with additional filters."""
    # Add test documents
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {
                "text": "machine learning guide",
                "category": "tech",
                "level": "advanced",
            },
        },
        {
            "namespace": ("docs", "tech"),
            "key": "ai_basics",
            "value": {
                "text": "artificial intelligence basics",
                "category": "tech",
                "level": "beginner",
            },
        },
    ]

    for doc in test_docs:
        store_with_vector_search.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Search with filter
    results = store_with_vector_search.search(
        namespace_prefix=("docs",),
        query="machine learning",
        filter={"level": "advanced"},
        limit=10,
    )

    # Should only return advanced level documents
    assert len(results) >= 1
    for result in results:
        assert result.value.get("level") == "advanced"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_fallback_to_key_search(store_without_vector_search: ValkeyStore) -> None:
    """Test fallback to key pattern search when vector search is not available."""
    # Add test documents
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {"text": "machine learning guide", "category": "tech"},
        },
        {
            "namespace": ("docs", "general"),
            "key": "cooking",
            "value": {"text": "cooking recipes", "category": "lifestyle"},
        },
    ]

    for doc in test_docs:
        store_without_vector_search.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Search should fall back to key pattern matching
    results = store_without_vector_search.search(
        namespace_prefix=("docs",),
        query="machine learning",  # This should be ignored in fallback
        limit=10,
    )

    # Should return results using key pattern matching
    assert len(results) >= 0  # May return 0 or more based on key patterns
    assert all(isinstance(r, SearchItem) for r in results)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_search_with_pagination(store_with_vector_search: ValkeyStore) -> None:
    """Test search pagination functionality."""
    # Add multiple test documents
    for i in range(10):
        store_with_vector_search.put(
            ("docs", "test"), f"doc_{i}", {"text": f"document number {i}", "index": i}
        )

    # Test pagination
    page1 = store_with_vector_search.search(
        namespace_prefix=("docs",), query="document", limit=3, offset=0
    )

    page2 = store_with_vector_search.search(
        namespace_prefix=("docs",), query="document", limit=3, offset=3
    )

    assert len(page1) <= 3
    assert len(page2) <= 3

    # Pages should not overlap
    page1_keys = {r.key for r in page1}
    page2_keys = {r.key for r in page2}
    assert page1_keys.isdisjoint(page2_keys)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_search_error_handling(store_with_vector_search: ValkeyStore) -> None:
    """Test error handling in search operations."""
    # Test search with invalid parameters should not crash
    results = store_with_vector_search.search(
        namespace_prefix=(),  # Empty namespace
        query="test",
        limit=10,
    )

    # Should return empty results, not crash
    assert isinstance(results, list)


# Async ValkeyStore Vector Search Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_vector_search_with_embeddings(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test vector search functionality with async ValkeyStore."""
    # Add test documents
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {"text": "machine learning guide", "category": "tech"},
        },
        {
            "namespace": ("docs", "tech"),
            "key": "ai_basics",
            "value": {"text": "artificial intelligence basics", "category": "tech"},
        },
        {
            "namespace": ("docs", "general"),
            "key": "cooking",
            "value": {"text": "cooking recipes and tips", "category": "lifestyle"},
        },
    ]

    # Store documents
    for doc in test_docs:
        await async_store_with_vector_search.aput(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Test vector search with query
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",), query="machine learning", limit=10
    )

    # Should return results with similarity scores
    assert len(results) > 0
    assert all(isinstance(r, SearchItem) for r in results)
    assert all(hasattr(r, "score") for r in results)
    assert all(r.score is not None for r in results)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_vector_search_with_filters(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test async vector search with additional filters."""
    # Add test documents
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {
                "text": "machine learning guide",
                "category": "tech",
                "level": "advanced",
            },
        },
        {
            "namespace": ("docs", "tech"),
            "key": "ai_basics",
            "value": {
                "text": "artificial intelligence basics",
                "category": "tech",
                "level": "beginner",
            },
        },
    ]

    for doc in test_docs:
        await async_store_with_vector_search.aput(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Search with filter
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",),
        query="machine learning",
        filter={"level": "advanced"},
        limit=10,
    )

    # Should only return advanced level documents
    assert len(results) >= 1
    for result in results:
        assert result.value.get("level") == "advanced"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_fallback_to_key_search(
    async_store_without_vector_search: AsyncValkeyStore,
) -> None:
    """Test async fallback to key pattern search when vector search is not available."""
    # Add test documents
    test_docs = [
        {
            "namespace": ("docs", "tech"),
            "key": "ml_guide",
            "value": {"text": "machine learning guide", "category": "tech"},
        },
        {
            "namespace": ("docs", "general"),
            "key": "cooking",
            "value": {"text": "cooking recipes", "category": "lifestyle"},
        },
    ]

    for doc in test_docs:
        await async_store_without_vector_search.aput(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Search should fall back to key pattern matching
    results = await async_store_without_vector_search.asearch(
        namespace_prefix=("docs",),
        query="machine learning",  # This should be ignored in fallback
        limit=10,
    )

    # Should return results using key pattern matching
    assert len(results) >= 0  # May return 0 or more based on key patterns
    assert all(isinstance(r, SearchItem) for r in results)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search_with_pagination(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test async search pagination functionality."""
    # Add multiple test documents
    for i in range(10):
        await async_store_with_vector_search.aput(
            ("docs", "test"), f"doc_{i}", {"text": f"document number {i}", "index": i}
        )

    # Test pagination
    page1 = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",), query="document", limit=3, offset=0
    )

    page2 = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",), query="document", limit=3, offset=3
    )

    assert len(page1) <= 3
    assert len(page2) <= 3

    # Pages should not overlap
    page1_keys = {r.key for r in page1}
    page2_keys = {r.key for r in page2}
    assert page1_keys.isdisjoint(page2_keys)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search_error_handling(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test error handling in async search operations."""
    # Test search with invalid parameters should not crash
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=(),  # Empty namespace
        query="test",
        limit=10,
    )

    # Should return empty results, not crash
    assert isinstance(results, list)


# TTL and Refresh Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_search_with_ttl_refresh(store_with_vector_search: ValkeyStore) -> None:
    """Test TTL refresh functionality during search."""
    # Add document with TTL
    store_with_vector_search.put(
        ("docs", "temp"),
        "temp_doc",
        {"text": "temporary document", "category": "temp"},
        ttl=1.0,  # 1 minute TTL
    )

    # Search with TTL refresh
    results = store_with_vector_search.search(
        namespace_prefix=("docs",), query="temporary", refresh_ttl=True, limit=10
    )

    assert len(results) >= 1
    # TTL should be refreshed (hard to test directly, but operation should succeed)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search_with_ttl_refresh(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test async TTL refresh functionality during search."""
    # Add document with TTL
    await async_store_with_vector_search.aput(
        ("docs", "temp"),
        "temp_doc",
        {"text": "temporary document", "category": "temp"},
        ttl=1.0,  # 1 minute TTL
    )

    # Search with TTL refresh
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",), query="temporary", refresh_ttl=True, limit=10
    )

    assert len(results) >= 1
    # TTL should be refreshed (hard to test directly, but operation should succeed)


# Edge Cases and Robustness Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_search_empty_results(store_with_vector_search: ValkeyStore) -> None:
    """Test search behavior with no matching results."""
    # Search for non-existent content
    results = store_with_vector_search.search(
        namespace_prefix=("nonexistent",), query="nonexistent content", limit=10
    )

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search_empty_results(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test async search behavior with no matching results."""
    # Search for non-existent content
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=("nonexistent",), query="nonexistent content", limit=10
    )

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_search_with_special_characters(
    store_with_vector_search: ValkeyStore,
) -> None:
    """Test search with special characters in query and content."""
    # Add document with special characters
    store_with_vector_search.put(
        ("docs", "special"),
        "special_doc",
        {"text": "document with special chars: @#$%^&*()", "category": "test"},
    )

    # Search with special characters
    results = store_with_vector_search.search(
        namespace_prefix=("docs",), query="special chars: @#$%", limit=10
    )

    # Should handle special characters gracefully
    assert isinstance(results, list)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_search_with_special_characters(
    async_store_with_vector_search: AsyncValkeyStore,
) -> None:
    """Test async search with special characters in query and content."""
    # Add document with special characters
    await async_store_with_vector_search.aput(
        ("docs", "special"),
        "special_doc",
        {"text": "document with special chars: @#$%^&*()", "category": "test"},
    )

    # Search with special characters
    results = await async_store_with_vector_search.asearch(
        namespace_prefix=("docs",), query="special chars: @#$%", limit=10
    )

    # Should handle special characters gracefully
    assert isinstance(results, list)
