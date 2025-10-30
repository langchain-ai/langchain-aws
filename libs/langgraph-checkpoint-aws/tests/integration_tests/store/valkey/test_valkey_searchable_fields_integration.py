"""Integration tests for ValkeyStore searchable fields functionality."""

from collections.abc import Generator

import pytest

from langgraph_checkpoint_aws import ValkeyIndexConfig, ValkeyStore

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
        vector = [float(ord(c)) for c in text[:4].ljust(4, "a")]
        embeddings.append(vector)
    return embeddings


class MockEmbeddings:
    """Mock embeddings class for testing."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Make the class callable for direct invocation."""
        return mock_embedding_sync(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return mock_embedding_sync(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return mock_embedding_sync([text])[0]


@pytest.fixture
def store_with_searchable_fields(valkey_url: str) -> Generator[ValkeyStore, None, None]:
    """Create a ValkeyStore with comprehensive searchable fields configuration."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")

    client = Valkey.from_url(valkey_url)

    # Configuration similar to the notebook example
    index_config: ValkeyIndexConfig = {
        "collection_name": "enterprise_memory_vectors",
        "dims": 4,
        "embed": MockEmbeddings(),  # type: ignore
        "fields": [
            "user_id",
            "memory_type",
            "importance",
            "created_at",
            "updated_at",
            "content",
            "tags",
            "version",
        ],
    }

    store = ValkeyStore(client, index=index_config, ttl={"default_ttl": 60.0})
    store.setup()
    yield store

    # Cleanup
    try:
        store.client.flushdb()
    except Exception:
        pass


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_searchable_fields_are_indexed(
    store_with_searchable_fields: ValkeyStore,
) -> None:
    """Test that searchable fields are properly indexed
    and can be used for filtering."""

    # Add test documents with various searchable fields
    test_docs = [
        {
            "namespace": ("enterprise_memories",),
            "key": "mem_001",
            "value": {
                "user_id": "enterprise_user_001",
                "memory_type": "fact",
                "importance": "0.9",
                "content": (
                    "Alice Johnson is a Senior Software Engineer specializing in "
                    "machine learning."
                ),
                "tags": ["professional", "role", "expertise"],
                "version": "1",
                "created_at": "2024-01-15T10:00:00",
                "updated_at": "2024-01-15T10:00:00",
            },
        },
        {
            "namespace": ("enterprise_memories",),
            "key": "mem_002",
            "value": {
                "user_id": "enterprise_user_001",
                "memory_type": "preference",
                "importance": "0.8",
                "content": (
                    "Alice prefers Python for data analysis and has experience "
                    "with TensorFlow."
                ),
                "tags": ["programming", "tools", "preference"],
                "version": "1",
                "created_at": "2024-01-15T11:00:00",
                "updated_at": "2024-01-15T11:00:00",
            },
        },
        {
            "namespace": ("enterprise_memories",),
            "key": "mem_003",
            "value": {
                "user_id": "enterprise_user_002",
                "memory_type": "fact",
                "importance": "0.9",
                "content": (
                    "Bob Smith is a DevOps Engineer with expertise in Kubernetes "
                    "and AWS."
                ),
                "tags": ["professional", "devops", "cloud"],
                "version": "1",
                "created_at": "2024-01-15T12:00:00",
                "updated_at": "2024-01-15T12:00:00",
            },
        },
    ]

    # Store all documents
    for doc in test_docs:
        store_with_searchable_fields.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Test 1: Search with user_id filter
    results = store_with_searchable_fields.search(
        namespace_prefix=("enterprise_memories",),
        query="machine learning",
        filter={"user_id": "enterprise_user_001"},
        limit=10,
    )

    # Should return only documents for enterprise_user_001
    assert len(results) >= 1
    for result in results:
        assert result.value.get("user_id") == "enterprise_user_001"

    # Test 2: Search with memory_type filter
    results = store_with_searchable_fields.search(
        namespace_prefix=("enterprise_memories",),
        query="expertise",
        filter={"memory_type": "fact"},
        limit=10,
    )

    # Should return only fact-type memories
    assert len(results) >= 1
    for result in results:
        assert result.value.get("memory_type") == "fact"

    # Test 3: Search without filters (should return all relevant results)
    results = store_with_searchable_fields.search(
        namespace_prefix=("enterprise_memories",), query="engineer", limit=10
    )

    # Should return results for both Alice and Bob
    assert len(results) >= 2


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_fields_searchable(store_with_searchable_fields: ValkeyStore) -> None:
    """Test that list fields (like tags) are properly searchable."""

    # Add documents with different tag combinations
    test_docs = [
        {
            "namespace": ("test_tags",),
            "key": "doc1",
            "value": {
                "user_id": "user1",
                "content": "Document about machine learning",
                "tags": ["machine-learning", "ai", "python"],
                "memory_type": "technical",
            },
        },
        {
            "namespace": ("test_tags",),
            "key": "doc2",
            "value": {
                "user_id": "user1",
                "content": "Document about web development",
                "tags": ["web", "javascript", "react"],
                "memory_type": "technical",
            },
        },
        {
            "namespace": ("test_tags",),
            "key": "doc3",
            "value": {
                "user_id": "user2",
                "content": "Personal note about cooking",
                "tags": ["cooking", "recipe", "personal"],
                "memory_type": "personal",
            },
        },
    ]

    # Store documents
    for doc in test_docs:
        store_with_searchable_fields.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Test filtering by tags (note: exact tag matching depends on
    # Valkey Search implementation)
    # For now, we test that documents with tags are stored and retrievable
    results = store_with_searchable_fields.search(
        namespace_prefix=("test_tags",), query="machine learning", limit=10
    )

    # Should find the machine learning document
    assert len(results) >= 1
    ml_doc = next(
        (r for r in results if "machine learning" in r.value.get("content", "")), None
    )
    assert ml_doc is not None
    assert "machine-learning" in ml_doc.value.get("tags", [])


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_configured_collection_name_used(
    store_with_searchable_fields: ValkeyStore,
) -> None:
    """Test that the configured collection name is actually used."""

    # Verify the store is using the configured collection name
    assert store_with_searchable_fields.collection_name == "enterprise_memory_vectors"

    # Add a test document
    store_with_searchable_fields.put(
        ("test_collection",),
        "test_doc",
        {
            "user_id": "test_user",
            "content": "Test document for collection name verification",
            "memory_type": "test",
        },
    )

    # Verify the document was stored correctly
    stored_item = store_with_searchable_fields.get(("test_collection",), "test_doc")
    assert stored_item is not None, "Document was not stored properly"
    assert "Test document for collection name verification" in stored_item.value.get(
        "content", ""
    )

    # Search should work (implicitly testing that the correct index is being used)
    results = store_with_searchable_fields.search(
        namespace_prefix=("test_collection",), query="test document", limit=10
    )

    # Should find the test document
    assert len(results) >= 1, "No search results found. Expected at least 1 result."

    # Check if any result contains the expected content (case-insensitive)
    matching_results = [
        r for r in results if "test document" in r.value.get("content", "").lower()
    ]
    assert len(matching_results) >= 1, (
        f"No results contain 'test document' in content. "
        f"Results: {[r.value for r in results]}"
    )


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_multiple_field_filters(store_with_searchable_fields: ValkeyStore) -> None:
    """Test filtering by multiple fields simultaneously."""

    # Add test documents with various combinations
    test_docs = [
        {
            "namespace": ("multi_filter",),
            "key": "doc1",
            "value": {
                "user_id": "alice",
                "memory_type": "fact",
                "importance": "high",
                "content": "Important fact about Alice",
                "version": "1",
            },
        },
        {
            "namespace": ("multi_filter",),
            "key": "doc2",
            "value": {
                "user_id": "alice",
                "memory_type": "preference",
                "importance": "medium",
                "content": "Alice's preference",
                "version": "1",
            },
        },
        {
            "namespace": ("multi_filter",),
            "key": "doc3",
            "value": {
                "user_id": "bob",
                "memory_type": "fact",
                "importance": "high",
                "content": "Important fact about Bob",
                "version": "1",
            },
        },
    ]

    # Store documents
    for doc in test_docs:
        store_with_searchable_fields.put(  # type: ignore[arg-type]
            doc["namespace"],  # type: ignore[arg-type]
            doc["key"],  # type: ignore[arg-type]
            doc["value"],  # type: ignore[arg-type]
        )

    # Test filtering by user_id AND memory_type
    results = store_with_searchable_fields.search(
        namespace_prefix=("multi_filter",),
        query="fact",
        filter={"user_id": "alice", "memory_type": "fact"},
        limit=10,
    )

    # Should return only Alice's facts
    assert len(results) >= 1
    for result in results:
        assert result.value.get("user_id") == "alice"
        assert result.value.get("memory_type") == "fact"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_search_performance_with_fields(
    store_with_searchable_fields: ValkeyStore,
) -> None:
    """Test that search performance is reasonable with many indexed fields."""

    # Add multiple documents to test performance
    import time

    # Store multiple documents
    for i in range(20):
        store_with_searchable_fields.put(
            ("performance_test",),
            f"doc_{i}",
            {
                "user_id": f"user_{i % 5}",  # 5 different users
                "memory_type": "fact" if i % 2 == 0 else "preference",
                "importance": str(0.5 + (i % 5) * 0.1),
                "content": (f"This is test document number {i} with various content"),
                "tags": [f"tag_{i % 3}", f"category_{i % 4}"],
                "version": "1",
                "created_at": f"2024-01-{15 + (i % 10):02d}T10:00:00",
            },
        )

    # Measure search performance
    start_time = time.time()

    results = store_with_searchable_fields.search(
        namespace_prefix=("performance_test",),
        query="test document",
        filter={"memory_type": "fact"},
        limit=10,
    )

    end_time = time.time()
    search_duration = end_time - start_time

    # Should complete search reasonably quickly (under 1 second for this small dataset)
    assert search_duration < 1.0, f"Search took too long: {search_duration:.2f} seconds"

    # Should return some results
    assert len(results) > 0

    # All results should match the filter
    for result in results:
        assert result.value.get("memory_type") == "fact"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_field_types_handling(store_with_searchable_fields: ValkeyStore) -> None:
    """Test that different field types are handled correctly."""

    # Test document with various field types
    test_doc = {
        "user_id": "test_user",
        "memory_type": "mixed_types",
        "importance": 0.85,  # float
        "content": "Test content with various types",
        "tags": ["string", "list", "values"],  # list
        "version": 42,  # int
        "created_at": "2024-01-15T10:00:00",  # string (datetime)
        "metadata": {"nested": "object"},  # dict (not indexed)
    }

    # Store the document
    store_with_searchable_fields.put(("type_test",), "mixed_doc", test_doc)

    # Search should work
    results = store_with_searchable_fields.search(
        namespace_prefix=("type_test",), query="test content", limit=10
    )

    # Should find the document
    assert len(results) >= 1
    found_doc = results[0]

    # Verify all field types are preserved in the stored document
    assert found_doc.value["user_id"] == "test_user"
    assert found_doc.value["importance"] == 0.85
    assert found_doc.value["tags"] == ["string", "list", "values"]
    assert found_doc.value["version"] == 42
    assert found_doc.value["metadata"] == {"nested": "object"}
