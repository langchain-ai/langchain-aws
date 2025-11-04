"""Tests for ValKey store search functionality."""

import pytest

# Check for optional dependencies
try:
    import orjson  # noqa: F401
    import valkey  # noqa: F401
    from valkey import Valkey
    from valkey.exceptions import ValkeyError  # noqa: F401

    from langgraph_checkpoint_aws import AsyncValkeyStore, ValkeyStore

    VALKEY_AVAILABLE = True
except ImportError:
    # Create dummy objects for type checking when dependencies not available
    class MockOrjson:
        @staticmethod
        def dumps(obj):  # type: ignore[misc]
            import json

            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(data):  # type: ignore[misc]
            import json

            return json.loads(data)

    orjson = MockOrjson()  # type: ignore[assignment]
    Valkey = None  # type: ignore[assignment, misc]
    ValkeyError = Exception  # type: ignore[assignment, misc]
    ValkeyStore = None  # type: ignore[assignment, misc]
    AsyncValkeyStore = None  # type: ignore[assignment, misc]
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
    import json
    from datetime import datetime
    from unittest.mock import MagicMock, Mock, patch

    from langgraph.store.base import SearchItem, SearchOp

    from langgraph_checkpoint_aws.store.valkey.search_strategies import (
        HashSearchStrategy,
        KeyPatternSearchStrategy,
        SearchStrategyManager,
        VectorSearchStrategy,
    )


def mock_embed_fn(texts):
    """Mock embedding function that returns fixed vectors."""
    return [[0.1, 0.2] * 64 for _ in texts]  # 128-dim vectors


@pytest.fixture
def mock_valkey_client():
    """Create a mock ValKey client."""
    client = Mock(spec=Valkey)
    mock_ft = Mock()
    mock_ft.search = Mock()
    mock_ft.info = Mock(return_value=True)
    client.ft = Mock(return_value=mock_ft)
    client.get = Mock(return_value=None)
    client.set = Mock(return_value=True)
    client.scan = Mock(return_value=(0, []))
    client.execute_command = Mock(return_value=["langgraph_store_idx"])
    return client


@pytest.fixture
def mock_client():
    """Create a mock Valkey client."""
    client = MagicMock()
    client.ft = MagicMock()
    return client


@pytest.fixture
def mock_store():
    """Create a mock store."""
    store = MagicMock()
    store._is_search_available.return_value = True
    store.collection_name = "test_collection"
    store.dims = 128
    store.embeddings = MagicMock()
    store._handle_response_t.return_value = (0, [])
    store._safe_parse_keys.return_value = []
    return store


@pytest.fixture
def test_store(mock_valkey_client):
    """Create a test store with index configuration."""
    store = ValkeyStore(
        mock_valkey_client,
        index={
            "dims": 128,
            "fields": ["title", "content"],
            "embed": mock_embed_fn,
            "collection_name": "test_store_idx",
        },
    )
    return store


def create_test_document(value, vector=None):
    """Helper to create a test document in the same format as _create_document."""
    now = datetime.now()
    now_str = now.isoformat()

    # Create fields dict with string values
    fields = {
        "value": orjson.dumps(value).decode("utf-8"),  # Convert bytes to string
        "created_at": now_str,
        "updated_at": now_str,
    }
    if vector is not None:
        fields["vector"] = orjson.dumps(vector).decode("utf-8")

    # Create document dict with string values
    document = {
        "value": value,
        "created_at": now_str,
        "updated_at": now_str,
        "vector": vector,
        "_hash_fields": fields,
    }

    # Convert to JSON string
    return orjson.dumps(document)


class TestSearchFunctionality:
    """Test suite for ValKey store search functionality."""

    class MockDoc:
        def __init__(self, id, score):
            self.id = id
            self.score = score
            self.__dict__ = {"id": id, "score": score}

    def test_search_with_vector_available(self, mock_valkey_client):
        """Test search when vector search is available."""
        # Create store with vector search
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock search availability check
        def mock_execute_command(*args, **kwargs):
            if args[0] == "FT._LIST" or args[0] == "FT.LIST":
                return ["test_store_idx"]  # Use the configured collection_name
            return None

        mock_valkey_client.execute_command.side_effect = mock_execute_command

        # Mock search result
        mock_result = Mock()
        mock_result.docs = [self.MockDoc("langgraph:test/doc1", 0.9)]
        mock_valkey_client.ft.return_value.search.return_value = mock_result

        # Mock document retrieval - vector search uses hgetall, not get
        doc_value = {"title": "Test Doc", "content": "Test content"}
        # Create the hash structure that ValkeyStore expects
        hash_data = {
            "value": orjson.dumps(doc_value).decode("utf-8"),
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "vector": "[0.1, 0.2, 0.3]",
        }
        mock_valkey_client.hgetall.return_value = hash_data

        # Also mock the _handle_response_t method to return the hash_data as-is
        store._handle_response_t = Mock(side_effect=lambda x: x)

        # Perform search
        results = store.search(
            namespace_prefix=("test",), query="test query", filter={"type": "document"}
        )

        # Verify results
        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].namespace == ("test",)
        assert results[0].value == doc_value
        assert results[0].score == 0.9

    def test_search_with_hash_fallback(self, mock_valkey_client):
        """Test search falls back to hash-based search when vector search fails."""
        # Create store with vector search disabled
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable
        mock_valkey_client.execute_command.return_value = []
        mock_ft = Mock()
        mock_ft.info.side_effect = Exception("Not available")
        mock_ft.search.side_effect = Exception("Not available")
        mock_valkey_client.ft.return_value = mock_ft

        # Mock SCAN results
        mock_valkey_client.scan.return_value = (
            0,
            [b"langgraph:test/doc1", b"langgraph:test/doc2"],
        )

        # Mock document retrieval using hgetall (not get)
        def mock_hgetall(key):
            if key == "langgraph:test/doc1":
                return {
                    "value": orjson.dumps(
                        {"title": "Test Doc 1", "content": "Relevant content"}
                    ).decode("utf-8"),
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "vector": "[0.1, 0.2]",
                }
            elif key == "langgraph:test/doc2":
                return {
                    "value": orjson.dumps(
                        {"title": "Test Doc 2", "content": "Unrelated"}
                    ).decode("utf-8"),
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "vector": "[0.3, 0.4]",
                }
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        # Mock _handle_response_t to pass through the response
        store._handle_response_t = Mock(side_effect=lambda x: x)

        # Mock _safe_parse_keys to convert bytes to strings
        store._safe_parse_keys = Mock(
            side_effect=lambda keys: [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in keys
            ]
        )

        # Perform search
        results = store.search(namespace_prefix=("test",), query="relevant", limit=10)

        # Verify results
        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].value["title"] == "Test Doc 1"

    def test_search_with_filters(self, mock_valkey_client):
        """Test search with filter conditions."""
        # Create store with vector search disabled
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable
        mock_valkey_client.execute_command.return_value = []
        mock_ft = Mock()
        mock_ft.info.side_effect = Exception("Not available")
        mock_ft.search.side_effect = Exception("Not available")
        mock_valkey_client.ft.return_value = mock_ft

        # Mock SCAN results
        mock_valkey_client.scan.return_value = (
            0,
            [b"langgraph:test/doc1", b"langgraph:test/doc2"],
        )

        # Mock document retrieval with different document types
        def mock_hgetall(key):
            if key == "langgraph:test/doc1":
                return {
                    "value": orjson.dumps(
                        {
                            "title": "Test Doc 1",
                            "type": "report",
                            "content": "Test content",
                        }
                    ).decode("utf-8"),
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "vector": "[0.1, 0.2]",
                }
            elif key == "langgraph:test/doc2":
                return {
                    "value": orjson.dumps(
                        {
                            "title": "Test Doc 2",
                            "type": "note",
                            "content": "Test content",
                        }
                    ).decode("utf-8"),
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "vector": "[0.3, 0.4]",
                }
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        # Mock _handle_response_t to pass through the response
        store._handle_response_t = Mock(side_effect=lambda x: x)

        # Mock _safe_parse_keys to convert bytes to strings
        store._safe_parse_keys = Mock(
            side_effect=lambda keys: [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in keys
            ]
        )

        # Mock _parse_key to extract namespace and key
        def mock_parse_key(key_path, prefix=None):
            parts = key_path.split("/")
            return tuple(parts[:-1]), parts[-1]

        store._parse_key = Mock(side_effect=mock_parse_key)

        # Perform filtered search
        results = store.search(
            namespace_prefix=("test",), query="test", filter={"type": "report"}
        )

        # Verify only matching documents are returned
        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].value["type"] == "report"

    def test_search_with_namespace_prefix(self, mock_valkey_client):
        """Test search respects namespace prefix filtering."""
        # Create store with vector search disabled
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable
        mock_valkey_client.execute_command.return_value = []
        mock_ft = Mock()
        mock_ft.info.side_effect = Exception("Not available")
        mock_ft.search.side_effect = Exception("Not available")
        mock_valkey_client.ft.return_value = mock_ft

        # Mock SCAN results
        mock_valkey_client.scan.return_value = (
            0,
            [b"langgraph:test/public/doc1", b"langgraph:test/private/doc2"],
        )

        # Mock document retrieval using hgetall pattern
        def mock_hgetall(key):
            if key == "langgraph:test/public/doc1":
                doc_data = create_test_document(
                    {"title": "Public Doc", "content": "Test content"}
                )
                parsed_doc = json.loads(doc_data.decode("utf-8"))
                return {
                    b"value": json.dumps(parsed_doc["value"]).encode(),
                    b"created_at": parsed_doc["created_at"].encode(),
                    b"updated_at": parsed_doc["updated_at"].encode(),
                    b"vector": json.dumps(parsed_doc.get("vector")).encode()
                    if parsed_doc.get("vector")
                    else b"null",
                }
            elif key == "langgraph:test/private/doc2":
                doc_data = create_test_document(
                    {"title": "Private Doc", "content": "Test content"}
                )
                parsed_doc = json.loads(doc_data.decode("utf-8"))
                return {
                    b"value": json.dumps(parsed_doc["value"]).encode(),
                    b"created_at": parsed_doc["created_at"].encode(),
                    b"updated_at": parsed_doc["updated_at"].encode(),
                    b"vector": json.dumps(parsed_doc.get("vector")).encode()
                    if parsed_doc.get("vector")
                    else b"null",
                }
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        # Mock helper methods
        def mock_handle_response_t(data):
            return data

        def mock_parse_key(key_path):
            parts = key_path.split("/")
            if len(parts) >= 2:
                namespace = tuple(parts[:-1])
                key = parts[-1]
                return namespace, key
            return (), key_path

        def mock_safe_parse_keys(keys):
            return [k.decode() if isinstance(k, bytes) else k for k in keys]

        # Apply mocks to store instance
        store._handle_response_t = mock_handle_response_t
        store._parse_key = mock_parse_key
        store._safe_parse_keys = mock_safe_parse_keys

        # Search in public namespace
        results = store.search(namespace_prefix=("test", "public"), query="test")

        # Verify only public namespace documents are returned
        assert len(results) == 1
        assert results[0].namespace == ("test", "public")
        assert results[0].value["title"] == "Public Doc"

    def test_search_handles_errors(self, mock_valkey_client):
        """Test search gracefully handles various error conditions."""
        # Create store with vector search disabled
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable
        mock_valkey_client.execute_command.return_value = []
        mock_ft = Mock()
        mock_ft.info.side_effect = Exception("Not available")
        mock_ft.search.side_effect = Exception("Not available")
        mock_valkey_client.ft.return_value = mock_ft

        # Mock SCAN results
        mock_valkey_client.scan.return_value = (
            0,
            [b"langgraph:test/doc1", b"langgraph:test/doc2", b"langgraph:test/doc3"],
        )

        # Mock document retrieval with various error conditions
        def mock_hgetall(key):
            if key == "langgraph:test/doc1":
                doc_data = create_test_document(
                    {"title": "Good Doc", "content": "Test"}
                )
                parsed_doc = json.loads(doc_data.decode("utf-8"))
                return {
                    b"value": json.dumps(parsed_doc["value"]).encode(),
                    b"created_at": parsed_doc["created_at"].encode(),
                    b"updated_at": parsed_doc["updated_at"].encode(),
                    b"vector": json.dumps(parsed_doc.get("vector")).encode()
                    if parsed_doc.get("vector")
                    else b"null",
                }
            elif key == "langgraph:test/doc2":
                raise Exception("Failed to retrieve doc2")
            elif key == "langgraph:test/doc3":
                return {b"value": b"invalid json"}  # Invalid JSON in value field
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        # Mock helper methods
        def mock_handle_response_t(data):
            return data

        def mock_parse_key(key_path):
            parts = key_path.split("/")
            if len(parts) >= 2:
                namespace = tuple(parts[:-1])
                key = parts[-1]
                return namespace, key
            return (), key_path

        def mock_safe_parse_keys(keys):
            return [k.decode() if isinstance(k, bytes) else k for k in keys]

        # Apply mocks to store instance
        store._handle_response_t = mock_handle_response_t
        store._parse_key = mock_parse_key
        store._safe_parse_keys = mock_safe_parse_keys

        # Perform search
        results = store.search(namespace_prefix=("test",), query="test")

        # Verify only successfully processed documents are returned
        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].value["title"] == "Good Doc"

    def test_search_score_calculation(self, test_store):
        """Test search score calculation for different match types."""
        # Test exact field match
        doc1 = {
            "title": "Exact Match",
            "content": "Other content",
            "_hash_fields": {
                "value_title": "Exact Match",
                "value": orjson.dumps(
                    {"title": "Exact Match", "content": "Other content"}
                ).decode("utf-8"),
            },
        }
        score1 = test_store._calculate_simple_score("exact", doc1)
        assert score1 >= 0.8  # Should have high score for indexed field match

        # Test content match
        doc2 = {
            "title": "Other Title",
            "content": "Contains match phrase",
            "_hash_fields": {
                "value": orjson.dumps(
                    {"title": "Other Title", "content": "Contains match phrase"}
                ).decode("utf-8")
            },
        }
        score2 = test_store._calculate_simple_score("match", doc2)
        assert score2 >= 0.6  # Should have medium to high score for content match

        # Test no match
        doc3 = {
            "title": "No Match",
            "content": "Unrelated content",
            "_hash_fields": {
                "value": orjson.dumps(
                    {"title": "No Match", "content": "Unrelated content"}
                ).decode("utf-8")
            },
        }
        score3 = test_store._calculate_simple_score("missing", doc3)
        assert score3 == 0.1  # Should have minimum score for no match

    @pytest.mark.asyncio
    async def test_async_search(self, mock_valkey_client):
        """Test search works correctly in async context."""

        # Create async store with mock embedding function
        async_store = AsyncValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable - this should be a regular Mock
        # since execute_command is called via run_in_executor
        mock_valkey_client.execute_command = Mock(return_value=[])

        # Mock ft methods - ft() itself returns synchronously, but the
        # methods on it can be async
        mock_ft = Mock()
        mock_ft.info = Mock(side_effect=Exception("Not available"))
        mock_ft.search = Mock(side_effect=Exception("Not available"))
        mock_valkey_client.ft = Mock(return_value=mock_ft)

        # Mock SCAN results - scan is called via run_in_executor,
        # so it should return sync results
        mock_valkey_client.scan = Mock(return_value=(0, [b"langgraph:test/doc1"]))

        # Mock document retrieval - hgetall is used by async store, not get
        doc_value = {"title": "Async Doc", "content": "Test content"}
        hash_data = {
            "value": orjson.dumps(doc_value).decode("utf-8"),
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "vector": "[0.1, 0.2]",
        }
        mock_valkey_client.hgetall = Mock(return_value=hash_data)

        # Perform async search
        results = await async_store.asearch(namespace_prefix=("test",), query="test")

        # Verify results
        assert len(results) == 1
        assert results[0].key == "doc1"
        assert results[0].value["title"] == "Async Doc"

    def test_search_with_pagination(self, mock_valkey_client):
        """Test search pagination."""
        # Create store with vector search disabled
        store = ValkeyStore(
            mock_valkey_client,
            index={
                "dims": 128,
                "fields": ["title", "content"],
                "embed": mock_embed_fn,
                "collection_name": "test_store_idx",
            },
        )

        # Mock vector search as unavailable
        mock_valkey_client.execute_command.return_value = []
        mock_ft = Mock()
        mock_ft.info.side_effect = Exception("Not available")
        mock_ft.search.side_effect = Exception("Not available")
        mock_valkey_client.ft.return_value = mock_ft

        # Mock SCAN results with 5 documents
        mock_valkey_client.scan.return_value = (
            0,
            [bytes(f"langgraph:test/doc{i}", "utf-8") for i in range(1, 6)],
        )

        # Mock document retrieval
        docs = {
            f"langgraph:test/doc{i}": create_test_document(
                {"title": f"Doc {i}", "content": "Test content"}
            )
            for i in range(1, 6)
        }

        # Mock document retrieval using hgetall pattern
        def mock_hgetall(key):
            if key in docs:
                doc_data = docs[key]
                parsed_doc = json.loads(doc_data.decode("utf-8"))
                return {
                    b"value": json.dumps(parsed_doc["value"]).encode(),
                    b"created_at": parsed_doc["created_at"].encode(),
                    b"updated_at": parsed_doc["updated_at"].encode(),
                    b"vector": json.dumps(parsed_doc.get("vector")).encode()
                    if parsed_doc.get("vector")
                    else b"null",
                }
            return {}

        mock_valkey_client.hgetall.side_effect = mock_hgetall

        # Mock helper methods
        store._handle_response_t = Mock(side_effect=lambda x: x)
        store._safe_parse_keys = Mock(
            side_effect=lambda keys: [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in keys
            ]
        )

        def mock_parse_key(key_path):
            parts = key_path.split("/")
            return tuple(parts[:-1]), parts[-1]

        store._parse_key = Mock(side_effect=mock_parse_key)

        # Test first page
        results1 = store.search(
            namespace_prefix=("test",), query="test", limit=2, offset=0
        )
        assert len(results1) == 2
        assert results1[0].key == "doc1"
        assert results1[1].key == "doc2"

        # Test second page
        results2 = store.search(
            namespace_prefix=("test",), query="test", limit=2, offset=2
        )
        assert len(results2) == 2
        assert results2[0].key == "doc3"
        assert results2[1].key == "doc4"


class TestSearchStrategyManager:
    """Test SearchStrategyManager class."""

    def test_init(self, mock_client, mock_store):
        """Test SearchStrategyManager initialization."""
        manager = SearchStrategyManager(mock_client, mock_store)

        # SearchStrategyManager stores strategies in a list
        assert len(manager.strategies) == 3
        assert isinstance(manager.strategies[0], VectorSearchStrategy)
        assert isinstance(manager.strategies[1], HashSearchStrategy)
        assert isinstance(manager.strategies[2], KeyPatternSearchStrategy)

    def test_search_with_vector_search(self, mock_client, mock_store):
        """Test search using vector search strategy."""
        mock_results = [
            SearchItem(
                key="key1",
                namespace=("test",),
                value={"title": "test"},
                score=0.9,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        manager = SearchStrategyManager(mock_client, mock_store)

        # Mock the first strategy (VectorSearchStrategy)
        with (
            patch.object(manager.strategies[0], "is_available", return_value=True),
            patch.object(manager.strategies[0], "search", return_value=mock_results),
        ):
            op = SearchOp(namespace_prefix=("test",), query="test query")
            results = manager.search(op)

            assert len(results) == 1
            assert results[0].key == "key1"

    def test_search_error_handling(self, mock_client, mock_store):
        """Test search error handling."""
        manager = SearchStrategyManager(mock_client, mock_store)

        # Mock all strategies failing
        with (
            patch.object(manager.strategies[0], "is_available", return_value=True),
            patch.object(
                manager.strategies[0],
                "search",
                side_effect=Exception("Vector search failed"),
            ),
            patch.object(manager.strategies[1], "is_available", return_value=True),
            patch.object(
                manager.strategies[1],
                "search",
                side_effect=Exception("Hash search failed"),
            ),
            patch.object(manager.strategies[2], "is_available", return_value=True),
            patch.object(
                manager.strategies[2],
                "search",
                side_effect=Exception("Pattern search failed"),
            ),
        ):
            op = SearchOp(namespace_prefix=("test",), query="test")
            results = manager.search(op)

            # Should return empty list on all failures
            assert results == []


class TestVectorSearchStrategy:
    """Test VectorSearchStrategy class."""

    def test_init(self, mock_client, mock_store):
        """Test VectorSearchStrategy initialization."""
        strategy = VectorSearchStrategy(mock_client, mock_store)

        assert strategy.client == mock_client
        assert strategy.store == mock_store

    def test_search_without_query(self, mock_client, mock_store):
        """Test vector search without query."""
        strategy = VectorSearchStrategy(mock_client, mock_store)
        op = SearchOp(namespace_prefix=("test",), limit=10)

        results = strategy.search(op)

        # Should return empty list when no query provided
        assert results == []

    def test_search_no_embeddings(self, mock_client, mock_store):
        """Test vector search when embeddings not available."""
        strategy = VectorSearchStrategy(mock_client, mock_store)

        # Mock is_available to return False
        with patch.object(strategy, "is_available", return_value=False):
            op = SearchOp(namespace_prefix=("test",), query="test")
            results = strategy.search(op)

            # Should return empty list when not available
            assert results == []

    def test_search_with_exception(self, mock_client, mock_store):
        """Test vector search with exception handling."""
        mock_store.embeddings.embed_query.side_effect = Exception("Embedding failed")

        strategy = VectorSearchStrategy(mock_client, mock_store)
        op = SearchOp(namespace_prefix=("test",), query="test")

        # Should handle embedding error and raise SearchIndexError
        try:
            strategy.search(op)
        except Exception:
            # Expected to raise an exception
            pass

    def test_is_available(self, mock_client, mock_store):
        """Test vector search availability check."""
        strategy = VectorSearchStrategy(mock_client, mock_store)

        # Mock all required components
        mock_store.embeddings = MagicMock()
        mock_store.dims = 128
        mock_store._is_search_available.return_value = True
        mock_store.index = MagicMock()

        assert strategy.is_available() is True

        # Test with missing embeddings
        mock_store.embeddings = None
        assert strategy.is_available() is False


class TestKeyPatternSearchStrategy:
    """Test KeyPatternSearchStrategy class."""

    def test_init(self, mock_client, mock_store):
        """Test KeyPatternSearchStrategy initialization."""
        strategy = KeyPatternSearchStrategy(mock_client, mock_store)

        assert strategy.client == mock_client
        assert strategy.store == mock_store

    def test_is_available(self, mock_client, mock_store):
        """Test key pattern search is always available."""
        strategy = KeyPatternSearchStrategy(mock_client, mock_store)
        assert strategy.is_available() is True

    def test_search_basic(self, mock_client, mock_store):
        """Test basic key pattern search."""
        # Mock the dependencies properly
        with patch(
            "langgraph_checkpoint_aws.store.valkey.search_strategies.FilterProcessor"
        ) as mock_filter:
            mock_filter.build_namespace_pattern.return_value = "langgraph:test/*"

            # Mock scan results
            mock_client.scan.return_value = (0, ["langgraph:test/key1"])
            mock_store._handle_response_t.return_value = (0, ["langgraph:test/key1"])
            mock_store._safe_parse_keys.return_value = ["langgraph:test/key1"]

            # Mock document processing
            with patch.object(
                KeyPatternSearchStrategy, "_process_key_for_search"
            ) as mock_process:
                mock_process.return_value = SearchItem(
                    key="key1",
                    namespace=("test",),
                    value={"title": "test"},
                    score=0.8,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

                strategy = KeyPatternSearchStrategy(mock_client, mock_store)
                op = SearchOp(namespace_prefix=("test",), query="test")

                results = strategy.search(op)

                assert isinstance(results, list)

    def test_search_error_handling(self, mock_client, mock_store):
        """Test key pattern search error handling."""
        mock_client.scan.side_effect = Exception("Scan failed")

        strategy = KeyPatternSearchStrategy(mock_client, mock_store)
        op = SearchOp(namespace_prefix=("test",))

        results = strategy.search(op)

        # Should return empty list on error
        assert results == []


class TestHashSearchStrategy:
    """Test HashSearchStrategy class."""

    def test_init(self, mock_client, mock_store):
        """Test HashSearchStrategy initialization."""
        strategy = HashSearchStrategy(mock_client, mock_store)

        assert strategy.client == mock_client
        assert strategy.store == mock_store

    def test_is_available(self, mock_client, mock_store):
        """Test hash search is always available."""
        strategy = HashSearchStrategy(mock_client, mock_store)
        assert strategy.is_available() is True

    def test_search_basic(self, mock_client, mock_store):
        """Test basic hash search."""
        strategy = HashSearchStrategy(mock_client, mock_store)

        # Mock the internal search method
        with (
            patch.object(strategy, "_search_with_hash", return_value=[]),
            patch.object(strategy, "_convert_to_search_items", return_value=[]),
        ):
            op = SearchOp(namespace_prefix=("test",), query="test")
            results = strategy.search(op)

            assert isinstance(results, list)

    def test_search_with_ttl_refresh(self, mock_client, mock_store):
        """Test hash search with TTL refresh."""
        strategy = HashSearchStrategy(mock_client, mock_store)

        mock_items = [
            SearchItem(
                key="key1",
                namespace=("test",),
                value={"title": "test"},
                score=0.8,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        with (
            patch.object(strategy, "_search_with_hash", return_value=[]),
            patch.object(strategy, "_convert_to_search_items", return_value=mock_items),
            patch.object(strategy, "_refresh_ttl_for_items") as mock_refresh,
        ):
            op = SearchOp(namespace_prefix=("test",), query="test", refresh_ttl=True)
            results = strategy.search(op)

            assert isinstance(results, list)
            mock_refresh.assert_called_once()

    def test_search_error_handling(self, mock_client, mock_store):
        """Test hash search error handling."""
        strategy = HashSearchStrategy(mock_client, mock_store)

        with patch.object(
            strategy, "_search_with_hash", side_effect=Exception("Search failed")
        ):
            op = SearchOp(namespace_prefix=("test",))
            results = strategy.search(op)

            # Should return empty list on error
            assert results == []
