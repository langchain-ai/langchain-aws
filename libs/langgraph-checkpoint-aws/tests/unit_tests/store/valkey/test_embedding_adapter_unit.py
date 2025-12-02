"""Unit tests for EmbeddingAdapter - simplified test approach."""

import pytest

from langgraph_checkpoint_aws.store.valkey.exceptions import SearchIndexError
from langgraph_checkpoint_aws.store.valkey.search import EmbeddingAdapter


class MockSyncEmbeddings:
    """Mock embeddings with only sync embed_query."""

    def embed_query(self, query: str):
        return [1.0, 2.0, 3.0]


class MockAsyncEmbeddings:
    """Mock embeddings with only async aembed_query."""

    async def aembed_query(self, query: str):
        return [4.0, 5.0, 6.0]


class MockCallableEmbeddings:
    """Mock callable embeddings."""

    def __call__(self, queries: list[str]):
        return [[7.0, 8.0, 9.0] for _ in queries]


class TestEmbeddingAdapter:
    """Test EmbeddingAdapter with real-ish mock objects."""

    def test_sync_embeddings_detection(self):
        """Test detection of sync embeddings."""
        embeddings = MockSyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        assert adapter.can_embed_sync() is True
        assert adapter.can_embed_async() is False  # No async methods, won't fall back

    def test_async_embeddings_detection(self):
        """Test detection of async embeddings."""
        embeddings = MockAsyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        assert adapter.can_embed_sync() is False
        assert adapter.can_embed_async() is True

    def test_callable_embeddings_detection(self):
        """Test detection of callable embeddings."""
        embeddings = MockCallableEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        assert adapter.can_embed_sync() is True
        assert adapter.can_embed_async() is True

    def test_sync_embedding_generation(self):
        """Test sync embedding generation."""
        embeddings = MockSyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = adapter.embed_query_sync("test query")
        assert result == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_async_embedding_generation(self):
        """Test async embedding generation."""
        embeddings = MockAsyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = await adapter.embed_query_async("test query")
        assert result == [4.0, 5.0, 6.0]

    def test_callable_embedding_sync(self):
        """Test callable embeddings in sync mode."""
        embeddings = MockCallableEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = adapter.embed_query_sync("test query")
        assert result == [7.0, 8.0, 9.0]  # Adapter extracts first element

    @pytest.mark.asyncio
    async def test_callable_embedding_async(self):
        """Test callable embeddings in async mode."""
        embeddings = MockCallableEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = await adapter.embed_query_async("test query")
        assert result == [7.0, 8.0, 9.0]  # Adapter extracts first element

    def test_sync_with_async_only_raises_error(self):
        """Test sync embedding raises error when only async available."""
        embeddings = MockAsyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        with pytest.raises(SearchIndexError) as exc_info:
            adapter.embed_query_sync("test query")

        assert "only has async methods" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_with_sync_only_raises_error(self):
        """Test async embedding raises error when only sync methods available."""
        embeddings = MockSyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        with pytest.raises(SearchIndexError) as exc_info:
            await adapter.embed_query_async("test query")

        assert "only has sync methods" in str(exc_info.value)
        assert "Use ValkeyStore" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_with_no_methods_returns_none(self):
        """Test async embedding returns None when no methods available."""

        class NoMethods:
            pass

        adapter = EmbeddingAdapter(NoMethods())
        result = await adapter.embed_query_async("test query")

        assert result is None

    def test_sync_with_no_methods_raises_error(self):
        """Test sync embedding raises error when no methods available."""

        class NoMethods:
            pass

        adapter = EmbeddingAdapter(NoMethods())

        with pytest.raises(SearchIndexError) as exc_info:
            adapter.embed_query_sync("test query")

        assert "No embedding method available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_handles_exceptions(self):
        """Test async embedding returns None on exception."""

        class FailingEmbeddings:
            async def aembed_query(self, query: str):
                raise RuntimeError("Embedding failed")

        adapter = EmbeddingAdapter(FailingEmbeddings())
        result = await adapter.embed_query_async("test query")

        assert result is None

    def test_none_embeddings(self):
        """Test adapter with None embeddings."""
        adapter = EmbeddingAdapter(None)

        assert adapter.can_embed_sync() is False
        assert adapter.can_embed_async() is False
