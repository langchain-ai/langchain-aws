"""Unit tests for EmbeddingAdapter.

Tests the adapter's ability to route to correct sync/async methods
and provide helpful errors when using the wrong store type.
"""

import pytest

# Skip entire module if valkey not available
pytest.importorskip("valkey")

from langgraph_checkpoint_aws.store.valkey.exceptions import SearchIndexError
from langgraph_checkpoint_aws.store.valkey.search import EmbeddingAdapter


class MockSyncEmbeddings:
    """Mock embeddings with only sync embed_query."""

    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]


class MockAsyncEmbeddings:
    """Mock embeddings with only async aembed_query."""

    async def aembed_query(self, query: str):
        return [0.1, 0.2, 0.3]


class MockCallableEmbeddings:
    """Mock callable embeddings."""

    def __call__(self, queries: list[str]):
        return [[0.1, 0.2, 0.3] for _ in queries]


class TestEmbeddingAdapter:
    """Test suite for EmbeddingAdapter."""

    def test_sync_embedding_generation(self):
        """Test sync embedding generation with sync embeddings."""
        embeddings = MockSyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = adapter.embed_query_sync("test query")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_async_embedding_generation(self):
        """Test async embedding generation with async embeddings."""
        embeddings = MockAsyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = await adapter.embed_query_async("test query")
        assert result == [0.1, 0.2, 0.3]

    def test_callable_embedding_sync(self):
        """Test callable embeddings work in sync mode."""
        embeddings = MockCallableEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = adapter.embed_query_sync("test query")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_callable_embedding_async(self):
        """Test callable embeddings work in async mode (runs in executor)."""
        embeddings = MockCallableEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        result = await adapter.embed_query_async("test query")
        assert result == [0.1, 0.2, 0.3]

    def test_sync_with_async_only_raises_error(self):
        """Test helpful error when using sync method with async-only embeddings.

        This catches the common mistake of using ValkeyStore with async embeddings.
        """
        embeddings = MockAsyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        with pytest.raises(SearchIndexError) as exc_info:
            adapter.embed_query_sync("test query")

        assert "async methods" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_with_sync_only_raises_error(self):
        """Test helpful error when using async method with sync-only embeddings.

        This catches the common mistake of using AsyncValkeyStore with sync embeddings.
        """
        embeddings = MockSyncEmbeddings()
        adapter = EmbeddingAdapter(embeddings)

        with pytest.raises(SearchIndexError) as exc_info:
            await adapter.embed_query_async("test query")

        assert "sync methods" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_with_no_methods_returns_none(self):
        """Test that async embedding with no methods logs warning and returns None.

        This is a graceful failure mode rather than raising an exception.
        """

        class NoMethods:
            pass

        adapter = EmbeddingAdapter(NoMethods())
        result = await adapter.embed_query_async("test")

        assert result is None

    def test_sync_with_no_methods_raises_error(self):
        """Test that sync embedding with no methods raises clear error.

        Sync mode fails fast rather than returning None.
        """

        class NoMethods:
            pass

        adapter = EmbeddingAdapter(NoMethods())

        with pytest.raises(SearchIndexError) as exc_info:
            adapter.embed_query_sync("test")

        assert "No embedding method available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_handles_exceptions(self):
        """Test that async embedding handles exceptions gracefully by returning None.

        This prevents one failed embedding from breaking an entire batch.
        """

        class FailingEmbeddings:
            async def aembed_query(self, query: str):
                raise RuntimeError("Embedding failed")

        adapter = EmbeddingAdapter(FailingEmbeddings())

        result = await adapter.embed_query_async("test")
        assert result is None
