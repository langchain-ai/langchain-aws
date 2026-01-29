"""Integration tests for Valkey filter expressions.

These tests require a running Valkey instance with the search module loaded.
Set VALKEY_HOST environment variable to specify the Valkey server.
"""

import asyncio
import os

import pytest

pytest.importorskip("glide")

from langchain_core.embeddings import DeterministicFakeEmbedding
from glide import GlideClient, GlideClientConfiguration, NodeAddress, ProtocolVersion

from langchain_aws.vectorstores.valkey import ValkeyVectorStore
from langchain_aws.vectorstores.valkey.filters import ValkeyFilter


@pytest.fixture(scope="module")
def valkey_url() -> str:
    """Get Valkey connection URL from environment."""
    host = os.getenv("VALKEY_HOST", "localhost")
    return f"valkey://{host}"


@pytest.fixture(scope="module")
def embeddings() -> DeterministicFakeEmbedding:
    """Create deterministic fake embeddings for testing."""
    return DeterministicFakeEmbedding(size=128)


@pytest.fixture(scope="module")
def index_name() -> str:
    """Return unique index name for tests."""
    return "test_filters_index"


async def _create_client(host: str) -> GlideClient:
    """Create GLIDE client."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress(host, 6379)],
        protocol=ProtocolVersion.RESP3,
    )
    return await GlideClient.create(config)


@pytest.fixture(scope="module")
def vectorstore(valkey_url: str, embeddings: DeterministicFakeEmbedding, index_name: str) -> ValkeyVectorStore:
    """Create and populate a Valkey vector store with test data."""
    host = os.getenv("VALKEY_HOST", "localhost")
    
    async def setup():
        client = await _create_client(host)
        
        # Drop index if exists
        try:
            await client.custom_command(["FT.DROPINDEX", index_name])
        except Exception:
            pass
        
        # Create index using raw FT.CREATE command
        # Schema: VECTOR content_vector FLAT 6 TYPE FLOAT32 DIM 128 DISTANCE_METRIC COSINE TAG category NUMERIC year NUMERIC price
        await client.custom_command([
            "FT.CREATE", index_name,
            "ON", "HASH",
            "PREFIX", "1", f"doc:{index_name}:",
            "SCHEMA",
            "content_vector", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", "128",
            "DISTANCE_METRIC", "COSINE",
            "category", "TAG",
            "year", "NUMERIC",
            "price", "NUMERIC",
        ])
        
        await client.close()
    
    asyncio.run(setup())
    
    # Create vector store
    store = ValkeyVectorStore(
        embedding=embeddings,
        valkey_url=valkey_url,
        index_name=index_name,
        vector_schema={
            "name": "content_vector",
            "algorithm": "FLAT",
            "dims": 128,
            "distance_metric": "COSINE",
            "datatype": "FLOAT32",
        }
    )
    
    # Add test documents
    texts = [
        "Laptop computer with high performance",
        "Desktop computer for gaming",
        "Tablet device for reading",
        "Smartphone with camera",
        "Wireless headphones",
    ]
    metadatas = [
        {"category": "electronics", "year": 2024, "price": 1200},
        {"category": "electronics", "year": 2023, "price": 1500},
        {"category": "electronics", "year": 2024, "price": 500},
        {"category": "mobile", "year": 2024, "price": 800},
        {"category": "audio", "year": 2023, "price": 200},
    ]
    store.add_texts(texts, metadatas=metadatas)
    
    yield store
    
    # Cleanup - drop index and delete documents
    async def cleanup():
        client = await _create_client(host)
        try:
            # Delete all documents with the index prefix
            keys = await client.keys(f"doc:{index_name}:*")
            if keys:
                await client.delete(keys)
            # Drop the index
            await client.custom_command(["FT.DROPINDEX", index_name])
        except Exception:
            pass
        finally:
            await client.close()
    
    asyncio.run(cleanup())


class TestValkeyTagFilters:
    """Test tag filter expressions."""

    def test_tag_equals_single(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.tag("category") == "electronics"
        results = vectorstore.similarity_search("computer", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(doc.metadata.get("category") == "electronics" for doc in results)

    def test_tag_equals_multiple(self, vectorstore: ValkeyVectorStore) -> None:
        # Note: Multiple tag matching with OR requires tags to be stored in tag field format
        # Currently metadata is stored as regular hash fields, so this tests the filter generation
        filter_expr = ValkeyFilter.tag("category") == ["electronics", "mobile"]
        assert str(filter_expr) == "@category:{electronics|mobile}"

    def test_tag_not_equals(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.tag("category") != "electronics"
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 2
        assert all(doc.metadata.get("category") != "electronics" for doc in results)


class TestValkeyNumFilters:
    """Test numeric filter expressions."""

    def test_num_equals(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.num("year") == 2024
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(doc.metadata.get("year") == "2024" for doc in results)

    def test_num_greater_than(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.num("price") > 500
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(int(doc.metadata.get("price", 0)) > 500 for doc in results)

    def test_num_less_than(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.num("price") < 1000
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(int(doc.metadata.get("price", 0)) < 1000 for doc in results)

    def test_num_greater_equal(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.num("price") >= 800
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(int(doc.metadata.get("price", 0)) >= 800 for doc in results)

    def test_num_less_equal(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = ValkeyFilter.num("price") <= 800
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(int(doc.metadata.get("price", 0)) <= 800 for doc in results)


class TestValkeyTextFilters:
    """Test text filter expressions."""

    def test_text_equals(self, vectorstore: ValkeyVectorStore) -> None:
        # Text filters require TextField in schema, which isn't supported by valkey-search yet
        # This test validates the filter expression generation
        filter_expr = ValkeyFilter.text("content") == "computer"
        assert str(filter_expr) == '@content:("computer")'

    def test_text_like(self, vectorstore: ValkeyVectorStore) -> None:
        # Text filters require TextField in schema, which isn't supported by valkey-search yet
        # This test validates the filter expression generation
        filter_expr = ValkeyFilter.text("content") % "comput*"
        assert str(filter_expr) == "@content:(comput*)"


class TestValkeyComplexFilters:
    """Test complex filter expressions with AND/OR operations."""

    def test_and_operation(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = (ValkeyFilter.tag("category") == "electronics") & (ValkeyFilter.num("year") == 2024)
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 2
        assert all(doc.metadata.get("category") == "electronics" for doc in results)
        assert all(doc.metadata.get("year") == "2024" for doc in results)

    def test_or_operation(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = (ValkeyFilter.tag("category") == "mobile") | (ValkeyFilter.tag("category") == "audio")
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 2
        categories = {doc.metadata.get("category") for doc in results}
        assert categories == {"mobile", "audio"}

    def test_complex_nested(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = (
            (ValkeyFilter.tag("category") == "electronics") & (ValkeyFilter.num("price") > 1000)
        ) | (ValkeyFilter.tag("category") == "audio")
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        
        electronics_expensive = [
            doc for doc in results 
            if doc.metadata.get("category") == "electronics" and int(doc.metadata.get("price", 0)) > 1000
        ]
        audio = [doc for doc in results if doc.metadata.get("category") == "audio"]
        assert len(electronics_expensive) + len(audio) == len(results)

    def test_price_range(self, vectorstore: ValkeyVectorStore) -> None:
        filter_expr = (ValkeyFilter.num("price") >= 500) & (ValkeyFilter.num("price") <= 1200)
        results = vectorstore.similarity_search("product", k=10, filter=str(filter_expr))
        assert len(results) == 3
        assert all(500 <= int(doc.metadata.get("price", 0)) <= 1200 for doc in results)
