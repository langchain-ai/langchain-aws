"""Tests for the ValkeyCache implementation."""

import asyncio
import os
from collections.abc import Generator
from typing import Any

import pytest

# Skip all tests in this module if valkey is not available or cannot be imported
valkey = pytest.importorskip("valkey")

try:
    from valkey import Valkey
    from valkey.connection import ConnectionPool
except ImportError:
    pytest.skip("valkey package not properly installed", allow_module_level=True)

# Only import ValkeyCache after verifying valkey is available
# This prevents import errors when valkey is not installed
from langgraph_checkpoint_aws import ValkeyCache, valkey_available  # noqa: E402

VALKEY_AVAILABLE = valkey_available


def _is_valkey_server_available() -> bool:
    """Check if a Valkey server is available for testing."""
    if not VALKEY_AVAILABLE:
        return False

    try:
        from valkey import Valkey

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
def valkey_pool(valkey_url: str) -> Generator[Any, None, None]:
    """Create a ValkeyPool instance."""
    if not VALKEY_AVAILABLE:
        pytest.skip("Valkey not available")
    pool = ConnectionPool.from_url(
        url=valkey_url, min_size=1, max_connections=5, timeout=30.0
    )
    yield pool
    # Pool cleanup will be automatic


@pytest.fixture
def cache(valkey_url: str) -> ValkeyCache:
    """Create a ValkeyCache instance."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    return ValkeyCache(Valkey.from_url(valkey_url), prefix="test:cache:")


@pytest.fixture
def cache_with_ttl(valkey_url: str) -> ValkeyCache:
    """Create a ValkeyCache instance with default TTL."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    return ValkeyCache(
        Valkey.from_url(valkey_url),
        prefix="test:cache:ttl:",
        ttl=1.0,  # 1 second default TTL
    )


@pytest.fixture
def clean_cache(cache: ValkeyCache) -> Generator[ValkeyCache, None, None]:
    """Provide a clean cache and cleanup after tests."""
    yield cache
    # Clear all test data
    asyncio.run(cache.aclear())


@pytest.fixture
def clean_cache_with_ttl(
    cache_with_ttl: ValkeyCache,
) -> Generator[ValkeyCache, None, None]:
    """Provide a clean cache with TTL and cleanup after tests."""
    yield cache_with_ttl
    # Clear all test data
    asyncio.run(cache_with_ttl.aclear())


class TestValkeyCache:
    """Test suite for ValkeyCache."""

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_init(self, valkey_url: str):
        """Test ValkeyCache initialization."""
        client = Valkey.from_url(valkey_url)
        cache: ValkeyCache = ValkeyCache(client, prefix="test:", ttl=60.0)

        assert cache.client is client
        assert cache.prefix == "test:"
        assert cache.ttl == 60  # 60 seconds
        assert cache.serde is not None  # Has default serializer

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_init_with_custom_serde(self, valkey_url: str):
        """Test ValkeyCache initialization with custom serializer."""
        # Get the default serde implementation from base cache
        client = Valkey.from_url(valkey_url)
        default_cache: ValkeyCache = ValkeyCache(client)
        custom_serde = default_cache.serde
        cache: ValkeyCache = ValkeyCache(client, serde=custom_serde)

        assert cache.serde is custom_serde

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_make_key(self, clean_cache: ValkeyCache):
        """Test key creation from namespace and key."""
        # Test with empty namespace
        key = clean_cache._make_key((), "test_key")
        assert key == "test:cache:test_key"

        # Test with single namespace
        key = clean_cache._make_key(("ns1",), "test_key")
        assert key == "test:cache:ns1/test_key"

        # Test with nested namespace
        key = clean_cache._make_key(("ns1", "ns2", "ns3"), "test_key")
        assert key == "test:cache:ns1/ns2/ns3/test_key"

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_parse_key(self, clean_cache: ValkeyCache):
        """Test parsing Valkey key back to namespace and key."""
        # Test with empty namespace
        ns, key = clean_cache._parse_key("test:cache:test_key")
        assert ns == ()
        assert key == "test_key"

        # Test with single namespace
        ns, key = clean_cache._parse_key("test:cache:ns1/test_key")
        assert ns == ("ns1",)
        assert key == "test_key"

        # Test with nested namespace
        ns, key = clean_cache._parse_key("test:cache:ns1/ns2/ns3/test_key")
        assert ns == ("ns1", "ns2", "ns3")
        assert key == "test_key"

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_parse_key_invalid_prefix(self, clean_cache: ValkeyCache):
        """Test parsing key with invalid prefix raises error."""
        with pytest.raises(ValueError, match="does not start with prefix"):
            clean_cache._parse_key("invalid:key")

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aget_empty_keys(self, clean_cache: ValkeyCache):
        """Test async get with empty keys list."""
        result = await clean_cache.aget([])
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aget_nonexistent_keys(self, clean_cache: ValkeyCache):
        """Test async get with nonexistent keys."""
        keys = [((), "nonexistent1"), (("ns1",), "nonexistent2")]
        result = await clean_cache.aget(keys)
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aset_and_aget_basic(self, clean_cache: ValkeyCache):
        """Test basic async set and get operations."""
        # Test data
        test_data = {
            ((), "key1"): ({"value": "test1"}, None),
            (("ns1",), "key2"): ({"value": "test2", "number": 42}, None),
            (("ns1", "ns2"), "key3"): ([1, 2, 3], None),
        }

        # Set values
        await clean_cache.aset(test_data)  # type: ignore[arg-type]

        # Get values
        keys = list(test_data.keys())
        result = await clean_cache.aget(keys)

        assert len(result) == 3
        assert result[((), "key1")] == {"value": "test1"}
        assert result[(("ns1",), "key2")] == {"value": "test2", "number": 42}
        assert result[(("ns1", "ns2"), "key3")] == [1, 2, 3]

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_set_and_get_sync(self, clean_cache: ValkeyCache):
        """Test synchronous set and get operations."""
        # Test data
        test_data = {
            ((), "sync_key1"): ({"value": "sync_test1"}, None),
            (("sync_ns",), "sync_key2"): ({"value": "sync_test2"}, None),
        }

        # Set values synchronously
        clean_cache.set(test_data)

        # Get values synchronously
        keys = list(test_data.keys())
        result = clean_cache.get(keys)

        assert len(result) == 2
        assert result[((), "sync_key1")] == {"value": "sync_test1"}
        assert result[(("sync_ns",), "sync_key2")] == {"value": "sync_test2"}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aset_with_ttl(self, clean_cache: ValkeyCache):
        """Test async set with TTL."""
        # Set value with short TTL (6 seconds)
        test_data = {((), "ttl_key"): ({"value": "expires_soon"}, 6)}

        await clean_cache.aset(test_data)  # type: ignore[arg-type]

        # Verify value exists immediately
        result = await clean_cache.aget([((), "ttl_key")])
        assert result[((), "ttl_key")] == {"value": "expires_soon"}

        # Wait for TTL to expire (6 seconds + buffer)
        await asyncio.sleep(7)

        # Verify value is gone
        result = await clean_cache.aget([((), "ttl_key")])
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aset_with_default_ttl(self, clean_cache_with_ttl: ValkeyCache):
        """Test async set with default TTL."""
        # Set value without explicit TTL (should use default)
        test_data = {((), "default_ttl_key"): ({"value": "uses_default_ttl"}, None)}

        await clean_cache_with_ttl.aset(test_data)  # type: ignore[arg-type]

        # Verify value exists
        result = await clean_cache_with_ttl.aget([((), "default_ttl_key")])
        assert result[((), "default_ttl_key")] == {"value": "uses_default_ttl"}

        # Note: We don't wait for default TTL to expire as it's 1 second

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aset_empty_pairs(self, clean_cache: ValkeyCache):
        """Test async set with empty pairs."""
        await clean_cache.aset({})
        # Should not raise any errors

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aclear_all(self, clean_cache: ValkeyCache):
        """Test clearing all cached values."""
        # Set some test data
        test_data = {
            ((), "key1"): ({"value": "test1"}, None),
            (("ns1",), "key2"): ({"value": "test2"}, None),
            (("ns1", "ns2"), "key3"): ({"value": "test3"}, None),
        }
        await clean_cache.aset(test_data)

        # Verify data exists
        keys = list(test_data.keys())
        result = await clean_cache.aget(keys)
        assert len(result) == 3

        # Clear all
        await clean_cache.aclear()

        # Verify all data is gone
        result = await clean_cache.aget(keys)
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_aclear_specific_namespaces(self, clean_cache: ValkeyCache):
        """Test clearing specific namespaces."""
        # Set test data in different namespaces
        test_data = {
            ((), "root_key"): ({"value": "root"}, None),
            (("ns1",), "key1"): ({"value": "ns1_val"}, None),
            (("ns1", "sub"), "key2"): ({"value": "ns1_sub_val"}, None),
            (("ns2",), "key3"): ({"value": "ns2_val"}, None),
        }
        await clean_cache.aset(test_data)  # type: ignore[arg-type]

        # Clear only ns1 namespace
        await clean_cache.aclear([("ns1",)])

        # Verify ns1 data is gone but others remain
        result = await clean_cache.aget(list(test_data.keys()))
        assert ((), "root_key") in result
        assert (("ns2",), "key3") in result
        assert (("ns1",), "key1") not in result
        assert (("ns1", "sub"), "key2") not in result

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_clear_sync(self, clean_cache: ValkeyCache):
        """Test synchronous clear operation."""
        # Set some test data
        test_data = {
            ((), "sync_clear_key"): ({"value": "sync_clear_test"}, None),
        }
        clean_cache.set(test_data)  # type: ignore[arg-type]

        # Verify data exists
        result = clean_cache.get([((), "sync_clear_key")])
        assert len(result) == 1

        # Clear synchronously
        clean_cache.clear()

        # Verify data is gone
        result = clean_cache.get([((), "sync_clear_key")])
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_from_conn_string(self, valkey_url: str):
        """Test creating cache from connection string."""
        cache: ValkeyCache
        with ValkeyCache.from_conn_string(
            valkey_url, prefix="test:conn:", ttl_seconds=1800.0
        ) as cache:
            assert cache.prefix == "test:conn:"
            assert cache.ttl == 1800  # 30 minutes in seconds
            if VALKEY_AVAILABLE:
                assert isinstance(cache.client, Valkey)

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_from_conn_string_with_pool(self, valkey_url: str):
        """Test creating cache from connection string with pool."""
        cache: ValkeyCache
        with ValkeyCache.from_conn_string(
            valkey_url,
            prefix="test:pool:",
            ttl_seconds=900.0,
            pool_size=5,
            pool_timeout=20.0,
        ) as cache:
            assert cache.prefix == "test:pool:"
            assert cache.ttl == 900  # 15 minutes in seconds
            if VALKEY_AVAILABLE:
                assert isinstance(cache.client, Valkey)

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    def test_from_pool(self, valkey_pool: Any):
        """Test creating cache from existing pool."""
        cache: ValkeyCache
        with ValkeyCache.from_pool(
            valkey_pool, prefix="test:existing_pool:", ttl_seconds=2700.0
        ) as cache:
            assert cache.prefix == "test:existing_pool:"
            assert cache.ttl == 2700  # 45 minutes in seconds
            if VALKEY_AVAILABLE:
                assert isinstance(cache.client, Valkey)

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_integration_workflow(self, clean_cache: ValkeyCache):
        """Test a complete workflow with multiple operations."""
        # Step 1: Set initial data
        initial_data = {
            (("users",), "user1"): ({"name": "Alice", "age": 30}, None),
            (("users",), "user2"): ({"name": "Bob", "age": 25}, None),
            (("posts",), "post1"): ({"title": "Hello World", "author": "Alice"}, None),
        }
        await clean_cache.aset(initial_data)  # type: ignore[arg-type]

        # Step 2: Verify all data exists
        all_keys = list(initial_data.keys())
        result = await clean_cache.aget(all_keys)
        assert len(result) == 3

        # Step 3: Add more data with TTL
        ttl_data = {(("temp",), "session1"): ({"token": "abc123", "expires": True}, 6)}
        await clean_cache.aset(ttl_data)  # type: ignore[arg-type]

        # Step 4: Verify TTL data exists
        result = await clean_cache.aget([(("temp",), "session1")])
        assert len(result) == 1

        # Step 5: Clear specific namespace
        await clean_cache.aclear([("users",)])

        # Step 6: Verify users are gone but posts remain
        result = await clean_cache.aget(all_keys)
        assert (("posts",), "post1") in result
        assert (("users",), "user1") not in result
        assert (("users",), "user2") not in result

        # Step 7: Wait for TTL to expire
        await asyncio.sleep(7)

        # Step 8: Verify TTL data is gone
        result = await clean_cache.aget([(("temp",), "session1")])
        assert result == {}

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_complex_data_types(self, clean_cache: ValkeyCache):
        """Test caching complex data types."""
        complex_data = {
            (("complex",), "nested"): (
                {
                    "list": [1, 2, {"nested": "dict"}],
                    "dict": {"key": "value", "number": 42},
                    "tuple_as_list": [1, 2, 3],  # Tuples become lists in JSON
                    "boolean": True,
                    "null": None,
                    "float": 3.14159,
                },
                None,
            )
        }

        await clean_cache.aset(complex_data)  # type: ignore[arg-type]
        result = await clean_cache.aget([(("complex",), "nested")])

        expected = complex_data[(("complex",), "nested")][0]
        assert result[(("complex",), "nested")] == expected

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_batch_operations(self, clean_cache: ValkeyCache):
        """Test batch operations with many keys."""
        # Create a large batch of data
        batch_size = 50
        batch_data = {}
        for i in range(batch_size):
            ns = ("batch", f"group_{i // 10}")
            key = f"item_{i}"
            value = {"id": i, "data": f"batch_item_{i}"}
            batch_data[(ns, key)] = (value, None)

        # Set all data in batch
        await clean_cache.aset(batch_data)  # type: ignore[arg-type]

        # Get all data in batch
        keys = list(batch_data.keys())
        result = await clean_cache.aget(keys)

        assert len(result) == batch_size
        for i in range(batch_size):
            ns = ("batch", f"group_{i // 10}")
            key = f"item_{i}"
            expected_value = {"id": i, "data": f"batch_item_{i}"}
            assert result[(ns, key)] == expected_value

    @pytest.mark.skipif(
        not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available"
    )
    @pytest.mark.asyncio
    async def test_namespace_isolation(self, clean_cache: ValkeyCache):
        """Test that namespaces properly isolate data."""
        # Set same key in different namespaces
        test_data = {
            (("ns1",), "same_key"): ({"source": "ns1"}, None),
            (("ns2",), "same_key"): ({"source": "ns2"}, None),
            (("ns1", "sub"), "same_key"): ({"source": "ns1_sub"}, None),
        }
        await clean_cache.aset(test_data)

        # Verify all values are distinct
        result = await clean_cache.aget(list(test_data.keys()))
        assert result[(("ns1",), "same_key")] == {"source": "ns1"}
        assert result[(("ns2",), "same_key")] == {"source": "ns2"}
        assert result[(("ns1", "sub"), "same_key")] == {"source": "ns1_sub"}

        # Clear one namespace and verify others are unaffected
        await clean_cache.aclear([("ns1",)])

        result = await clean_cache.aget(list(test_data.keys()))
        assert (("ns1",), "same_key") not in result
        assert (("ns1", "sub"), "same_key") not in result
        assert result[(("ns2",), "same_key")] == {"source": "ns2"}
