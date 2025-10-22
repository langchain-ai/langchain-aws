"""Tests for the AsyncValkeyCheckpointSaver implementation."""

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio

from langgraph_checkpoint_aws import AsyncValkeySaver

# Check for optional dependencies
try:
    import orjson  # noqa: F401
    import valkey  # noqa: F401
    from valkey.asyncio import Valkey as AsyncValkey  # noqa: F401
    from valkey.asyncio.connection import ConnectionPool as AsyncConnectionPool
    from valkey.exceptions import ValkeyError  # noqa: F401

    VALKEY_AVAILABLE = True
except ImportError:
    AsyncValkey = None  # type: ignore[assignment, misc]
    AsyncConnectionPool = None  # type: ignore[assignment, misc]
    VALKEY_AVAILABLE = False

# Skip all tests if valkey dependencies are not available
pytestmark = pytest.mark.skipif(
    not VALKEY_AVAILABLE,
    reason=(
        "valkey and orjson dependencies not available. "
        "Install with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ),
)


def _is_valkey_server_available() -> bool:
    """Check if a Valkey server is available for testing."""
    if not VALKEY_AVAILABLE or AsyncValkey is None:
        return False

    try:
        import asyncio

        valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")

        async def check_connection():
            client = AsyncValkey.from_url(valkey_url)
            try:
                await client.ping()
                return True
            except Exception:
                return False
            finally:
                await client.aclose()

        return asyncio.run(check_connection())
    except Exception:
        return False


VALKEY_SERVER_AVAILABLE = _is_valkey_server_available()


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey server URL from environment or use default."""
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest_asyncio.fixture
async def async_valkey_pool(valkey_url: str) -> Any:
    """Create an AsyncConnectionPool instance."""
    if not VALKEY_AVAILABLE:
        pytest.skip("Valkey not available")
    pool = AsyncConnectionPool.from_url(
        valkey_url, max_connections=5, retry_on_timeout=True
    )
    return pool


@pytest_asyncio.fixture
async def async_saver(
    valkey_url: str,
) -> AsyncGenerator[AsyncValkeySaver, None]:
    """Create an AsyncValkeySaver instance."""
    if not VALKEY_AVAILABLE or AsyncValkey is None:
        pytest.skip("Valkey not available")
    client = AsyncValkey.from_url(valkey_url)
    saver = AsyncValkeySaver(client, ttl=60.0)
    yield saver
    await client.aclose()


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_from_conn_string(valkey_url: str) -> None:
    """Test creating async saver from connection string."""
    async with AsyncValkeySaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        assert saver.ttl == 3600  # 3600 seconds


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_from_pool(async_valkey_pool: Any) -> None:
    """Test creating async saver from existing pool."""
    async with AsyncValkeySaver.from_pool(
        async_valkey_pool, ttl_seconds=3600.0
    ) as saver:
        assert saver.ttl == 3600


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_operations(valkey_url: str) -> None:
    """Test async operations using connection pool."""
    async with AsyncValkeySaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        metadata = {"timestamp": datetime.now().isoformat(), "user": "test"}
        new_versions: dict[str, int] = {}

        # Store checkpoint
        result = await saver.aput(
            config,  # type: ignore[arg-type]
            checkpoint,  # type: ignore[arg-type]
            metadata,  # type: ignore[arg-type]
            new_versions,  # type: ignore[arg-type]
        )
        assert result["configurable"]["checkpoint_id"] == checkpoint["id"]  # type: ignore

        # Get checkpoint
        checkpoint_tuple = await saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint["id"],
                }
            }
        )
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]  # type: ignore
        assert checkpoint_tuple.checkpoint["state"] == checkpoint["state"]  # type: ignore
        assert checkpoint_tuple.metadata["user"] == metadata["user"]  # type: ignore


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_async_shared_pool(async_valkey_pool: Any) -> None:
    """Test sharing connection pool between async savers."""
    async with (
        AsyncValkeySaver.from_pool(async_valkey_pool, ttl_seconds=3600.0) as saver1,
        AsyncValkeySaver.from_pool(async_valkey_pool, ttl_seconds=3600.0) as saver2,
    ):
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint1 = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        checkpoint2 = {"id": "test-2", "state": {"value": 2}, "versions": {}}
        metadata = {"timestamp": datetime.now().isoformat(), "user": "test"}
        new_versions: dict[str, int] = {}

        # Store checkpoints in both savers
        await saver1.aput(config, checkpoint1, metadata, new_versions)  # type: ignore[arg-type]
        await saver2.aput(config, checkpoint2, metadata, new_versions)  # type: ignore[arg-type]

        # Get checkpoints from both savers
        result1 = await saver1.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint1["id"],
                }
            }
        )
        result2 = await saver2.aget_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint2["id"],
                }
            }
        )

        assert result1 is not None
        assert result2 is not None
        assert result1.checkpoint["id"] == checkpoint1["id"]  # type: ignore
        assert result2.checkpoint["id"] == checkpoint2["id"]  # type: ignore


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
@pytest.mark.asyncio
async def test_alist_checkpoints_before_nonexistent(
    async_saver: AsyncValkeySaver,
) -> None:
    """Test listing checkpoints with before filter for nonexistent checkpoint."""
    thread_id = f"test-thread-before-nonexistent-{uuid.uuid4()}"
    checkpoint_ns = "test"

    # Store a checkpoint
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    checkpoint = {
        "id": "checkpoint-1",
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"value": 1},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
    await async_saver.aput(config, checkpoint, metadata, {})  # type: ignore

    # List checkpoints before nonexistent checkpoint
    before_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": "nonexistent-checkpoint",
        }
    }

    result = []
    async for checkpoint_tuple in async_saver.alist(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
        before=before_config,  # type: ignore
    ):
        result.append(checkpoint_tuple)

    # Should get all checkpoints since before checkpoint doesn't exist
    assert len(result) == 1
    assert result[0].checkpoint["id"] == "checkpoint-1"
