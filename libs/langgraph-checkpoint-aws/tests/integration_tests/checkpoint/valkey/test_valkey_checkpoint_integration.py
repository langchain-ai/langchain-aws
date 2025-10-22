"""Integration tests for ValkeyCheckpointSaver implementation."""

import asyncio
import os
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph_checkpoint_aws import ValkeySaver

# Check for optional dependencies
try:
    import orjson  # noqa: F401
    import valkey  # noqa: F401
    from valkey import Valkey  # noqa: F401
    from valkey.connection import ConnectionPool
    from valkey.exceptions import ValkeyError  # noqa: F401

    VALKEY_AVAILABLE = True
except ImportError:
    Valkey = None  # type: ignore[assignment, misc]
    ConnectionPool = None  # type: ignore[assignment, misc]
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
def valkey_pool(valkey_url: str) -> Generator[Any, None, None]:
    """Create a ValkeyPool instance."""
    if not VALKEY_AVAILABLE:
        pytest.skip("Valkey not available")
    pool = ConnectionPool.from_url(valkey_url, max_connections=5)
    yield pool
    # Pool cleanup will be automatic


@pytest.fixture
def saver(valkey_url: str) -> ValkeySaver:
    """Create a ValkeySaver instance."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")
    return ValkeySaver(Valkey.from_url(valkey_url), ttl=60.0)


# Basic Integration Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_from_conn_string(valkey_url: str) -> None:
    """Test creating saver from connection string."""
    with ValkeySaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        assert saver.ttl == 3600  # 3600 seconds


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_from_pool(valkey_pool: Any) -> None:
    """Test creating saver from existing pool."""
    with ValkeySaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver:
        assert saver.ttl == 3600


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_operations(valkey_url: str) -> None:
    """Test sync operations using connection pool."""
    with ValkeySaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        new_versions: dict[str, int] = {}

        # Store checkpoint
        result = saver.put(config, checkpoint, metadata, new_versions)  # type: ignore[arg-type]
        assert result["configurable"]["checkpoint_id"] == checkpoint["id"]  # type: ignore

        # Get checkpoint
        checkpoint_tuple = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint["id"],
                }
            }
        )  # type: ignore[arg-type]
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]  # type: ignore[typeddict-item]
        assert checkpoint_tuple.checkpoint["state"] == checkpoint["state"]  # type: ignore[typeddict-item]
        assert checkpoint_tuple.metadata["user"] == metadata["user"]  # type: ignore[typeddict-item]


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_sync_shared_pool(valkey_pool: Any) -> None:
    """Test sharing connection pool between savers."""
    with (
        ValkeySaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver1,
        ValkeySaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver2,
    ):
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint1 = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        checkpoint2 = {"id": "test-2", "state": {"value": 2}, "versions": {}}
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        new_versions: dict[str, int] = {}

        # Store checkpoints in both savers
        saver1.put(config, checkpoint1, metadata, new_versions)  # type: ignore[arg-type]
        saver2.put(config, checkpoint2, metadata, new_versions)  # type: ignore[arg-type]

        # Get checkpoints from both savers
        result1 = saver1.get_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint1["id"],
                }
            }
        )
        result2 = saver2.get_tuple(
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


# Coverage Tests


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_get_tuple_nonexistent_checkpoint(saver: ValkeySaver) -> None:
    """Test getting a nonexistent checkpoint returns None."""
    config = {
        "configurable": {
            "thread_id": "nonexistent-thread",
            "checkpoint_ns": "test",
            "checkpoint_id": "nonexistent-checkpoint",
        }
    }
    result = saver.get_tuple(config)  # type: ignore[arg-type]
    assert result is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_get_tuple_latest_checkpoint_empty_thread(saver: ValkeySaver) -> None:
    """Test getting latest checkpoint from empty thread returns None."""
    config = {"configurable": {"thread_id": "empty-thread", "checkpoint_ns": "test"}}
    result = saver.get_tuple(config)  # type: ignore
    assert result is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_get_tuple_latest_checkpoint_with_data(saver: ValkeySaver) -> None:
    """Test getting latest checkpoint when data exists."""
    # First store a checkpoint
    config = {
        "configurable": {"thread_id": "test-thread-latest", "checkpoint_ns": "test"}
    }
    checkpoint = {
        "id": "test-checkpoint-1",
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"value": 42},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
    new_versions: dict[str, int] = {}

    # Store checkpoint
    saver.put(config, checkpoint, metadata, new_versions)  # type: ignore[arg-type]

    # Get latest checkpoint (without specifying checkpoint_id)
    result = saver.get_tuple(config)  # type: ignore[arg-type]
    assert result is not None
    assert result.checkpoint["id"] == checkpoint["id"]
    assert result.checkpoint["channel_values"] == checkpoint["channel_values"]


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_empty_config(saver: ValkeySaver) -> None:
    """Test listing checkpoints with None config returns empty iterator."""
    result = list(saver.list(None))
    assert result == []


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_with_before_filter(saver: ValkeySaver) -> None:
    """Test listing checkpoints with before filter."""
    thread_id = f"test-thread-before-{uuid.uuid4()}"
    checkpoint_ns = "test"

    # Store multiple checkpoints
    for i in range(3):
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = {
            "id": f"checkpoint-{i}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"value": i},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        saver.put(config, checkpoint, metadata, {})  # type: ignore[arg-type]

    # List checkpoints before the first one
    before_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": "checkpoint-2",  # Most recent
        }
    }

    result = list(
        saver.list(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
            before=before_config,  # type: ignore[arg-type]
        )
    )

    # Should get checkpoints before checkpoint-2
    assert len(result) == 2
    assert result[0].checkpoint["id"] == "checkpoint-1"
    assert result[1].checkpoint["id"] == "checkpoint-0"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_with_limit(saver: ValkeySaver) -> None:
    """Test listing checkpoints with limit."""
    thread_id = f"test-thread-limit-{uuid.uuid4()}"
    checkpoint_ns = "test"

    # Store multiple checkpoints
    for i in range(5):
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = {
            "id": f"checkpoint-{i}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"value": i},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        saver.put(config, checkpoint, metadata, {})  # type: ignore

    # List with limit
    result = list(
        saver.list(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
            limit=2,
        )
    )

    assert len(result) == 2
    assert result[0].checkpoint["id"] == "checkpoint-4"  # Most recent
    assert result[1].checkpoint["id"] == "checkpoint-3"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_with_metadata_filter(saver: ValkeySaver) -> None:
    """Test listing checkpoints with metadata filter."""
    thread_id = f"test-thread-filter-{uuid.uuid4()}"
    checkpoint_ns = "test"

    # Store checkpoints with different metadata
    for i in range(3):
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = {
            "id": f"checkpoint-{i}",
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"value": i},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": "test" if i % 2 == 0 else "other",
        }
        saver.put(config, checkpoint, metadata, {})  # type: ignore

    # List with metadata filter
    result = list(
        saver.list(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
            filter={"user": "test"},
        )
    )

    # Should only get checkpoints with user="test"
    assert len(result) == 2
    for checkpoint_tuple in result:
        assert checkpoint_tuple.metadata["user"] == "test"  # type: ignore


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_put_writes(saver: ValkeySaver) -> None:
    """Test storing writes linked to a checkpoint."""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "test-thread-writes",
            "checkpoint_ns": "test",
            "checkpoint_id": "test-checkpoint",
        }
    }

    # First store a checkpoint
    checkpoint = {
        "id": "test-checkpoint",
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"value": 1},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
    saver.put(
        {"configurable": {"thread_id": "test-thread-writes", "checkpoint_ns": "test"}},
        checkpoint,  # type: ignore
        metadata,  # type: ignore
        {},
    )

    # Store writes
    writes = [("channel1", "value1"), ("channel2", "value2")]
    saver.put_writes(config, writes, "task-1")

    # Store additional writes
    more_writes = [("channel3", "value3")]
    saver.put_writes(config, more_writes, "task-2")

    # Get checkpoint to verify writes are stored
    result = saver.get_tuple(config)  # type: ignore
    assert result is not None
    # The writes should be included in the checkpoint tuple
    if result.pending_writes:
        assert len(result.pending_writes) >= 3


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_delete_thread(saver: ValkeySaver) -> None:
    """Test deleting all data for a thread."""
    thread_id = "test-thread-delete"

    # Store checkpoints in multiple namespaces
    for ns in ["ns1", "ns2"]:
        for i in range(2):
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}
            checkpoint = {
                "id": f"checkpoint-{i}",
                "ts": datetime.now(timezone.utc).isoformat(),
                "channel_values": {"value": i},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": "test",
            }
            saver.put(config, checkpoint, metadata, {})  # type: ignore

            # Also store writes
            writes_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": f"checkpoint-{i}",
                }
            }
            saver.put_writes(writes_config, [("channel", "value")], "task")

    # Verify data exists
    result = saver.get_tuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "ns1",
                "checkpoint_id": "checkpoint-0",
            }
        }
    )
    assert result is not None

    # Delete thread
    saver.delete_thread(thread_id)

    # Verify data is deleted
    result = saver.get_tuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "ns1",
                "checkpoint_id": "checkpoint-0",
            }
        }
    )
    assert result is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_async_methods_not_implemented(saver: ValkeySaver) -> None:
    """Test that async methods raise NotImplementedError."""
    config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}

    # Test aget_tuple
    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(saver.aget_tuple(config))  # type: ignore
    assert "The ValkeySaver does not support async methods" in str(exc_info.value)
    assert "AsyncValkeySaver" in str(exc_info.value)

    # Test alist
    async def test_alist():
        async for _ in saver.alist(config):  # type: ignore
            pass

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(test_alist())
    assert "The ValkeySaver does not support async methods" in str(exc_info.value)

    # Test aput
    checkpoint = {
        "id": "test-checkpoint",
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"value": 1},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(saver.aput(config, checkpoint, metadata, {}))  # type: ignore
    assert "The ValkeySaver does not support async methods" in str(exc_info.value)


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_missing_checkpoint_data(saver: ValkeySaver) -> None:
    """Test listing checkpoints when checkpoint data is missing."""
    thread_id = "test-thread-missing"
    checkpoint_ns = "test"

    # Manually add checkpoint ID to thread list without storing checkpoint data
    thread_key = f"thread:{thread_id}:{checkpoint_ns}"
    saver.client.lpush(thread_key, "missing-checkpoint")

    # List checkpoints - should skip missing checkpoint
    result = list(
        saver.list(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
        )
    )

    assert len(result) == 0


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_get_tuple_missing_checkpoint_data(saver: ValkeySaver) -> None:
    """Test getting checkpoint when checkpoint data is missing."""
    thread_id = "test-thread-missing-data"
    checkpoint_ns = "test"

    # Manually add checkpoint ID to thread list without storing checkpoint data
    thread_key = f"thread:{thread_id}:{checkpoint_ns}"
    saver.client.lpush(thread_key, "missing-checkpoint")

    # Try to get latest checkpoint - should return None
    result = saver.get_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    )

    assert result is None


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_list_checkpoints_before_nonexistent(saver: ValkeySaver) -> None:
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
    saver.put(config, checkpoint, metadata, {})  # type: ignore

    # List checkpoints before nonexistent checkpoint
    before_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": "nonexistent-checkpoint",
        }
    }

    result = list(
        saver.list(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
            before=before_config,  # type: ignore
        )
    )

    # Should get all checkpoints since before checkpoint doesn't exist
    assert len(result) == 1
    assert result[0].checkpoint["id"] == "checkpoint-1"


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_initialization_with_different_parameters() -> None:
    """Test ValkeySaver initialization with different parameters."""
    if not VALKEY_AVAILABLE or Valkey is None:
        pytest.skip("Valkey not available")

    valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")
    client = Valkey.from_url(valkey_url)

    # Test with no TTL
    saver1 = ValkeySaver(client)
    assert saver1.ttl is None
    assert saver1.lock is not None

    # Test with TTL
    saver2 = ValkeySaver(client, ttl=3600.0)
    assert saver2.ttl == 3600.0

    # Test with custom serde (None is valid)
    saver3 = ValkeySaver(client, serde=None)
    assert saver3.serde is not None  # Should use default serde


@pytest.mark.skipif(not VALKEY_SERVER_AVAILABLE, reason="Valkey server not available")
def test_from_conn_string_with_kwargs() -> None:
    """Test creating saver from connection string with additional kwargs."""
    valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")

    with ValkeySaver.from_conn_string(
        valkey_url,
        ttl_seconds=1800.0,
        pool_size=15,
        socket_timeout=30,
        socket_connect_timeout=10,
    ) as saver:
        assert saver.ttl == 1800.0
        # Verify the saver works
        config = {"configurable": {"thread_id": "test-kwargs", "checkpoint_ns": "test"}}
        result = saver.get_tuple(config)  # type: ignore
        assert result is None  # Should work without error
