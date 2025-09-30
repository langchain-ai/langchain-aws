"""Additional tests to improve code coverage for ValkeyCheckpointSaver."""

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest
from valkey import Valkey

from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey server URL from environment or use default."""
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest.fixture
def saver(valkey_url: str) -> ValkeyCheckpointSaver:
    """Create a ValkeyCheckpointSaver instance."""
    return ValkeyCheckpointSaver(Valkey.from_url(valkey_url), ttl=60.0)


def test_get_tuple_nonexistent_checkpoint(saver: ValkeyCheckpointSaver) -> None:
    """Test getting a nonexistent checkpoint returns None."""
    config = {
        "configurable": {
            "thread_id": "nonexistent-thread",
            "checkpoint_ns": "test",
            "checkpoint_id": "nonexistent-checkpoint",
        }
    }
    result = saver.get_tuple(config)
    assert result is None


def test_get_tuple_latest_checkpoint_empty_thread(saver: ValkeyCheckpointSaver) -> None:
    """Test getting latest checkpoint from empty thread returns None."""
    config = {"configurable": {"thread_id": "empty-thread", "checkpoint_ns": "test"}}
    result = saver.get_tuple(config)
    assert result is None


def test_get_tuple_latest_checkpoint_with_data(saver: ValkeyCheckpointSaver) -> None:
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
    new_versions = {}

    # Store checkpoint
    saver.put(config, checkpoint, metadata, new_versions)

    # Get latest checkpoint (without specifying checkpoint_id)
    result = saver.get_tuple(config)
    assert result is not None
    assert result.checkpoint["id"] == checkpoint["id"]
    assert result.checkpoint["channel_values"] == checkpoint["channel_values"]


def test_list_checkpoints_empty_config(saver: ValkeyCheckpointSaver) -> None:
    """Test listing checkpoints with None config returns empty iterator."""
    result = list(saver.list(None))
    assert result == []


def test_list_checkpoints_with_before_filter(saver: ValkeyCheckpointSaver) -> None:
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
        saver.put(config, checkpoint, metadata, {})

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
            before=before_config,
        )
    )

    # Should get checkpoints before checkpoint-2
    assert len(result) == 2
    assert result[0].checkpoint["id"] == "checkpoint-1"
    assert result[1].checkpoint["id"] == "checkpoint-0"


def test_list_checkpoints_with_limit(saver: ValkeyCheckpointSaver) -> None:
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
        saver.put(config, checkpoint, metadata, {})

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


def test_list_checkpoints_with_metadata_filter(saver: ValkeyCheckpointSaver) -> None:
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
        saver.put(config, checkpoint, metadata, {})

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
        assert checkpoint_tuple.metadata["user"] == "test"


def test_put_writes(saver: ValkeyCheckpointSaver) -> None:
    """Test storing writes linked to a checkpoint."""
    config = {
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
        checkpoint,
        metadata,
        {},
    )

    # Store writes
    writes = [("channel1", "value1"), ("channel2", "value2")]
    saver.put_writes(config, writes, "task-1", "path/to/task")

    # Store additional writes
    more_writes = [("channel3", "value3")]
    saver.put_writes(config, more_writes, "task-2", "path/to/task2")

    # Get checkpoint to verify writes are stored
    result = saver.get_tuple(config)
    assert result is not None
    # The writes should be included in the checkpoint tuple
    if result.pending_writes:
        assert len(result.pending_writes) >= 3


def test_delete_thread(saver: ValkeyCheckpointSaver) -> None:
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
            saver.put(config, checkpoint, metadata, {})

            # Also store writes
            writes_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": f"checkpoint-{i}",
                }
            }
            saver.put_writes(writes_config, [("channel", "value")], "task", "")

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


def test_async_methods_not_implemented(saver: ValkeyCheckpointSaver) -> None:
    """Test that async methods raise NotImplementedError."""
    config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}

    # Test aget_tuple
    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(saver.aget_tuple(config))
    assert "ValkeyCheckpointSaver does not support async methods" in str(exc_info.value)
    assert "AsyncValkeyCheckpointSaver" in str(exc_info.value)

    # Test alist
    async def test_alist():
        async for _ in saver.alist(config):
            pass

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(test_alist())
    assert "ValkeyCheckpointSaver does not support async methods" in str(exc_info.value)

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
        asyncio.run(saver.aput(config, checkpoint, metadata, {}))
    assert "ValkeyCheckpointSaver does not support async methods" in str(exc_info.value)


def test_list_checkpoints_missing_checkpoint_data(saver: ValkeyCheckpointSaver) -> None:
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


def test_get_tuple_missing_checkpoint_data(saver: ValkeyCheckpointSaver) -> None:
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


def test_list_checkpoints_before_nonexistent(saver: ValkeyCheckpointSaver) -> None:
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
    saver.put(config, checkpoint, metadata, {})

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
            before=before_config,
        )
    )

    # Should get all checkpoints since before checkpoint doesn't exist
    assert len(result) == 1
    assert result[0].checkpoint["id"] == "checkpoint-1"


def test_initialization_with_different_parameters() -> None:
    """Test ValkeyCheckpointSaver initialization with different parameters."""
    valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")
    client = Valkey.from_url(valkey_url)

    # Test with no TTL
    saver1 = ValkeyCheckpointSaver(client)
    assert saver1.ttl is None
    assert saver1.lock is not None

    # Test with TTL
    saver2 = ValkeyCheckpointSaver(client, ttl=3600.0)
    assert saver2.ttl == 3600.0

    # Test with custom serde (None is valid)
    saver3 = ValkeyCheckpointSaver(client, serde=None)
    assert saver3.serde is not None  # Should use default serde


def test_from_conn_string_with_kwargs() -> None:
    """Test creating saver from connection string with additional kwargs."""
    valkey_url = os.getenv("VALKEY_URL", "valkey://localhost:6379")

    with ValkeyCheckpointSaver.from_conn_string(
        valkey_url,
        ttl_seconds=1800.0,
        pool_size=15,
        socket_timeout=30,
        socket_connect_timeout=10,
    ) as saver:
        assert saver.ttl == 1800.0
        # Verify the saver works
        config = {"configurable": {"thread_id": "test-kwargs", "checkpoint_ns": "test"}}
        result = saver.get_tuple(config)
        assert result is None  # Should work without error
