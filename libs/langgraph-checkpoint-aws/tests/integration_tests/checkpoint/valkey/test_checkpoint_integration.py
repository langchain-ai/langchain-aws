"""Tests for the ValkeyCheckpointSaver implementation."""

import os
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from valkey import Valkey
from valkey.connection import ConnectionPool

from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey server URL from environment or use default."""
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest.fixture
def valkey_pool(valkey_url: str) -> Generator[ConnectionPool, None, None]:
    """Create a ValkeyPool instance."""
    pool = ConnectionPool.from_url(valkey_url, max_connections=5)
    yield pool
    # Pool cleanup will be automatic


@pytest.fixture
def saver(valkey_url: str) -> ValkeyCheckpointSaver:
    """Create a ValkeyCheckpointSaver instance."""
    return ValkeyCheckpointSaver(Valkey(valkey_url), ttl=60.0)


def test_from_conn_string(valkey_url: str) -> None:
    """Test creating saver from connection string."""
    with ValkeyCheckpointSaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        assert saver.ttl == 3600  # 3600 seconds


def test_from_pool(valkey_pool: ConnectionPool) -> None:
    """Test creating saver from existing pool."""
    with ValkeyCheckpointSaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver:
        assert saver.ttl == 3600


def test_sync_operations(valkey_url: str) -> None:
    """Test sync operations using connection pool."""
    with ValkeyCheckpointSaver.from_conn_string(
        valkey_url, ttl_seconds=3600.0, pool_size=5
    ) as saver:
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        new_versions = {}

        # Store checkpoint
        result = saver.put(config, checkpoint, metadata, new_versions)
        assert result["configurable"]["checkpoint_id"] == checkpoint["id"]

        # Get checkpoint
        result = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread",
                    "checkpoint_ns": "test",
                    "checkpoint_id": checkpoint["id"],
                }
            }
        )
        assert result is not None
        assert result.checkpoint["id"] == checkpoint["id"]
        assert result.checkpoint["state"] == checkpoint["state"]
        assert result.metadata["user"] == metadata["user"]


def test_sync_shared_pool(valkey_pool: ConnectionPool) -> None:
    """Test sharing connection pool between savers."""
    with (
        ValkeyCheckpointSaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver1,
        ValkeyCheckpointSaver.from_pool(valkey_pool, ttl_seconds=3600.0) as saver2,
    ):
        # Test data
        config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": "test"}}
        checkpoint1 = {"id": "test-1", "state": {"value": 1}, "versions": {}}
        checkpoint2 = {"id": "test-2", "state": {"value": 2}, "versions": {}}
        metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "user": "test"}
        new_versions = {}

        # Store checkpoints in both savers
        saver1.put(config, checkpoint1, metadata, new_versions)
        saver2.put(config, checkpoint2, metadata, new_versions)

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
        assert result1.checkpoint["id"] == checkpoint1["id"]
        assert result2.checkpoint["id"] == checkpoint2["id"]
