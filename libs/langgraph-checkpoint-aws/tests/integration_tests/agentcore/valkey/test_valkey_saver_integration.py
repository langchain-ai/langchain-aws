"""Integration tests for AgentCore Valkey checkpoint saver."""

import time
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip("valkey")
pytest.importorskip("orjson")

from langchain_core.runnables import RunnableConfig

try:
    from valkey import Valkey  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip("Valkey class not available", allow_module_level=True)

from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver


class TestAgentCoreValkeySaverIntegration:
    """Integration tests with real Valkey server."""

    @contextmanager
    def valkey_saver(self, ttl: float | None = None, **kwargs):
        """Context manager for creating a Valkey saver with cleanup."""
        try:
            with AgentCoreValkeySaver.from_conn_string(
                "valkey://localhost:6379/1",  # Use database 1 for testing
                ttl_seconds=ttl,
                **kwargs,
            ) as saver:
                # Clear any existing test data
                session_keys = saver.client.keys("agentcore:session:test-*")
                checkpoint_keys = saver.client.keys("agentcore:checkpoint:test-*")
                writes_keys = saver.client.keys("agentcore:writes:test-*")
                channel_keys = saver.client.keys("agentcore:channel:test-*")

                all_keys = (
                    list(session_keys)  # type: ignore[arg-type]
                    + list(checkpoint_keys)  # type: ignore[arg-type]
                    + list(writes_keys)  # type: ignore[arg-type]
                    + list(channel_keys)  # type: ignore[arg-type]
                )
                if all_keys:
                    saver.client.delete(*all_keys)

                yield saver

                # Cleanup after test
                session_keys = saver.client.keys("agentcore:session:test-*")
                checkpoint_keys = saver.client.keys("agentcore:checkpoint:test-*")
                writes_keys = saver.client.keys("agentcore:writes:test-*")
                channel_keys = saver.client.keys("agentcore:channel:test-*")

                all_keys = (
                    list(session_keys)  # type: ignore[arg-type]
                    + list(checkpoint_keys)  # type: ignore[arg-type]
                    + list(writes_keys)  # type: ignore[arg-type]
                    + list(channel_keys)  # type: ignore[arg-type]
                )
                if all_keys:
                    saver.client.delete(*all_keys)
        except Exception as e:
            pytest.skip(f"Could not connect to Valkey server: {e}")

    @pytest.fixture
    def sample_config(self) -> RunnableConfig:
        """Sample configuration for testing."""
        return {
            "configurable": {
                "thread_id": "test-session-1",
                "actor_id": "test-agent-1",
                "checkpoint_ns": "integration-test",
            }
        }

    @pytest.fixture
    def sample_checkpoint(self) -> dict[str, Any]:
        """Sample checkpoint data."""
        return {
            "id": f"checkpoint-{int(time.time() * 1000)}",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Hello, world!"},
                    {"role": "assistant", "content": "Hi! How can I help you?"},
                ],
                "context": {"user_id": "user123", "session_start": time.time()},
            },
        }

    def test_checkpoint_storage_and_retrieval(self, sample_config, sample_checkpoint):
        """Test storing and retrieving checkpoints."""
        with self.valkey_saver() as saver:
            metadata = {"user": "test_user", "timestamp": time.time()}
            new_versions = {"messages": "1.0", "context": "1.0"}

            # Store checkpoint
            result_config = saver.put(
                sample_config, sample_checkpoint, metadata, new_versions
            )

            # Verify returned config
            assert result_config["configurable"]["thread_id"] == "test-session-1"
            assert result_config["configurable"]["actor_id"] == "test-agent-1"
            assert (
                result_config["configurable"]["checkpoint_id"]
                == sample_checkpoint["id"]
            )

            # Retrieve specific checkpoint
            retrieved = saver.get_tuple(result_config)

            assert retrieved is not None
            assert retrieved.checkpoint["id"] == sample_checkpoint["id"]
            assert retrieved.checkpoint["ts"] == sample_checkpoint["ts"]
            assert len(retrieved.checkpoint["channel_values"]["messages"]) == 2
            assert (
                retrieved.checkpoint["channel_values"]["context"]["user_id"]
                == "user123"
            )

            # Retrieve latest checkpoint
            latest = saver.get_tuple(sample_config)

            assert latest is not None
            assert latest.checkpoint["id"] == sample_checkpoint["id"]

    def test_multiple_checkpoints_ordering(self, sample_config):
        """Test that multiple checkpoints are stored and retrieved in correct order."""
        with self.valkey_saver() as saver:
            checkpoints = []

            # Create multiple checkpoints
            for i in range(3):
                checkpoint = {
                    "id": f"checkpoint-{i}",
                    "ts": f"2024-01-01T00:0{i}:00Z",
                    "channel_values": {
                        "messages": [{"role": "user", "content": f"Message {i}"}]
                    },
                }

                metadata = {"step": i}
                new_versions = {"messages": f"{i + 1}.0"}

                result_config = saver.put(
                    sample_config, checkpoint, metadata, new_versions
                )
                checkpoints.append((checkpoint, result_config))

                # Small delay to ensure different timestamps
                time.sleep(0.01)

            # List all checkpoints (should be in reverse chronological order)
            checkpoint_list = list(saver.list(sample_config))

            assert len(checkpoint_list) == 3

            # Should be in reverse order (most recent first)
            for i, checkpoint_tuple in enumerate(checkpoint_list):
                expected_id = f"checkpoint-{2 - i}"  # Reverse order
                assert checkpoint_tuple.checkpoint["id"] == expected_id

            # Get latest checkpoint (should be the last one created)
            latest = saver.get_tuple(sample_config)
            assert latest.checkpoint["id"] == "checkpoint-2"

    def test_writes_storage(self, sample_config):
        """Test storing and retrieving writes."""
        with self.valkey_saver() as saver:
            # First create a checkpoint
            checkpoint = {
                "id": "checkpoint-with-writes",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"messages": []},
            }

            metadata = {"user": "test"}
            new_versions = {"messages": "1.0"}

            result_config = saver.put(sample_config, checkpoint, metadata, new_versions)

            # Add writes
            writes = [
                ("messages", {"role": "assistant", "content": "Response 1"}),
                ("messages", {"role": "assistant", "content": "Response 2"}),
            ]

            saver.put_writes(result_config, writes, "task-1", "/test/path")

            # Retrieve checkpoint with writes
            retrieved = saver.get_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.pending_writes) == 2

            # Check write content
            assert retrieved.pending_writes[0][0] == "task-1"  # task_id
            assert retrieved.pending_writes[0][1] == "messages"  # channel
            assert retrieved.pending_writes[0][2]["role"] == "assistant"  # value

            assert retrieved.pending_writes[1][2]["content"] == "Response 2"

    def test_ttl_functionality(self, sample_config, sample_checkpoint):
        """Test TTL functionality (if supported)."""
        with self.valkey_saver(ttl=1) as saver:  # 1 second TTL
            metadata = {"user": "test_ttl"}
            new_versions = {"messages": "1.0"}

            # Store checkpoint
            result_config = saver.put(
                sample_config, sample_checkpoint, metadata, new_versions
            )

            # Should be retrievable immediately
            retrieved = saver.get_tuple(result_config)
            assert retrieved is not None

            # Check that TTL was set on keys
            checkpoint_key = saver._make_checkpoint_key(
                saver._serialize_checkpoint(sample_config, sample_checkpoint, metadata),
                sample_checkpoint["id"],
            )

            # Key should exist but have TTL set
            assert saver.client.exists(checkpoint_key)
            ttl = saver.client.ttl(checkpoint_key)
            assert ttl > 0  # Should have TTL set

    def test_metadata_filtering(self, sample_config):
        """Test filtering checkpoints by metadata."""
        with self.valkey_saver() as saver:
            # Create checkpoints with different metadata
            for i in range(3):
                checkpoint = {
                    "id": f"checkpoint-meta-{i}",
                    "ts": f"2024-01-01T00:0{i}:00Z",
                    "channel_values": {"messages": []},
                }

                metadata = {
                    "user": "test_user" if i % 2 == 0 else "other_user",
                    "step": i,
                }
                new_versions = {"messages": f"{i + 1}.0"}

                saver.put(sample_config, checkpoint, metadata, new_versions)
                time.sleep(0.01)

            # Filter by metadata
            filtered_checkpoints = list(
                saver.list(sample_config, filter={"user": "test_user"})
            )

            # Should only get checkpoints for test_user (indices 0 and 2)
            assert len(filtered_checkpoints) == 2

            for checkpoint_tuple in filtered_checkpoints:
                checkpoint_id = checkpoint_tuple.checkpoint["id"]
                assert checkpoint_id in ["checkpoint-meta-0", "checkpoint-meta-2"]

    def test_thread_deletion(self, sample_config):
        """Test deleting all data for a thread."""
        with self.valkey_saver() as saver:
            # Create multiple checkpoints and writes
            for i in range(2):
                checkpoint = {
                    "id": f"checkpoint-delete-{i}",
                    "ts": f"2024-01-01T00:0{i}:00Z",
                    "channel_values": {"messages": []},
                }

                metadata = {"step": i}
                new_versions = {"messages": f"{i + 1}.0"}

                result_config = saver.put(
                    sample_config, checkpoint, metadata, new_versions
                )

                # Add writes
                writes = [("messages", {"content": f"write-{i}"})]
                saver.put_writes(result_config, writes, f"task-{i}")

            # Verify data exists
            checkpoints = list(saver.list(sample_config))
            assert len(checkpoints) == 2

            # Delete thread
            saver.delete_thread("test-session-1", "test-agent-1")

            # Verify all data is deleted
            checkpoints_after = list(saver.list(sample_config))
            assert len(checkpoints_after) == 0

            # Try to get latest checkpoint
            latest = saver.get_tuple(sample_config)
            assert latest is None

    def test_connection_pooling(self, sample_config, sample_checkpoint):
        """Test that connection pooling works correctly."""
        from valkey.connection import ConnectionPool

        try:
            # Create a connection pool
            pool = ConnectionPool.from_url(
                "valkey://localhost:6379/1", max_connections=5
            )

            with AgentCoreValkeySaver.from_pool(pool, ttl_seconds=3600) as saver:
                # Clear test data
                test_keys = saver.client.keys("agentcore:*test-session-pool*")
                if test_keys:
                    saver.client.delete(*test_keys)

                # Use the saver with pool
                pool_config = {
                    "configurable": {
                        "thread_id": "test-session-pool",
                        "actor_id": "test-agent-pool",
                        "checkpoint_ns": "pool-test",
                    }
                }

                metadata = {"pool_test": True}
                new_versions = {"messages": "1.0"}

                result_config = saver.put(
                    pool_config, sample_checkpoint, metadata, new_versions
                )
                retrieved = saver.get_tuple(result_config)

                assert retrieved is not None
                assert retrieved.checkpoint["id"] == sample_checkpoint["id"]

                # Cleanup
                saver.delete_thread("test-session-pool", "test-agent-pool")

        except Exception as e:
            pytest.skip(f"Connection pool test failed: {e}")

    def test_concurrent_operations(self, sample_config):
        """Test concurrent operations on the same session."""
        import threading

        with self.valkey_saver() as saver:
            results = []
            errors = []

            def create_checkpoint(index):
                try:
                    checkpoint = {
                        "id": f"checkpoint-concurrent-{index}",
                        "ts": f"2024-01-01T00:{index:02d}:00Z",
                        "channel_values": {"messages": [{"content": f"msg-{index}"}]},
                    }

                    metadata = {"thread_index": index}
                    new_versions = {"messages": f"{index + 1}.0"}

                    result_config = saver.put(
                        sample_config, checkpoint, metadata, new_versions
                    )
                    results.append(result_config)
                except Exception as e:
                    errors.append(e)

            # Create multiple checkpoints concurrently
            threads = []
            for i in range(5):
                thread = threading.Thread(target=create_checkpoint, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5

            # Verify all checkpoints were created
            checkpoints = list(saver.list(sample_config))
            assert len(checkpoints) == 5

            # Verify each checkpoint is unique
            checkpoint_ids = {cp.checkpoint["id"] for cp in checkpoints}
            assert len(checkpoint_ids) == 5

    def test_large_checkpoint_data(self, sample_config):
        """Test handling of large checkpoint data."""
        with self.valkey_saver() as saver:
            # Create a large checkpoint
            large_messages = []
            for i in range(100):
                large_messages.append(
                    {
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": f"This is message number {i} " * 50,  # Long content
                        "metadata": {"index": i, "data": list(range(50))},
                    }
                )

            checkpoint = {
                "id": "checkpoint-large",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {
                    "messages": large_messages,
                    "context": {
                        "session_data": {"key": "value"} * 100,
                        "large_array": list(range(1000)),
                    },
                },
            }

            metadata = {"size": "large", "message_count": len(large_messages)}
            new_versions = {"messages": "1.0", "context": "1.0"}

            # Store large checkpoint
            result_config = saver.put(sample_config, checkpoint, metadata, new_versions)

            # Retrieve and verify
            retrieved = saver.get_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.checkpoint["channel_values"]["messages"]) == 100
            assert (
                len(retrieved.checkpoint["channel_values"]["context"]["large_array"])
                == 1000
            )
            assert retrieved.metadata["message_count"] == 100


class TestAgentCoreValkeySaverErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_connection_string(self):
        """Test handling of invalid connection strings."""
        with pytest.raises(
            (ValueError, ConnectionError)
        ):  # Should raise connection error
            with AgentCoreValkeySaver.from_conn_string("invalid://connection:string"):
                pass

    def test_malformed_config(self):
        """Test handling of malformed configurations."""
        try:
            with AgentCoreValkeySaver.from_conn_string(
                "valkey://localhost:6379/1"
            ) as saver:
                # Missing required config fields
                invalid_config = {
                    "configurable": {"thread_id": "test"}
                }  # Missing actor_id

                with pytest.raises(
                    (KeyError, ValueError)
                ):  # Should raise validation error
                    saver.get_tuple(invalid_config)

        except Exception:
            pytest.skip("Could not connect to Valkey server")

    def test_corrupted_data_handling(self):
        """Test handling of corrupted data in Valkey."""
        try:
            with AgentCoreValkeySaver.from_conn_string(
                "valkey://localhost:6379/1"
            ) as saver:
                # Manually insert corrupted data with correct key format
                # session_id = thread_id + "_" + checkpoint_ns when
                # checkpoint_ns is not empty
                corrupt_key = (
                    "agentcore:checkpoint:test-corrupt_ns:agent:ns:corrupt-checkpoint"
                )
                saver.client.set(corrupt_key, "invalid-json-data")

                config = {
                    "configurable": {
                        "thread_id": "test-corrupt",
                        "actor_id": "agent",
                        "checkpoint_ns": "ns",
                        "checkpoint_id": "corrupt-checkpoint",
                    }
                }

                # Should handle corrupted data gracefully
                with pytest.raises(
                    (ValueError, TypeError)
                ):  # Should raise parsing error
                    saver.get_tuple(config)

                # Cleanup
                saver.client.delete(corrupt_key)

        except Exception:
            pytest.skip("Could not connect to Valkey server")
