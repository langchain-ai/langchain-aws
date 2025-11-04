"""Tests for AgentCore Valkey models."""

import pytest

pytest.importorskip("valkey")
pytest.importorskip("pydantic")

from pydantic import ValidationError

from langgraph_checkpoint_aws.agentcore.valkey.models import (
    StoredChannelData,
    StoredCheckpoint,
    StoredWrite,
    ValkeyCheckpointerConfig,
)


class TestValkeyCheckpointerConfig:
    """Test ValkeyCheckpointerConfig model."""

    def test_session_key(self):
        """Test session key generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns=""
        )

        expected = "agentcore:session:session-1:agent-1"
        assert config.session_key == expected

    def test_checkpoint_key_prefix(self):
        """Test checkpoint key prefix generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns="test-ns"
        )

        expected = "agentcore:checkpoint:session-1_test-ns:agent-1:test-ns"
        assert config.checkpoint_key_prefix == expected

    def test_writes_key_prefix(self):
        """Test writes key prefix generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns=""
        )

        expected = "agentcore:writes:session-1:agent-1:"
        assert config.writes_key_prefix == expected

    def test_channel_key_prefix(self):
        """Test channel key prefix generation."""
        config = ValkeyCheckpointerConfig(
            thread_id="session-1", actor_id="agent-1", checkpoint_ns="test-ns"
        )

        expected = "agentcore:channel:session-1_test-ns:agent-1:test-ns"
        assert config.channel_key_prefix == expected

    def test_from_runnable_config(self):
        """Test creating config from runnable config."""
        runnable_config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
                "checkpoint_ns": "test-ns",
                "checkpoint_id": "checkpoint-1",
            }
        }

        config = ValkeyCheckpointerConfig.from_runnable_config(runnable_config)

        assert config.session_id == "session-1_test-ns"  # session_id includes namespace
        assert config.actor_id == "agent-1"
        assert config.thread_id == "session-1"  # thread_id is the original value
        assert config.checkpoint_ns == "test-ns"
        assert config.checkpoint_id == "checkpoint-1"

    def test_from_runnable_config_no_namespace(self):
        """Test creating config from runnable config without namespace."""
        runnable_config = {
            "configurable": {
                "thread_id": "session-1",
                "actor_id": "agent-1",
            }
        }

        config = ValkeyCheckpointerConfig.from_runnable_config(runnable_config)

        assert config.session_id == "session-1"  # No namespace
        assert config.actor_id == "agent-1"
        assert config.thread_id == "session-1"
        assert config.checkpoint_ns == ""
        assert config.checkpoint_id is None


class TestStoredCheckpoint:
    """Test StoredCheckpoint model."""

    def test_valid_checkpoint(self):
        """Test creating a valid checkpoint."""
        checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            checkpoint_data={"type": "test", "data": "dGVzdA=="},
            metadata={"type": "test", "data": "dGVzdA=="},
            created_at=1234567890.0,
        )

        assert checkpoint.checkpoint_id == "checkpoint-1"
        assert checkpoint.session_id == "session-1"
        assert checkpoint.actor_id == "agent-1"
        assert checkpoint.checkpoint_ns == ""
        assert checkpoint.parent_checkpoint_id is None
        assert checkpoint.created_at == 1234567890.0

    def test_with_parent_checkpoint(self):
        """Test checkpoint with parent."""
        checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-2",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_ns="",
            parent_checkpoint_id="checkpoint-1",
            checkpoint_data={"type": "test", "data": "dGVzdA=="},
            metadata={"type": "test", "data": "dGVzdA=="},
            created_at=1234567890.0,
        )

        assert checkpoint.parent_checkpoint_id == "checkpoint-1"

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            StoredCheckpoint(
                checkpoint_id="checkpoint-1",
                # Missing required fields
            )

    def test_frozen_model(self):
        """Test that model is frozen (immutable)."""
        checkpoint = StoredCheckpoint(
            checkpoint_id="checkpoint-1",
            session_id="session-1",
            thread_id="session-1",
            actor_id="agent-1",
            checkpoint_data={"type": "test", "data": "dGVzdA=="},
            metadata={"type": "test", "data": "dGVzdA=="},
            created_at=1234567890.0,
        )

        with pytest.raises(ValidationError):
            checkpoint.checkpoint_id = "new-id"


class TestStoredWrite:
    """Test StoredWrite model."""

    def test_valid_write(self):
        """Test creating a valid write."""
        write = StoredWrite(
            checkpoint_id="checkpoint-1",
            task_id="task-1",
            channel="test_channel",
            value={"type": "test", "data": "dGVzdA=="},
            task_path="/test/path",
            created_at=1234567890.0,
        )

        assert write.checkpoint_id == "checkpoint-1"
        assert write.task_id == "task-1"
        assert write.channel == "test_channel"
        assert write.task_path == "/test/path"
        assert write.created_at == 1234567890.0

    def test_default_task_path(self):
        """Test default task path."""
        write = StoredWrite(
            checkpoint_id="checkpoint-1",
            task_id="task-1",
            channel="test_channel",
            value={"type": "test", "data": "dGVzdA=="},
            created_at=1234567890.0,
        )

        assert write.task_path == ""

    def test_frozen_model(self):
        """Test that model is frozen (immutable)."""
        write = StoredWrite(
            checkpoint_id="checkpoint-1",
            task_id="task-1",
            channel="test_channel",
            value={"type": "test", "data": "dGVzdA=="},
            created_at=1234567890.0,
        )

        with pytest.raises(ValidationError):
            write.task_id = "new-task"


class TestStoredChannelData:
    """Test StoredChannelData model."""

    def test_valid_channel_data(self):
        """Test creating valid channel data."""
        channel_data = StoredChannelData(
            channel="test_channel",
            version="1.0",
            value={"type": "test", "data": "dGVzdA=="},
            checkpoint_id="checkpoint-1",
            created_at=1234567890.0,
        )

        assert channel_data.channel == "test_channel"
        assert channel_data.version == "1.0"
        assert channel_data.checkpoint_id == "checkpoint-1"
        assert channel_data.created_at == 1234567890.0

    def test_frozen_model(self):
        """Test that model is frozen (immutable)."""
        channel_data = StoredChannelData(
            channel="test_channel",
            version="1.0",
            value={"type": "test", "data": "dGVzdA=="},
            checkpoint_id="checkpoint-1",
            created_at=1234567890.0,
        )

        with pytest.raises(ValidationError):
            channel_data.channel = "new-channel"
