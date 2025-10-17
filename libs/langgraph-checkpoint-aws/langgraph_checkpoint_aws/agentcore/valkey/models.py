"""
Models for AgentCore Valkey integration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import CheckpointerConfig


class ValkeyCheckpointerConfig(CheckpointerConfig):
    """Configuration for AgentCore Valkey checkpointer.

    Extends CheckpointerConfig to include Valkey-specific settings.
    """

    @property
    def session_key(self) -> str:
        """Generate session key for Valkey storage."""
        return f"agentcore:session:{self.session_id}:{self.actor_id}"

    @property
    def checkpoint_key_prefix(self) -> str:
        """Generate checkpoint key prefix for Valkey storage."""
        return (
            f"agentcore:checkpoint:{self.session_id}:{self.actor_id}:"
            f"{self.checkpoint_ns}"
        )

    @property
    def writes_key_prefix(self) -> str:
        """Generate writes key prefix for Valkey storage."""
        return (
            f"agentcore:writes:{self.session_id}:{self.actor_id}:{self.checkpoint_ns}"
        )

    @property
    def channel_key_prefix(self) -> str:
        """Generate channel key prefix for Valkey storage."""
        return (
            f"agentcore:channel:{self.session_id}:{self.actor_id}:{self.checkpoint_ns}"
        )


class StoredCheckpoint(BaseModel):
    """Represents a checkpoint stored in Valkey."""

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    session_id: str = Field(..., description="Session identifier")
    actor_id: str = Field(..., description="Actor identifier")
    thread_id: str = Field(..., description="Thread identifier")
    checkpoint_ns: str = Field("", description="Checkpoint namespace")
    parent_checkpoint_id: str | None = Field(None, description="Parent checkpoint ID")

    checkpoint_data: dict[str, Any] = Field(
        ..., description="Serialized checkpoint data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Checkpoint metadata"
    )

    created_at: float = Field(..., description="Timestamp when checkpoint was created")

    model_config = ConfigDict(frozen=True)


class StoredWrite(BaseModel):
    """Represents a write operation stored in Valkey."""

    checkpoint_id: str = Field(..., description="Associated checkpoint ID")
    task_id: str = Field(..., description="Task identifier")
    channel: str = Field(..., description="Channel name")
    value: Any = Field(..., description="Serialized write value")
    task_path: str = Field("", description="Task path")

    created_at: float = Field(..., description="Timestamp when write was created")

    model_config = ConfigDict(frozen=True)


class StoredChannelData(BaseModel):
    """Represents channel data stored in Valkey."""

    channel: str = Field(..., description="Channel name")
    version: str = Field(..., description="Channel version")
    value: Any = Field(..., description="Serialized channel value")
    checkpoint_id: str = Field(..., description="Associated checkpoint ID")

    created_at: float = Field(
        ..., description="Timestamp when channel data was created"
    )

    model_config = ConfigDict(frozen=True)
