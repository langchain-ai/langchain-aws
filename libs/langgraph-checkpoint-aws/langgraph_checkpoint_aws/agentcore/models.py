"""
Data models for AgentCore Memory Checkpoint Saver.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CheckpointerConfig(BaseModel):
    """Configuration for checkpoint operations."""

    thread_id: str
    actor_id: str
    checkpoint_ns: str = ""
    checkpoint_id: Optional[str] = None

    @property
    def session_id(self) -> str:
        """Generate session ID from thread_id and checkpoint_ns."""
        if self.checkpoint_ns:
            # Use underscore separator to ensure valid session ID pattern
            return f"{self.thread_id}_{self.checkpoint_ns}"
        return self.thread_id

    @classmethod
    def from_runnable_config(cls, config: Dict[str, Any]) -> "CheckpointerConfig":
        """Create CheckpointerConfig from RunnableConfig."""
        from .constants import InvalidConfigError

        configurable = config.get("configurable", {})

        if not configurable.get("thread_id"):
            raise InvalidConfigError(
                "RunnableConfig must contain 'thread_id' for AgentCore Checkpointer"
            )

        if not configurable.get("actor_id"):
            raise InvalidConfigError(
                "RunnableConfig must contain 'actor_id' for AgentCore Checkpointer"
            )

        return cls(
            thread_id=configurable["thread_id"],
            actor_id=configurable["actor_id"],
            checkpoint_ns=configurable.get("checkpoint_ns", ""),
            checkpoint_id=configurable.get("checkpoint_id"),
        )


class WriteItem(BaseModel):
    """Individual write operation."""

    task_id: str
    channel: str
    value: Any
    task_path: str = ""


class CheckpointEvent(BaseModel):
    """Event representing a checkpoint."""

    event_type: str = Field(default="checkpoint")
    checkpoint_id: str
    checkpoint_data: Dict[str, Any]
    metadata: Dict[str, Any]
    parent_checkpoint_id: Optional[str] = None
    thread_id: str
    checkpoint_ns: str = ""


class ChannelDataEvent(BaseModel):
    """Event representing channel data."""

    event_type: str = Field(default="channel_data")
    channel: str
    version: str
    value: Any
    thread_id: str
    checkpoint_ns: str = ""


class WritesEvent(BaseModel):
    """Event representing pending writes."""

    event_type: str = Field(default="writes")
    checkpoint_id: str
    writes: List[WriteItem]
