"""Base class for Valkey checkpoint savers."""

from __future__ import annotations

import base64
import json
import random
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_metadata,
)

from .utils import set_client_info


class BaseValkeySaver(BaseCheckpointSaver[str]):
    """Base class for Valkey checkpoint savers.

    This class contains common functionality shared between synchronous and
    asynchronous Valkey checkpoint savers, including key generation, serialization,
    and deserialization logic.

    Args:
        client: The Valkey client instance (sync or async).
        ttl: Time-to-live for stored checkpoints in seconds. Defaults to None (no
            expiration).
        serde: The serializer to use for serializing and deserializing checkpoints.
    """

    def __init__(
        self,
        client: Any,
        *,
        ttl: float | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.client = client
        self.ttl = ttl

        # Set client info for library identification
        # Check if this is an async client by looking for async methods
        if hasattr(client, "aclose") or hasattr(client, "__aenter__"):
            # This is likely an async client, skip sync set_client_info
            # The async subclass should handle this with aset_client_info
            pass
        else:
            # This is a sync client, safe to call set_client_info
            set_client_info(client)

    def _make_checkpoint_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Generate a key for storing checkpoint data."""
        return f"checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _make_writes_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Generate a key for storing writes data."""
        return f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _make_thread_key(self, thread_id: str, checkpoint_ns: str) -> str:
        """Generate a key for storing thread checkpoint list."""
        return f"thread:{thread_id}:{checkpoint_ns}"

    def _serialize_checkpoint_data(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> dict[str, Any]:
        """Serialize checkpoint data for storage.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to serialize.
            metadata: Additional metadata to serialize.

        Returns:
            dict: Serialized checkpoint data ready for JSON storage.
        """
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint and metadata
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)

        # Use plain JSON for metadata (simple types only)
        metadata_dict = get_checkpoint_metadata(config, metadata)
        metadata_json = json.dumps(metadata_dict, ensure_ascii=False).encode(
            "utf-8", "ignore"
        )

        # Prepare checkpoint data - encode bytes as base64 for JSON serialization
        return {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": configurable.get("checkpoint_id"),
            "type": type_,
            "checkpoint": base64.b64encode(serialized_checkpoint).decode("utf-8")
            if isinstance(serialized_checkpoint, bytes)
            else serialized_checkpoint,
            "metadata": base64.b64encode(metadata_json).decode("utf-8"),
        }

    def _deserialize_checkpoint_data(
        self,
        checkpoint_info: dict[str, Any],
        writes: list[dict[str, Any]],
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        config: RunnableConfig | None = None,
    ) -> CheckpointTuple:
        """Deserialize checkpoint data from storage.

        Args:
            checkpoint_info: Raw checkpoint data from storage.
            writes: Raw writes data from storage.
            thread_id: Thread ID for the checkpoint.
            checkpoint_ns: Checkpoint namespace.
            checkpoint_id: Checkpoint ID.
            config: Optional config to use, will be generated if not provided.

        Returns:
            CheckpointTuple: Deserialized checkpoint tuple.
        """
        # Deserialize checkpoint and metadata - decode base64 if needed
        checkpoint_data = checkpoint_info["checkpoint"]
        if isinstance(checkpoint_data, str):
            checkpoint_data = base64.b64decode(checkpoint_data.encode("utf-8"))
        checkpoint = self.serde.loads_typed((checkpoint_info["type"], checkpoint_data))

        # Deserialize metadata from plain JSON
        metadata_data = checkpoint_info["metadata"]
        if isinstance(metadata_data, str):
            metadata_data = base64.b64decode(metadata_data.encode("utf-8"))
        metadata = cast(
            CheckpointMetadata,
            json.loads(metadata_data) if metadata_data else {},
        )

        # Create parent config if exists
        parent_config: RunnableConfig | None = None
        if checkpoint_info["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_info["parent_checkpoint_id"],
                }
            }

        # Deserialize writes - decode base64 if needed
        pending_writes = []
        for write in writes:
            write_value = write["value"]
            if isinstance(write_value, str):
                write_value = base64.b64decode(write_value.encode("utf-8"))
            pending_writes.append(
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed((write["type"], write_value)),
                )
            )

        # Use provided config or generate one
        if config is None:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        return CheckpointTuple(
            config,
            checkpoint,
            metadata,
            parent_config,
            pending_writes,
        )

    def _serialize_writes_data(
        self,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Serialize writes data for storage.

        Args:
            writes: List of writes to serialize, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.

        Returns:
            list: Serialized writes data ready for JSON storage.
        """
        serialized_writes = []
        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.serde.dumps_typed(value)
            write_data = {
                "task_id": task_id,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "value": base64.b64encode(serialized_value).decode("utf-8")
                if isinstance(serialized_value, bytes)
                else serialized_value,
            }
            serialized_writes.append(write_data)
        return serialized_writes

    def _should_include_checkpoint(
        self,
        checkpoint_info: dict[str, Any],
        filter: dict[str, Any] | None,
    ) -> bool:
        """Check if a checkpoint should be included based on metadata filter.

        Args:
            checkpoint_info: Raw checkpoint data from storage.
            filter: Metadata filter criteria.

        Returns:
            bool: True if checkpoint should be included, False otherwise.
        """
        if not filter:
            return True

        # Deserialize metadata from plain JSON
        metadata_data = checkpoint_info["metadata"]
        if isinstance(metadata_data, str):
            metadata_data = base64.b64decode(metadata_data.encode("utf-8"))
        metadata = json.loads(metadata_data) if metadata_data else {}

        return all(
            key in metadata and metadata[key] == value for key, value in filter.items()
        )

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its
        current version.

        Args:
            current (Optional[str]): The current version identifier of the channel.

        Returns:
            str: The next version identifier, which is guaranteed to be
                monotonically increasing.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
