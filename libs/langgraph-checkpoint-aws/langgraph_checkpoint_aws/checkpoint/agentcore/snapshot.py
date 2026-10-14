from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from .constants import EventDecodingError, EventNotFoundError
from .helpers import (
    MAX_PAYLOAD_BYTES_PER_EVENT,
    MAX_PAYLOAD_ITEMS_PER_EVENT,
    AgentCoreEventClient,
    EventType,
)
from .models import ChannelDataEvent, CheckpointEvent, WriteItem, WritesEvent

SNAPSHOT_VERSION = 1
SNAPSHOT_LAYOUT = "snapshot-v1"
SNAPSHOT_PAYLOAD_BYTES = MAX_PAYLOAD_BYTES_PER_EVENT - 4096
logger = logging.getLogger(__name__)


def _checkpoint_key(checkpoint_id: str) -> str:
    """Encode arbitrary checkpoint IDs within AgentCore metadata constraints."""
    return hashlib.sha256(checkpoint_id.encode()).hexdigest()


def _writes_session_prefix(thread_id: str) -> str:
    return f"lgw-{hashlib.sha256(thread_id.encode()).hexdigest()[:32]}-"


def writes_session_id(
    thread_id: str, checkpoint_id: str, checkpoint_ns: str = ""
) -> str:
    """Return a deterministic session ID for one checkpoint's pending writes."""
    key = hashlib.sha256(f"{checkpoint_ns}\x1f{checkpoint_id}".encode()).hexdigest()
    return _writes_session_prefix(thread_id) + key[:32]


def _checkpoint_timestamp(checkpoint: CheckpointEvent) -> datetime.datetime:
    """Event timestamp for a snapshot: the checkpoint's own creation time."""
    ts = checkpoint.checkpoint_data.get("ts")
    if isinstance(ts, str):
        try:
            parsed = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed
    return datetime.datetime.now(datetime.timezone.utc)


def _metadata_equals(key: str, value: str) -> dict[str, Any]:
    return {
        "left": {"metadataKey": key},
        "operator": "EQUALS_TO",
        "right": {"metadataValue": {"stringValue": value}},
    }


class AgentCoreSnapshotClient(AgentCoreEventClient):
    """Store snapshots with direct references to any additional payload chunks."""

    def _metadata(self, checkpoint_id: str, kind: str) -> dict[str, Any]:
        return {
            "lc_layout": {"stringValue": SNAPSHOT_LAYOUT},
            "lc_kind": {"stringValue": kind},
            "lc_checkpoint_id": {"stringValue": _checkpoint_key(checkpoint_id)},
        }

    def _create_snapshot_event(
        self,
        payload: list[str],
        session_id: str,
        actor_id: str,
        checkpoint: CheckpointEvent,
        kind: str,
        *,
        offset_ms: int = 0,
    ) -> dict[str, Any]:
        digest = hashlib.sha256()
        for part in (session_id, actor_id, checkpoint.checkpoint_id, kind, *payload):
            digest.update(part.encode())
            digest.update(b"\x1f")
        timestamp = _checkpoint_timestamp(checkpoint) - datetime.timedelta(
            milliseconds=offset_ms
        )
        return self.client.create_event(
            memoryId=self.memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=timestamp,
            clientToken=str(uuid5(NAMESPACE_OID, digest.hexdigest())),
            metadata=self._metadata(checkpoint.checkpoint_id, kind),
            payload=[json.loads(item) for item in payload],
        )

    def _blob_payload(self, event: EventType) -> str:
        return json.dumps({"blob": self.serializer.serialize_event(event)})

    def store_snapshot(
        self,
        events: list[EventType],
        session_id: str,
        actor_id: str,
    ) -> None:
        """Persist a snapshot, publishing its checkpoint after all other chunks."""
        checkpoint = next(e for e in events if isinstance(e, CheckpointEvent))
        checkpoint = checkpoint.model_copy(
            update={"snapshot_version": SNAPSHOT_VERSION, "chunk_event_ids": []}
        )
        payload = [
            self._blob_payload(event)
            for event in events
            if not isinstance(event, CheckpointEvent)
        ]
        payload.append(self._blob_payload(checkpoint))
        if any(len(item.encode()) > SNAPSHOT_PAYLOAD_BYTES for item in payload):
            msg = "A snapshot payload item exceeds the AgentCore event size limit."
            raise ValueError(msg)
        chunks = self._chunk_payload(
            payload, MAX_PAYLOAD_ITEMS_PER_EVENT, SNAPSHOT_PAYLOAD_BYTES
        )
        event_ids = []
        for index, chunk in enumerate(chunks[:-1]):
            response = self._create_snapshot_event(
                chunk,
                session_id,
                actor_id,
                checkpoint,
                "checkpoint_data",
                offset_ms=len(chunks) + 1 - index,
            )
            event_ids.append(response["event"]["eventId"])

        final_chunk = chunks[-1]
        checkpoint.chunk_event_ids = event_ids
        final_chunk[-1] = self._blob_payload(checkpoint)
        if sum(len(item.encode()) for item in final_chunk) > SNAPSHOT_PAYLOAD_BYTES:
            if len(final_chunk) > 1:
                response = self._create_snapshot_event(
                    final_chunk[:-1],
                    session_id,
                    actor_id,
                    checkpoint,
                    "checkpoint_data",
                    offset_ms=1,
                )
                checkpoint.chunk_event_ids.append(response["event"]["eventId"])
            final_chunk = [self._blob_payload(checkpoint)]
            if len(final_chunk[0].encode()) > SNAPSHOT_PAYLOAD_BYTES:
                msg = "The snapshot's checkpoint record exceeds the event size limit."
                raise ValueError(msg)
        self._create_snapshot_event(
            final_chunk, session_id, actor_id, checkpoint, "checkpoint"
        )

    def _decode_payload(self, event: dict[str, Any]) -> list[EventType]:
        return [
            self.serializer.deserialize_event(item["blob"])
            for item in event.get("payload", [])
            if "blob" in item
        ]

    def _load_snapshot(
        self, event: dict[str, Any], session_id: str, actor_id: str
    ) -> list[EventType]:
        events = self._decode_payload(event)
        checkpoints = [e for e in events if isinstance(e, CheckpointEvent)]
        if len(checkpoints) != 1 or checkpoints[0].snapshot_version != SNAPSHOT_VERSION:
            msg = "Invalid or unsupported AgentCore checkpoint snapshot."
            raise EventDecodingError(msg)
        checkpoint = checkpoints[0]
        expected_metadata = self._metadata(checkpoint.checkpoint_id, "checkpoint")
        if any(
            event.get("metadata", {}).get(key) != value
            for key, value in expected_metadata.items()
        ):
            msg = "Checkpoint metadata does not match the snapshot payload."
            raise EventDecodingError(msg)
        expected_metadata = self._metadata(checkpoint.checkpoint_id, "checkpoint_data")
        for event_id in checkpoint.chunk_event_ids:
            response = self.client.get_event(
                memoryId=self.memory_id,
                actorId=actor_id,
                sessionId=session_id,
                eventId=event_id,
            )
            chunk = response["event"]
            if any(
                chunk.get("metadata", {}).get(key) != value
                for key, value in expected_metadata.items()
            ):
                msg = f"Unexpected metadata for checkpoint chunk {event_id}."
                raise EventDecodingError(msg)
            chunk_events = self._decode_payload(chunk)
            if any(isinstance(item, CheckpointEvent) for item in chunk_events):
                msg = f"Unexpected checkpoint in data chunk {event_id}."
                raise EventDecodingError(msg)
            events.extend(chunk_events)
        expected_channels = {
            (channel, str(version))
            for channel, version in checkpoint.checkpoint_data.get(
                "channel_versions", {}
            ).items()
        }
        actual_channels = {
            (e.channel, e.version) for e in events if isinstance(e, ChannelDataEvent)
        }
        if expected_channels - actual_channels:
            msg = f"Missing channel data for checkpoint {checkpoint.checkpoint_id}."
            raise EventNotFoundError(msg)
        return events

    def get_snapshot(
        self,
        session_id: str,
        actor_id: str,
        checkpoint_id: str | None,
        max_results: int | None,
    ) -> list[EventType] | None:
        """Find a snapshot, or return None to request a legacy session scan."""
        params: dict[str, Any] = {
            "memoryId": self.memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "includePayloads": True,
            "maxResults": (max_results or 100) if checkpoint_id else 1,
        }
        commit_filter = _metadata_equals("lc_kind", "checkpoint")
        if checkpoint_id:
            params["filter"] = {
                "eventMetadata": [
                    _metadata_equals(
                        "lc_checkpoint_id", _checkpoint_key(checkpoint_id)
                    ),
                    commit_filter,
                ]
            }
        legacy_writes: list[EventType] = []
        legacy_blobs_seen = False
        while True:
            response = self.client.list_events(**params)
            for event in response.get("events", []):
                metadata = event.get("metadata", {})
                if metadata.get("lc_layout", {}).get("stringValue") != SNAPSHOT_LAYOUT:
                    for item in event.get("payload", []):
                        if "blob" not in item:
                            continue
                        legacy_blobs_seen = True
                        try:
                            decoded = self.serializer.deserialize_event(item["blob"])
                        except EventDecodingError as exc:
                            logger.warning("Failed to decode event: %s", exc)
                            continue
                        if isinstance(decoded, CheckpointEvent):
                            return None
                        if isinstance(decoded, WritesEvent):
                            legacy_writes.append(decoded)
                    continue
                if metadata.get("lc_kind", {}).get("stringValue") == "checkpoint":
                    return legacy_writes + self._load_snapshot(
                        event, session_id, actor_id
                    )
            if not response.get("nextToken"):
                if checkpoint_id or "filter" in params:
                    return None
                return []
            if checkpoint_id:
                params["nextToken"] = response["nextToken"]
                params["maxResults"] = max_results or 100
            elif "filter" not in params and not legacy_blobs_seen:
                params["filter"] = {"eventMetadata": [commit_filter]}
                params["maxResults"] = max_results or 100
            else:
                params["nextToken"] = response["nextToken"]

    def get_pending_writes(
        self,
        thread_id: str,
        actor_id: str,
        checkpoint_id: str,
        max_results: int | None = 100,
        *,
        checkpoint_ns: str = "",
    ) -> list[WriteItem]:
        """Read all pending writes for a checkpoint, regardless of write order.

        Args:
            thread_id: Thread that owns the checkpoint.
            actor_id: Actor owning the session.
            checkpoint_id: Checkpoint whose pending writes are being retrieved.
            max_results: Service page size, or None to use the service default.
            checkpoint_ns: Checkpoint namespace (empty for the root graph).

        Returns:
            All writes stored in the checkpoint's pending-write session.

        Raises:
            EventDecodingError: A pending-write event is invalid.
        """
        params: dict[str, Any] = {
            "memoryId": self.memory_id,
            "sessionId": writes_session_id(thread_id, checkpoint_id, checkpoint_ns),
            "actorId": actor_id,
            "includePayloads": True,
        }
        if max_results is not None:
            params["maxResults"] = max_results
        writes: list[WriteItem] = []
        while True:
            response = self.client.list_events(**params)
            for raw_event in response.get("events", []):
                for event in self._decode_payload(raw_event):
                    if not isinstance(event, WritesEvent):
                        msg = (
                            "Unexpected event in a checkpoint's pending-write session."
                        )
                        raise EventDecodingError(msg)
                    if event.checkpoint_id != checkpoint_id:
                        msg = "Pending writes belong to a different checkpoint."
                        raise EventDecodingError(msg)
                    writes.extend(event.writes)
            if not response.get("nextToken"):
                return writes
            params["nextToken"] = response["nextToken"]

    def delete_events(self, session_id: str, actor_id: str) -> None:
        """Delete the thread and its pending-write sessions, including orphans.

        Args:
            session_id: Thread ID of the root checkpoint session.
            actor_id: Actor owning the session.
        """
        prefix = _writes_session_prefix(session_id)
        params: dict[str, Any] = {
            "memoryId": self.memory_id,
            "actorId": actor_id,
            "maxResults": 100,
        }
        sessions: list[str] = []
        while True:
            response = self.client.list_sessions(**params)
            sessions.extend(
                item["sessionId"]
                for item in response.get("sessionSummaries", [])
                if item["sessionId"].startswith(prefix)
            )
            if not response.get("nextToken"):
                break
            params["nextToken"] = response["nextToken"]
        for writes_session in sessions:
            super().delete_events(writes_session, actor_id)
        super().delete_events(session_id, actor_id)
