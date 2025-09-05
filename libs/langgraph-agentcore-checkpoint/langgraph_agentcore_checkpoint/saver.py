import base64
import json
import random
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

import boto3
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph_agentcore_checkpoint.constants import BLOBS_BRANCH, CHECKPOINT_BRANCH


class AgentCoreMemorySaver(BaseCheckpointSaver[str]):
    """AgentCore Memory checkpoint saver using AWS Bedrock AgentCore APIs."""

    def __init__(
        self,
        memory_id: str,
        actor_id: str = "agent",
        *,
        serde: Optional[SerializerProtocol] = None,
        **boto3_kwargs: Any,
    ) -> None:
        super().__init__(serde=serde)
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.client = boto3.client("bedrock-agentcore", **boto3_kwargs)
        self._root_event_cache = {}
        self._branch_exists_cache = {}

    def _get_session_id(self, thread_id: str, checkpoint_ns: str = "") -> str:
        return f"{thread_id}#{checkpoint_ns}" if checkpoint_ns else thread_id

    def _get_root_event_id(self, session_id: str) -> str:
        """Get or create root event for the session."""
        if session_id in self._root_event_cache:
            return self._root_event_cache[session_id]

        try:
            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                actorId=self.actor_id,
                includePayloads=False,
                maxResults=1,
            )

            events = response.get("events", [])
            if events:
                root_event_id = events[0]["eventId"]
                self._root_event_cache[session_id] = root_event_id
                return root_event_id

            root_response = self.client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=session_id,
                eventTimestamp=datetime.utcnow(),
                payload=[{"blob": {"session_root": True}}],
            )
            root_event_id = root_response["event"]["eventId"]
            self._root_event_cache[session_id] = root_event_id
            return root_event_id

        except Exception:
            raise

    def _branch_exists(self, session_id: str, branch_name: str) -> bool:
        """Check if a branch already exists for the session."""
        cache_key = f"{session_id}#{branch_name}"
        if cache_key in self._branch_exists_cache.keys():
            return self._branch_exists_cache[cache_key]

        try:
            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                actorId=self.actor_id,
                filter={"branch": {"name": branch_name}},
                includePayloads=False,
                maxResults=1,
            )

            exists = bool(response.get("events"))
            self._branch_exists_cache[cache_key] = exists
            return exists

        except Exception as e:
            print(f"Exception checking branch {branch_name}: {e}")
            self._branch_exists_cache[cache_key] = False
            return False

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        session_id = self._get_session_id(thread_id, checkpoint_ns)

        try:
            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                actorId=self.actor_id,
                filter={"branch": {"name": CHECKPOINT_BRANCH}},
                includePayloads=True,
                maxResults=100 if get_checkpoint_id(config) else 1,
            )

            events = response.get("events", [])
            if not events:
                return None

            if checkpoint_id := get_checkpoint_id(config):
                for event in events:
                    if checkpoint_id in event["eventId"]:
                        return self._event_to_tuple(event, config)
            else:
                for event in events:
                    tuple_result = self._event_to_tuple(event, config)
                    if tuple_result is not None:
                        return tuple_result
        except Exception:
            return None

    def _event_to_tuple(
        self, event: dict, config: RunnableConfig
    ) -> Optional[CheckpointTuple]:
        try:
            payload = event["payload"][0]["blob"]

            if "checkpoint" not in payload:
                return None

            checkpoint_data = self.serde.loads_typed(
                self._decode_data(payload["checkpoint"])
            )
            metadata = self.serde.loads_typed(self._decode_data(payload["metadata"]))
            writes = [
                (
                    w["task_id"],
                    w["channel"],
                    self.serde.loads_typed(self._decode_data(w["value"])),
                )
                for w in payload.get("writes", [])
            ]

            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            channel_values = self._load_channel_values(
                thread_id, checkpoint_ns, checkpoint_data.get("channel_versions", {})
            )

            checkpoint_with_values = {
                **checkpoint_data,
                "channel_values": channel_values,
            }

            parent_config = None
            if payload.get("parent_config"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": payload["parent_config"],
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_data["id"],
                    }
                },
                checkpoint=checkpoint_with_values,
                metadata=metadata,
                pending_writes=writes,
                parent_config=parent_config,
            )
        except Exception:
            return None

    def _load_channel_values(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        """Load channel values from blob storage."""
        session_id = self._get_session_id(thread_id, checkpoint_ns)
        channel_values: dict[str, Any] = {}

        try:
            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                actorId=self.actor_id,
                filter={"branch": {"name": BLOBS_BRANCH}},
                includePayloads=True,
                maxResults=100,
            )

            for event in response.get("events", []):
                try:
                    blob = event["payload"][0]["blob"]
                    for key, value in blob.items():
                        parts = key.split("#")
                        if len(parts) >= 3:
                            channel = parts[-2]
                            version = parts[-1]
                            if (
                                channel in versions
                                and str(versions[channel]) == version
                            ):
                                channel_values[channel] = self.serde.loads_typed(
                                    self._decode_data(value)
                                )
                except Exception:
                    continue

        except Exception:
            pass

        return channel_values

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        session_id = self._get_session_id(thread_id, checkpoint_ns)
        config_checkpoint_id = get_checkpoint_id(config)
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        try:
            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                actorId=self.actor_id,
                filter={"branch": {"name": CHECKPOINT_BRANCH}},
                includePayloads=True,
                maxResults=limit or 10,
            )

            count = 0
            for event in response.get("events", []):
                tuple_result = self._event_to_tuple(event, config)
                if tuple_result is None:
                    continue

                checkpoint_id = tuple_result.checkpoint["id"]

                if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                    continue

                if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                    continue

                if filter and not all(
                    query_value == tuple_result.metadata.get(query_key)
                    for query_key, query_value in filter.items()
                ):
                    continue

                if limit is not None and count >= limit:
                    break

                count += 1
                yield tuple_result
        except Exception:
            return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        session_id = self._get_session_id(thread_id, checkpoint_ns)

        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values", {})

        if values and new_versions:
            blob_data = {}
            for k, v in new_versions.items():
                if k in values:
                    blob_key = f"{session_id}#{k}#{v}"
                    blob_data[blob_key] = self._encode_data(
                        self.serde.dumps_typed(values[k])
                    )

            if blob_data:
                if not self._branch_exists(session_id, BLOBS_BRANCH):
                    root_event_id = self._get_root_event_id(session_id)
                    branch_spec = {"name": BLOBS_BRANCH, "rootEventId": root_event_id}
                else:
                    branch_spec = {"name": BLOBS_BRANCH}

                try:
                    self.client.create_event(
                        memoryId=self.memory_id,
                        actorId=self.actor_id,
                        sessionId=session_id,
                        eventTimestamp=datetime.utcnow(),
                        payload=[{"blob": blob_data}],
                        branch=branch_spec,
                    )
                    self._branch_exists_cache[f"{session_id}#{BLOBS_BRANCH}"] = True
                except Exception as e:
                    if "already exists" in str(e):
                        self._branch_exists_cache[f"{session_id}#{BLOBS_BRANCH}"] = True
                    else:
                        raise

        payload_data = {
            "checkpoint": self._encode_data(self.serde.dumps_typed(c)),
            "metadata": self._encode_data(
                self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            ),
            "parent_config": config["configurable"].get("checkpoint_id"),
        }

        cache_key = f"{session_id}#{CHECKPOINT_BRANCH}"
        if not self._branch_exists(session_id, CHECKPOINT_BRANCH):
            root_event_id = self._get_root_event_id(session_id)
            branch_spec = {"name": CHECKPOINT_BRANCH, "rootEventId": root_event_id}
        else:
            branch_spec = {"name": CHECKPOINT_BRANCH}

        try:
            self.client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=session_id,
                eventTimestamp=datetime.utcnow(),
                payload=[{"blob": payload_data}],
                branch=branch_spec,
            )
            self._branch_exists_cache[cache_key] = True
        except Exception as e:
            if "already exists" in str(e):
                self._branch_exists_cache[cache_key] = True
            else:
                raise

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        session_id = self._get_session_id(thread_id, checkpoint_ns)

        write_data = [
            {
                "task_id": task_id,
                "channel": channel,
                "value": self._encode_data(self.serde.dumps_typed(value)),
                "task_path": task_path,
            }
            for channel, value in writes
        ]

        cache_key = f"{session_id}#{CHECKPOINT_BRANCH}"
        if not self._branch_exists(session_id, CHECKPOINT_BRANCH):
            root_event_id = self._get_root_event_id(session_id)
            branch_spec = {"name": CHECKPOINT_BRANCH, "rootEventId": root_event_id}
        else:
            branch_spec = {"name": CHECKPOINT_BRANCH}

        try:
            self.client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=session_id,
                eventTimestamp=datetime.utcnow(),
                payload=[{"blob": {"writes": write_data}}],
                branch=branch_spec,
            )
            self._branch_exists_cache[cache_key] = True
        except Exception as e:
            if "already exists" in str(e):
                self._branch_exists_cache[cache_key] = True
            else:
                raise

    def delete_thread(self, thread_id: str) -> None:
        try:
            keys_to_remove = [
                k for k in self._root_event_cache.keys() if k.startswith(thread_id)
            ]
            for key in keys_to_remove:
                del self._root_event_cache[key]

            branch_keys_to_remove = [
                k for k in self._branch_exists_cache.keys() if k.startswith(thread_id)
            ]
            for key in branch_keys_to_remove:
                del self._branch_exists_cache[key]

            response = self.client.list_events(
                memoryId=self.memory_id,
                sessionId=thread_id,
                actorId=self.actor_id,
                filter={"branch": {"name": CHECKPOINT_BRANCH}},
                includePayloads=False,
                maxResults=1000,
            )

            for event in response.get("events", []):
                self.client.delete_event(
                    memoryId=self.memory_id,
                    sessionId=thread_id,
                    eventId=event["eventId"],
                    actorId=self.actor_id,
                )
        except Exception:
            pass

    def _encode_data(self, data: tuple) -> str:
        encoded_data = (
            data[0],
            (
                base64.b64encode(data[1]).decode()
                if isinstance(data[1], bytes)
                else data[1]
            ),
        )
        return base64.b64encode(json.dumps(encoded_data).encode()).decode()

    def _decode_data(self, data: str) -> tuple:
        decoded_data = json.loads(base64.b64decode(data).decode())
        return (
            decoded_data[0],
            (
                base64.b64decode(decoded_data[1])
                if isinstance(decoded_data[1], str)
                else decoded_data[1]
            ),
        )

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)
