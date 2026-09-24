from __future__ import annotations

import datetime
import json
from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.stub import Stubber
from botocore.validate import validate_parameters
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from langgraph_checkpoint_aws.checkpoint.agentcore import AgentCoreMemorySaver
from langgraph_checkpoint_aws.checkpoint.agentcore.constants import (
    EventDecodingError,
    EventNotFoundError,
    InvalidConfigError,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
    ChannelDataEvent,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.snapshot import (
    writes_session_id,
)
from langgraph_checkpoint_aws.checkpoint.deferred_saver import (
    DeferredCheckpointSaver,
)

MEMORY_ID = "snapshot_tests-0123456789"
ACTOR_ID = "snapshot-test-actor"
THREAD_ID = "snapshot-test-thread"


class EventService:
    """Select each service page before applying metadata filters."""

    def __init__(self) -> None:
        self.model = Session().get_service_model("bedrock-agentcore")
        self.events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sequence = 0
        self.fail_create_at: int | None = None
        self.session_page_size = 100
        self.tokens: dict[str, dict[str, Any]] = {}

    def _record(self, operation: str, params: dict[str, Any]) -> None:
        input_shape = self.model.operation_model(operation).input_shape
        assert input_shape is not None
        validate_parameters(params, input_shape)
        self.calls.append((operation, deepcopy(params)))

    def create_event(self, **params: Any) -> dict[str, Any]:
        self._record("CreateEvent", params)
        self.sequence += 1
        if self.sequence == self.fail_create_at:
            msg = "Simulated failure before publishing a checkpoint."
            raise OSError(msg)
        token = params.get("clientToken")
        event = {
            key: deepcopy(value)
            for key, value in params.items()
            if key != "clientToken"
        }
        if token in self.tokens:
            # Like the service: a reused token with identical parameters returns
            # the original event; different parameters are rejected.
            original = self.tokens[token]
            if {k: v for k, v in original.items() if k != "eventId"} != event:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "InvalidInputException",
                            "Message": "Param hash does not match existing event",
                        }
                    },
                    "CreateEvent",
                )
            return {"event": deepcopy(original)}
        event["eventId"] = f"1750000000000#{self.sequence:032x}"
        self.events[(params["actorId"], params["sessionId"])].insert(0, event)
        if token is not None:
            self.tokens[token] = event
        return {"event": deepcopy(event)}

    def list_events(self, **params: Any) -> dict[str, Any]:
        self._record("ListEvents", params)
        events = self.events[(params["actorId"], params["sessionId"])]
        filter_key = json.dumps(params.get("filter"), sort_keys=True)
        start = 0
        token = params.get("nextToken")
        if token is not None:
            # Like the service, a continuation token is bound to its filter.
            start_text, _, token_filter = token.partition("|")
            if token_filter != filter_key:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "Initial metadata filter in the token does "
                            "not match provided metadata filter",
                        }
                    },
                    "ListEvents",
                )
            start = int(start_text)
        end = start + params.get("maxResults", 20)
        page = events[start:end]
        for condition in params.get("filter", {}).get("eventMetadata", []):
            key = condition["left"]["metadataKey"]
            value = condition["right"]["metadataValue"]
            page = [
                event for event in page if event.get("metadata", {}).get(key) == value
            ]
        result: dict[str, Any] = {"events": deepcopy(page)}
        if not params.get("includePayloads", True):
            for event in result["events"]:
                event.pop("payload", None)
        if end < len(events):
            result["nextToken"] = f"{end}|{filter_key}"
        return result

    def delete_event(self, **params: Any) -> dict[str, Any]:
        self._record("DeleteEvent", params)
        key = (params["actorId"], params["sessionId"])
        self.events[key] = [
            event for event in self.events[key] if event["eventId"] != params["eventId"]
        ]
        return {}

    def get_event(self, **params: Any) -> dict[str, Any]:
        self._record("GetEvent", params)
        for event in self.events[(params["actorId"], params["sessionId"])]:
            if event["eventId"] == params["eventId"]:
                return {"event": deepcopy(event)}
        raise ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Missing chunk",
                }
            },
            "GetEvent",
        )

    def list_sessions(self, **params: Any) -> dict[str, Any]:
        self._record("ListSessions", params)
        sessions = sorted(
            session
            for (actor, session), events in self.events.items()
            if actor == params["actorId"] and events
        )
        start = int(params.get("nextToken", "0"))
        end = start + min(params.get("maxResults", 20), self.session_page_size)
        result: dict[str, Any] = {
            "sessionSummaries": [
                {
                    "sessionId": session,
                    "actorId": params["actorId"],
                    "createdAt": datetime.datetime.now(datetime.timezone.utc),
                }
                for session in sessions[start:end]
            ]
        }
        if end < len(sessions):
            result["nextToken"] = str(end)
        return result


@pytest.fixture
def service() -> Iterator[EventService]:
    service = EventService()
    with patch("boto3.client", return_value=service):
        yield service


@pytest.fixture
def saver(service: EventService) -> AgentCoreMemorySaver:
    return AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")


def config(
    checkpoint_id: str | None = None,
    *,
    actor_id: str = ACTOR_ID,
    namespace: str = "",
    thread_id: str = THREAD_ID,
) -> RunnableConfig:
    configurable = {
        "thread_id": thread_id,
        "actor_id": actor_id,
        "checkpoint_ns": namespace,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def checkpoint(index: int, *, channels: int = 2) -> Checkpoint:
    values: dict[str, Any] = {
        "messages": [HumanMessage(content=f"Remember fact {index}", id=f"msg-{index}")],
        "unchanged": {"customer": "synthetic-test-customer"},
    }
    for channel in range(2, channels):
        values[f"channel-{channel}"] = f"value-{channel}"
    return Checkpoint(
        v=4,
        id=f"{index:032x}",
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        channel_values=values,
        channel_versions={
            key: str(index) if key == "messages" else "1" for key in values
        },
        versions_seen={},
        updated_channels=None,
    )


def metadata(index: int) -> CheckpointMetadata:
    return {"source": "loop", "step": index, "parents": {}}


def save(
    saver: AgentCoreMemorySaver, index: int, *, channels: int = 2
) -> RunnableConfig:
    state = checkpoint(index, channels=channels)
    return saver.put(config(), state, metadata(index), {"messages": str(index)})


@pytest.mark.parametrize("history_length", [1, 30, 300])
def test_latest_read_cost_does_not_grow_with_history(
    service: EventService, saver: AgentCoreMemorySaver, history_length: int
) -> None:
    for index in range(1, history_length + 1):
        saved = save(saver, index)
        saver.put_writes(saved, [("result", index)], f"task-{index}")
    # A late write to an old checkpoint must not change the latest checkpoint.
    saver.put_writes(config(checkpoint(1)["id"]), [("late", True)], "old-task")
    service.calls.clear()

    # Use a new saver to ensure this is independent of process-local caches.
    result = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot").get_tuple(
        config()
    )

    assert result is not None
    assert result.checkpoint["id"] == checkpoint(history_length)["id"]
    assert (
        result.checkpoint["channel_values"]
        == checkpoint(history_length)["channel_values"]
    )
    assert result.pending_writes is not None
    assert (f"task-{history_length}", "result", history_length) in result.pending_writes
    reads = [params for operation, params in service.calls if operation == "ListEvents"]
    assert len(reads) == 2
    assert reads[0]["sessionId"] == THREAD_ID
    assert reads[0]["maxResults"] == 1
    assert reads[1]["sessionId"] == writes_session_id(
        THREAD_ID, checkpoint(history_length)["id"]
    )
    assert all(operation == "ListEvents" for operation, _ in service.calls)


@pytest.mark.parametrize("limit", [0, 1, 100])
def test_snapshot_rejects_blob_limits(service: EventService, limit: int) -> None:
    with pytest.raises(ValueError, match="requires limit=None"):
        AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot", limit=limit)
    assert service.calls == []


def test_pending_writes_can_arrive_before_and_after_checkpoint(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    saved = config(checkpoint(1)["id"])
    saver.put_writes(saved, [("before", 1)], "early")
    save(saver, 1)
    saver.put_writes(saved, [("after", 2)], "late")

    result = saver.get_tuple(config())

    assert result is not None
    assert result.pending_writes is not None
    assert set(result.pending_writes) == {("early", "before", 1), ("late", "after", 2)}


def test_unchanged_and_empty_channels_are_self_contained(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    save(saver, 1)
    state = checkpoint(2)
    state["channel_versions"]["empty"] = 3
    saver.put(config(), state, metadata(2), {"messages": "2", "empty": 3})
    # Remove the old snapshot: reconstructing the new one must still succeed.
    service.events[(ACTOR_ID, THREAD_ID)].pop()

    result = saver.get_tuple(config())

    assert result is not None
    assert result.checkpoint["channel_values"] == state["channel_values"]
    assert "empty" not in result.checkpoint["channel_values"]


def test_chunked_snapshot_uses_direct_get_event_references(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    save(saver, 1, channels=7)
    create_calls = [params for op, params in service.calls if op == "CreateEvent"]
    assert len(create_calls) == 4
    assert all(len(params["payload"]) <= 2 for params in create_calls)
    commit = saver.serializer.deserialize_event(create_calls[-1]["payload"][-1]["blob"])
    assert isinstance(commit, CheckpointEvent)
    assert len(commit.chunk_event_ids) == 3
    service.calls.clear()

    result = saver.get_tuple(config())

    assert result is not None
    assert (
        result.checkpoint["channel_values"]
        == checkpoint(1, channels=7)["channel_values"]
    )
    assert [op for op, _ in service.calls].count("ListEvents") == 2
    assert [op for op, _ in service.calls].count("GetEvent") == 3


def test_chunk_references_cannot_push_final_event_over_byte_limit(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = checkpoint(1, channels=3)
    saver.put(config(), state, metadata(1), state["channel_versions"])
    initial_payload = service.events[(ACTOR_ID, THREAD_ID)][0]["payload"]
    # The last channel and commit fit exactly before chunk references are added.
    budget = sum(len(json.dumps(item).encode()) for item in initial_payload[-2:])
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.SNAPSHOT_PAYLOAD_BYTES",
        budget,
    )
    service.calls.clear()

    saver.put(config(), state, metadata(1), state["channel_versions"])

    creates = [params for op, params in service.calls if op == "CreateEvent"]
    assert len(creates) == 3
    assert all(
        sum(len(json.dumps(item).encode()) for item in params["payload"]) <= budget
        for params in creates
    )
    result = saver.get_tuple(config())
    assert result is not None
    assert result.checkpoint["channel_values"] == state["channel_values"]


def test_oversized_payload_fails_before_writing_any_chunks(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.SNAPSHOT_PAYLOAD_BYTES",
        1,
    )
    with pytest.raises(ValueError, match="event size limit"):
        save(saver, 1)
    assert not service.calls


def test_uncommitted_chunks_do_not_replace_previous_checkpoint(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save(saver, 1)
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    service.fail_create_at = service.sequence + 2
    with pytest.raises(OSError, match="Simulated failure"):
        save(saver, 2, channels=5)

    result = saver.get_tuple(config())

    assert result is not None
    assert result.checkpoint["id"] == checkpoint(1)["id"]
    assert result.checkpoint["channel_values"] == checkpoint(1)["channel_values"]


def test_missing_chunk_raises_instead_of_returning_partial_state(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    save(saver, 1, channels=3)
    service.events[(ACTOR_ID, THREAD_ID)].pop()
    with pytest.raises(ClientError, match="Missing chunk"):
        saver.get_tuple(config())


def test_snapshot_accepts_additional_event_metadata(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    save(saver, 1, channels=3)
    for event in service.events[(ACTOR_ID, THREAD_ID)]:
        event["metadata"]["custom_label"] = {"stringValue": "test"}

    result = saver.get_tuple(config())

    assert result is not None
    assert (
        result.checkpoint["channel_values"]
        == checkpoint(1, channels=3)["channel_values"]
    )


def test_missing_channel_raises_instead_of_resetting_thread(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    save(saver, 1)
    service.events[(ACTOR_ID, THREAD_ID)][0]["payload"].pop(0)
    with pytest.raises(EventNotFoundError, match="Missing channel data"):
        saver.get_tuple(config())


def test_corrupt_snapshot_is_not_silently_skipped(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    save(saver, 1)
    service.events[(ACTOR_ID, THREAD_ID)][0]["payload"][-1]["blob"] = "{"
    with pytest.raises(EventDecodingError):
        saver.get_tuple(config())


def test_corrupt_pending_writes_raise(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    saved = save(saver, 1)
    saver.put_writes(saved, [("answer", "saved")], "task")
    session = writes_session_id(THREAD_ID, checkpoint(1)["id"])
    service.events[(ACTOR_ID, session)][0]["payload"][0]["blob"] = "{"
    with pytest.raises(EventDecodingError):
        saver.get_tuple(config())


def test_specific_checkpoint_follows_empty_filtered_pages(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    for index in range(1, 9):
        save(saver, index)
    service.calls.clear()

    result = AgentCoreMemorySaver(
        MEMORY_ID, checkpoint_format="snapshot", max_results=2
    ).get_tuple(config(checkpoint(1)["id"]))

    assert result is not None
    assert result.checkpoint["id"] == checkpoint(1)["id"]
    filtered = [
        params
        for op, params in service.calls
        if op == "ListEvents" and "filter" in params
    ]
    assert len(filtered) == 4
    assert filtered[-1]["nextToken"].startswith("6|")


def test_empty_latest_thread_requires_one_call(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    assert saver.get_tuple(config()) is None
    assert len(service.calls) == 1


@pytest.mark.parametrize("conversational_events", [1, 50])
def test_non_checkpoint_conversations_do_not_force_full_scan(
    service: EventService, saver: AgentCoreMemorySaver, conversational_events: int
) -> None:
    for index in range(1, 20):
        save(saver, index)
    for _ in range(conversational_events):
        service.create_event(
            memoryId=MEMORY_ID,
            actorId=ACTOR_ID,
            sessionId=THREAD_ID,
            eventTimestamp=datetime.datetime.now(datetime.timezone.utc),
            payload=[
                {"conversational": {"content": {"text": "New input"}, "role": "USER"}}
            ],
        )
    service.calls.clear()

    result = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot").get_tuple(
        config()
    )

    assert result is not None
    assert result.checkpoint["id"] == checkpoint(19)["id"]
    # Head peek, one filtered request for the newest commit, pending writes.
    assert len(service.calls) == 3
    head, filtered = [
        params
        for operation, params in service.calls
        if operation == "ListEvents" and params["sessionId"] == THREAD_ID
    ]
    assert head["maxResults"] == 1 and "filter" not in head
    assert filtered["filter"]["eventMetadata"][0]["left"]["metadataKey"] == "lc_kind"


def store_legacy_checkpoint(
    saver: AgentCoreMemorySaver, index: int, *, with_channels: bool = True
) -> None:
    state = checkpoint(index)
    state["channel_versions"]["unchanged"] = "1"
    values = state["channel_values"]
    checkpoint_data = dict(state)
    checkpoint_data.pop("channel_values")
    events: list[Any] = []
    for channel, value in values.items():
        if with_channels or channel != "unchanged":
            events.append(
                ChannelDataEvent(
                    channel=channel,
                    version=str(state["channel_versions"][channel]),
                    value=value,
                    thread_id=THREAD_ID,
                )
            )
    events.append(
        CheckpointEvent(
            checkpoint_id=state["id"],
            checkpoint_data=checkpoint_data,
            metadata=dict(metadata(index)),
            thread_id=THREAD_ID,
        )
    )
    saver.checkpoint_event_client.store_blob_events_batch(events, THREAD_ID, ACTOR_ID)


@pytest.mark.parametrize("max_results", [1, None])
def test_snapshot_fallback_preserves_legacy_channels_and_pending_writes(
    service: EventService, saver: AgentCoreMemorySaver, max_results: int | None
) -> None:
    store_legacy_checkpoint(saver, 1)
    store_legacy_checkpoint(saver, 2, with_channels=False)
    saved = config(checkpoint(2)["id"])
    saver.checkpoint_event_client.store_blob_event(
        WritesEvent(
            checkpoint_id=checkpoint(2)["id"],
            writes=[WriteItem(task_id="old-task", channel="old", value=1)],
        ),
        THREAD_ID,
        ACTOR_ID,
    )
    # New workers can also add pending writes to a legacy checkpoint.
    saver.put_writes(saved, [("new", 2)], "new-task")

    reader = AgentCoreMemorySaver(
        MEMORY_ID, checkpoint_format="snapshot", max_results=max_results
    )
    result = reader.get_tuple(config())

    assert result is not None
    assert result.checkpoint["channel_values"] == checkpoint(2)["channel_values"]
    assert result.pending_writes is not None
    assert set(result.pending_writes) == {
        ("old-task", "old", 1),
        ("new-task", "new", 2),
    }


def test_legacy_history_remains_readable_after_upgrade(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    store_legacy_checkpoint(saver, 1)
    save(saver, 2)
    latest = saver.get_tuple(config())
    assert latest is not None
    assert latest.checkpoint["id"] == checkpoint(2)["id"]
    old = saver.get_tuple(config(checkpoint(1)["id"]))
    assert old is not None
    assert old.checkpoint["channel_values"] == checkpoint(1)["channel_values"]


def test_newer_legacy_writer_is_not_hidden_by_snapshot_lookup(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    save(saver, 1)
    store_legacy_checkpoint(saver, 2)
    result = saver.get_tuple(config())
    assert result is not None
    assert result.checkpoint["id"] == checkpoint(2)["id"]


def test_list_limit_counts_checkpoints_and_applies_metadata_filter(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    for index in range(1, 6):
        saved = save(saver, index, channels=5)
        saver.put_writes(saved, [("result", index)], f"task-{index}")

    latest = list(saver.list(config(), limit=1))
    assert len(latest) == 1
    assert latest[0].checkpoint["id"] == checkpoint(5)["id"]
    assert latest[0].pending_writes == [("task-5", "result", 5)]
    filtered = list(
        saver.list(
            config(),
            limit=1,
            filter={"step": 3},
            before=config(checkpoint(5)["id"]),
        )
    )
    assert len(filtered) == 1
    assert filtered[0].checkpoint["id"] == checkpoint(3)["id"]


def test_deferred_saver_keeps_checkpoint_and_inline_writes_in_one_request(
    service: EventService,
) -> None:
    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    deferred = DeferredCheckpointSaver(saver, batch_writes=True)
    state = checkpoint(1)
    saved = deferred.put(config(), state, metadata(1), state["channel_versions"])
    deferred.put_writes(saved, [("answer", "saved")], "task")
    deferred.flush()

    creates = [params for op, params in service.calls if op == "CreateEvent"]
    assert len(creates) == 1
    assert all("blob" in item for item in creates[0]["payload"])
    result = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot").get_tuple(
        config()
    )
    assert result is not None
    assert result.pending_writes == [("task", "answer", "saved")]
    assert result.checkpoint["channel_values"] == state["channel_values"]


def test_save_does_not_mutate_checkpoint(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    state = checkpoint(1)
    original = deepcopy(state)
    saver.put(config(), state, metadata(1), state["channel_versions"])
    assert state == original
    assert all(
        "blob" in item for item in service.events[(ACTOR_ID, THREAD_ID)][0]["payload"]
    )


def test_botocore_client_serializes_snapshots_and_metadata_filters(
    service: EventService,
) -> None:
    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    saved = save(saver, 1)
    saver.get_tuple(saved)
    create_request = next(params for op, params in service.calls if op == "CreateEvent")
    list_request = next(
        params
        for op, params in service.calls
        if op == "ListEvents" and "filter" in params
    )
    # Use a real botocore client with synthetic credentials and stubbed responses.
    # This exercises request serialization as well as the service-model validator.
    client: Any = Session().create_client(
        "bedrock-agentcore",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    try:
        with Stubber(client) as stubber:
            stubber.add_response(
                "create_event",
                {"event": service.events[(ACTOR_ID, THREAD_ID)][0]},
                create_request,
            )
            stubber.add_response("list_events", {"events": []}, list_request)
            client.create_event(**create_request)
            client.list_events(**list_request)
            stubber.assert_no_pending_responses()
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_read_preserves_state(service: EventService) -> None:
    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    state = checkpoint(1)
    saved = await saver.aput(config(), state, metadata(1), state["channel_versions"])
    await saver.aput_writes(saved, [("answer", "saved")], "task")
    result = await saver.aget_tuple(config())
    assert result is not None
    assert result.checkpoint["channel_values"] == state["channel_values"]
    assert result.pending_writes == [("task", "answer", "saved")]
    assert len([item async for item in saver.alist(config(), limit=1)]) == 1


def test_actor_and_checkpoint_namespace_isolation(service: EventService) -> None:
    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    for actor, namespace, text in [
        (ACTOR_ID, "", "root"),
        (ACTOR_ID, "subgraph:one", "nested"),
        ("another-actor", "", "other actor"),
    ]:
        state = checkpoint(1)
        state["channel_values"]["unchanged"] = text
        saved = saver.put(
            config(actor_id=actor, namespace=namespace),
            state,
            metadata(1),
            state["channel_versions"],
        )
        saver.put_writes(saved, [("answer", text)], "task")
        result = saver.get_tuple(config(actor_id=actor, namespace=namespace))
        assert result is not None
        assert result.checkpoint["channel_values"]["unchanged"] == text
        assert result.pending_writes == [("task", "answer", text)]


def test_generated_session_ids_and_metadata_accept_arbitrary_checkpoint_ids(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    state = checkpoint(1)
    state["id"] = "special|checkpoint🙂/" * 30
    saved = saver.put(config(), state, metadata(1), state["channel_versions"])
    saver.put_writes(saved, [("answer", "saved")], "task")
    result = saver.get_tuple(saved)
    assert result is not None
    assert result.checkpoint["id"] == state["id"]
    assert len(writes_session_id(THREAD_ID, state["id"])) <= 100
    assert all(
        len(value["stringValue"]) <= 256
        for value in service.events[(ACTOR_ID, THREAD_ID)][0]["metadata"].values()
    )


def test_delete_thread_removes_pending_write_sessions_and_orphans_only(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    for index in range(1, 4):
        saved = save(saver, index)
        saver.put_writes(saved, [("answer", index)], "task")
    saver.put_writes(config("orphan-checkpoint"), [("orphan", True)], "task")
    other = config(thread_id="unrelated-thread")
    state = checkpoint(1)
    other_saved = saver.put(other, state, metadata(1), state["channel_versions"])
    saver.put_writes(other_saved, [("keep", True)], "task")
    service.session_page_size = 2

    saver.delete_thread(THREAD_ID, ACTOR_ID)

    assert saver.get_tuple(config()) is None
    assert not service.events[
        (ACTOR_ID, writes_session_id(THREAD_ID, "orphan-checkpoint"))
    ]
    remaining = saver.get_tuple(other)
    assert remaining is not None
    assert remaining.pending_writes == [("task", "keep", True)]
    assert len([op for op, _ in service.calls if op == "ListSessions"]) > 1


def test_interrupt_can_resume_using_a_fresh_saver(service: EventService) -> None:
    def approve(state: MessagesState) -> dict[str, Any]:
        decision = interrupt("Approve this response?")
        return {"messages": [AIMessage(content=f"Decision: {decision}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    graph = builder.compile(
        checkpointer=AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    )
    graph.invoke({"messages": [HumanMessage(content="Please proceed.")]}, config())

    reader = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    saved = reader.get_tuple(config())
    assert saved is not None
    assert saved.pending_writes is not None
    assert any(channel == "__interrupt__" for _, channel, _ in saved.pending_writes)
    resumed = builder.compile(checkpointer=reader).invoke(
        Command(resume="approved"), config()
    )
    assert resumed["messages"][-1].content == "Decision: approved"


def test_graph_remembers_first_turn(service: EventService) -> None:
    def reply(state: MessagesState) -> dict[str, Any]:
        first_input = state["messages"][0].content
        return {"messages": [AIMessage(content=f"First input: {first_input}")]}

    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=saver)

    graph.invoke(
        {"messages": [HumanMessage(content="My preferred seat is a window.")]},
        config(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What seat do I prefer?")]}, config()
    )

    assert (
        result["messages"][-1].content == "First input: My preferred seat is a window."
    )


def test_legacy_mode_rejects_snapshots_instead_of_losing_pending_writes(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    saved = save(saver, 1)
    saver.put_writes(saved, [("answer", "saved")], "task")
    legacy = AgentCoreMemorySaver(MEMORY_ID)

    with pytest.raises(InvalidConfigError, match="checkpoint_format='snapshot'"):
        legacy.get_tuple(config())
    with pytest.raises(InvalidConfigError, match="checkpoint_format='snapshot'"):
        list(legacy.list(config()))


def test_invalid_checkpoint_format_is_rejected() -> None:
    invalid: Any = "unknown"
    with pytest.raises(ValueError, match="checkpoint_format"):
        AgentCoreMemorySaver(MEMORY_ID, checkpoint_format=invalid)


def test_integer_channel_versions_restore_values(service: EventService) -> None:
    saver = AgentCoreMemorySaver(MEMORY_ID, checkpoint_format="snapshot")
    state = checkpoint(1)
    state["channel_versions"] = {key: 1 for key in state["channel_values"]}
    saver.put(config(), state, metadata(1), state["channel_versions"])

    restored = saver.get_tuple(config())

    assert restored is not None
    assert restored.checkpoint["channel_values"] == state["channel_values"]


@pytest.mark.parametrize("operation", ["get_tuple", "list"])
def test_legacy_reads_still_pass_blob_limit_to_event_client(
    service: EventService, operation: str
) -> None:
    legacy = AgentCoreMemorySaver(MEMORY_ID, limit=7, max_results=3)
    with patch.object(
        legacy.checkpoint_event_client, "get_events", return_value=[]
    ) as read:
        if operation == "get_tuple":
            legacy.get_tuple(config())
        else:
            list(legacy.list(config(), limit=7))
    read.assert_called_once_with(THREAD_ID, ACTOR_ID, 7, 3)


def test_specific_checkpoint_lookup_filters_out_data_chunks(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    saved = save(saver, 1, channels=7)
    service.calls.clear()

    result = saver.get_tuple(saved)

    assert result is not None
    filtered = [
        params
        for op, params in service.calls
        if op == "ListEvents" and "filter" in params
    ]
    assert len(filtered) == 1
    conditions = {
        condition["left"]["metadataKey"]: condition["right"]["metadataValue"]
        for condition in filtered[0]["filter"]["eventMetadata"]
    }
    assert conditions["lc_kind"] == {"stringValue": "checkpoint"}
    # Chunks are read once, through GetEvent, not also through the filtered page.
    assert [op for op, _ in service.calls].count("GetEvent") == 3


def test_retried_save_after_failed_commit_does_not_duplicate_writes(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph_checkpoint_aws.checkpoint.deferred_saver import PendingWrite

    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    state = checkpoint(1, channels=4)
    pending = [PendingWrite(config(), [("__interrupt__", "approve?")], "task", "")]
    # Chunks are written before the commit record; fail on the commit.
    service.fail_create_at = 3
    with pytest.raises(OSError, match="Simulated failure"):
        saver.put_with_writes(
            config(), state, metadata(1), state["channel_versions"], pending
        )
    orphans = len(service.events[(ACTOR_ID, THREAD_ID)])

    saver.put_with_writes(
        config(), state, metadata(1), state["channel_versions"], pending
    )

    # The retry reuses the already-written chunks instead of adding new ones.
    assert len(service.events[(ACTOR_ID, THREAD_ID)]) == orphans + 1
    listed = list(saver.list(config()))
    assert [item.checkpoint["id"] for item in listed] == [state["id"]]
    assert listed[0].pending_writes == [("task", "__interrupt__", "approve?")]
    latest = saver.get_tuple(config())
    assert latest is not None
    assert latest.pending_writes == [("task", "__interrupt__", "approve?")]


def test_legacy_integer_channel_versions_restore_values(
    service: EventService,
) -> None:
    legacy = AgentCoreMemorySaver(MEMORY_ID)
    state = checkpoint(1)
    state["channel_versions"] = {key: 1 for key in state["channel_values"]}
    legacy.put(config(), state, metadata(1), state["channel_versions"])

    restored = legacy.get_tuple(config())

    assert restored is not None
    assert restored.checkpoint["channel_values"] == state["channel_values"]


def test_writes_sessions_share_the_thread_prefix_across_namespaces() -> None:
    root = writes_session_id(THREAD_ID, "cp")
    nested = writes_session_id(THREAD_ID, "cp", "subgraph:one")
    assert root != nested
    assert root.rsplit("-", 1)[0] == nested.rsplit("-", 1)[0]
    assert max(len(root), len(nested)) <= 100


def test_delete_thread_removes_namespaced_pending_write_sessions(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    nested = config(namespace="subgraph:one")
    state = checkpoint(1)
    saved = saver.put(nested, state, metadata(1), state["channel_versions"])
    saver.put_writes(saved, [("__interrupt__", "approve?")], "task")
    assert saver.get_tuple(nested) is not None

    saver.delete_thread(THREAD_ID, ACTOR_ID)

    assert not service.events[
        (ACTOR_ID, writes_session_id(THREAD_ID, state["id"], "subgraph:one"))
    ]


def test_legacy_thread_under_conversational_events_still_falls_back(
    service: EventService, saver: AgentCoreMemorySaver
) -> None:
    store_legacy_checkpoint(saver, 1)
    store_legacy_checkpoint(saver, 2)
    for _ in range(3):
        service.create_event(
            memoryId=MEMORY_ID,
            actorId=ACTOR_ID,
            sessionId=THREAD_ID,
            eventTimestamp=datetime.datetime.now(datetime.timezone.utc),
            payload=[
                {"conversational": {"content": {"text": "New input"}, "role": "USER"}}
            ],
        )

    result = saver.get_tuple(config())

    assert result is not None
    assert result.checkpoint["id"] == checkpoint(2)["id"]
    assert result.checkpoint["channel_values"] == checkpoint(2)["channel_values"]


def test_chunk_timestamps_precede_the_commit_and_are_deterministic(
    service: EventService,
    saver: AgentCoreMemorySaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "langgraph_checkpoint_aws.checkpoint.agentcore.snapshot.MAX_PAYLOAD_ITEMS_PER_EVENT",
        2,
    )
    state = checkpoint(1, channels=7)
    saver.put(config(), state, metadata(1), state["channel_versions"])
    creates = [params for op, params in service.calls if op == "CreateEvent"]
    stamps = [params["eventTimestamp"] for params in creates]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    assert stamps[-1] == datetime.datetime.fromisoformat(state["ts"])

    # The same save again reproduces identical requests (tokens and timestamps).
    service.calls.clear()
    saver.put(config(), state, metadata(1), state["channel_versions"])
    retried = [params for op, params in service.calls if op == "CreateEvent"]
    assert [p["clientToken"] for p in retried] == [p["clientToken"] for p in creates]
    assert [p["eventTimestamp"] for p in retried] == stamps
