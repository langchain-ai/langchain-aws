from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from langgraph_checkpoint_aws import AgentCoreMemorySaver
from langgraph_checkpoint_aws.checkpoint.agentcore.constants import InvalidConfigError
from langgraph_checkpoint_aws.checkpoint.agentcore.snapshot import writes_session_id


@dataclass
class MemoryCase:
    memory_id: str
    region: str
    actor: str = field(default_factory=lambda: f"snapshot-{uuid4().hex}")
    threads: set[str] = field(default_factory=set)
    calls: Counter[str] = field(default_factory=Counter)

    def config(self, thread: str = "thread") -> RunnableConfig:
        self.threads.add(thread)
        return {"configurable": {"actor_id": self.actor, "thread_id": thread}}

    def saver(
        self,
        *,
        checkpoint_format: Literal["legacy", "snapshot"] = "snapshot",
        **kwargs: Any,
    ) -> AgentCoreMemorySaver:
        saver = AgentCoreMemorySaver(
            self.memory_id,
            region_name=self.region,
            checkpoint_format=checkpoint_format,
            **kwargs,
        )

        def after_call(model: Any, **_: Any) -> None:
            self.calls[model.name] += 1

        saver.checkpoint_event_client.client._client.meta.events.register(
            "after-call.bedrock-agentcore.*", after_call
        )
        return saver


@pytest.fixture
def memory_case() -> Iterator[MemoryCase]:
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        pytest.skip("AGENTCORE_MEMORY_ID environment variable not set")
    case = MemoryCase(memory_id, os.environ.get("AWS_REGION", "us-west-2"))
    try:
        yield case
    finally:
        saver = case.saver()
        for thread in case.threads:
            saver.delete_thread(thread, case.actor)


def make_checkpoint(index: int, values: dict[str, Any]) -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = {key: str(index) for key in values}
    return checkpoint


def test_latest_snapshot_does_not_page_through_history(memory_case: MemoryCase) -> None:
    config = memory_case.config()
    writer = memory_case.saver()
    parent = config
    for index in range(1, 26):
        checkpoint = make_checkpoint(index, {"counter": index, "static": "unchanged"})
        checkpoint["channel_versions"]["static"] = "1"
        parent = writer.put(
            parent,
            checkpoint,
            {"source": "loop", "step": index},
            checkpoint["channel_versions"] if index == 1 else {"counter": str(index)},
        )
    writer.put_writes(parent, [("answer", 25)], "task")
    reader = memory_case.saver(max_results=1)
    memory_case.calls.clear()

    restored = reader.get_tuple(config)

    assert restored is not None
    assert restored.checkpoint["id"] == checkpoint["id"]
    assert restored.checkpoint["channel_values"] == checkpoint["channel_values"]
    assert restored.pending_writes == [("task", "answer", 25)]
    assert memory_case.calls == {"ListEvents": 2}
    history = list(memory_case.saver().list(config, limit=10))
    assert [item.checkpoint["channel_values"]["counter"] for item in history] == list(
        range(25, 15, -1)
    )


def test_snapshot_retains_two_turns_with_a_fresh_saver(
    memory_case: MemoryCase,
) -> None:
    def reply(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=str(state["messages"][0].content))]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    config = memory_case.config()
    first = memory_case.saver()
    builder.compile(checkpointer=first).invoke(
        {"messages": [HumanMessage("My preferred seat is a window.")]}, config
    )
    second = memory_case.saver()

    result = builder.compile(checkpointer=second).invoke(
        {"messages": [HumanMessage("What seat do I prefer?")]}, config
    )

    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "My preferred seat is a window."


@pytest.mark.parametrize(
    "values",
    [
        {f"channel-{index}": index for index in range(205)},
        {f"channel-{index}": "x" * 900_000 for index in range(9)},
    ],
    ids=["item-limit", "byte-limit"],
)
def test_chunked_snapshot_restores_complete_state(
    memory_case: MemoryCase, values: dict[str, Any]
) -> None:
    config = memory_case.config()
    checkpoint = make_checkpoint(1, values)
    writer = memory_case.saver()
    writer.put(
        config,
        checkpoint,
        {"source": "loop", "step": 1},
        checkpoint["channel_versions"],
    )
    assert memory_case.calls["CreateEvent"] > 1
    reader = memory_case.saver()
    memory_case.calls.clear()

    restored = reader.get_tuple(config)

    assert restored is not None
    assert restored.checkpoint["channel_values"] == values
    assert memory_case.calls["GetEvent"] > 0
    assert memory_case.calls["ListEvents"] == 2


def test_pending_writes_before_and_after_snapshot(memory_case: MemoryCase) -> None:
    config = memory_case.config()
    checkpoint = make_checkpoint(1, {"counter": 1})
    specific: RunnableConfig = {
        "configurable": {**config["configurable"], "checkpoint_id": checkpoint["id"]}
    }
    writer = memory_case.saver()
    writer.put_writes(specific, [("before", 1)], "early")
    writer.put(
        config,
        checkpoint,
        {"source": "loop", "step": 1},
        checkpoint["channel_versions"],
    )
    writer.put_writes(specific, [("after", 2)], "late")

    restored = memory_case.saver(max_results=1).get_tuple(config)

    assert restored is not None
    assert restored.pending_writes is not None
    assert set(restored.pending_writes) == {
        ("early", "before", 1),
        ("late", "after", 2),
    }


def test_interrupt_resumes_with_a_fresh_saver(memory_case: MemoryCase) -> None:
    def approve(state: MessagesState) -> dict[str, Any]:
        decision = interrupt("Approve?")
        return {"messages": [AIMessage(content=str(decision))]}

    builder = StateGraph(MessagesState)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    config = memory_case.config()
    builder.compile(checkpointer=memory_case.saver()).invoke(
        {"messages": [HumanMessage("Please proceed.")]}, config
    )

    result = builder.compile(checkpointer=memory_case.saver()).invoke(
        Command(resume="approved"), config
    )

    assert result["messages"][-1].content == "approved"


def test_legacy_thread_can_migrate_to_snapshot_mode(memory_case: MemoryCase) -> None:
    config = memory_case.config()
    legacy = memory_case.saver(checkpoint_format="legacy")
    first = make_checkpoint(1, {"counter": 1, "static": "unchanged"})
    saved = legacy.put(
        config, first, {"source": "loop", "step": 1}, first["channel_versions"]
    )
    legacy.put_writes(saved, [("answer", "saved")], "task")
    snapshot = memory_case.saver()
    restored = snapshot.get_tuple(config)
    assert restored is not None
    assert restored.pending_writes == [("task", "answer", "saved")]

    second = make_checkpoint(2, {"counter": 2, "static": "unchanged"})
    second["channel_versions"]["static"] = "1"
    snapshot.put(saved, second, {"source": "loop", "step": 2}, {"counter": "2"})
    restored = memory_case.saver().get_tuple(config)
    assert restored is not None
    assert restored.checkpoint["channel_values"] == second["channel_values"]
    with pytest.raises(InvalidConfigError, match="checkpoint_format='snapshot'"):
        legacy.get_tuple(config)


def test_delete_thread_removes_pending_sessions_and_preserves_other_threads(
    memory_case: MemoryCase,
) -> None:
    target = memory_case.config("target")
    other = memory_case.config("control")
    writer = memory_case.saver()
    checkpoint = make_checkpoint(1, {"counter": 1})
    saved = writer.put(
        target,
        checkpoint,
        {"source": "loop", "step": 1},
        checkpoint["channel_versions"],
    )
    writer.put_writes(saved, [("answer", 1)], "task")
    orphan_id = uuid4().hex
    orphan: RunnableConfig = {
        "configurable": {**target["configurable"], "checkpoint_id": orphan_id}
    }
    writer.put_writes(orphan, [("orphan", 2)], "task")
    nested: RunnableConfig = {
        "configurable": {**target["configurable"], "checkpoint_ns": "subgraph:one"}
    }
    nested_saved = writer.put(
        nested,
        checkpoint,
        {"source": "loop", "step": 1},
        checkpoint["channel_versions"],
    )
    writer.put_writes(nested_saved, [("nested", 3)], "task")
    writer.put(
        other, checkpoint, {"source": "loop", "step": 1}, checkpoint["channel_versions"]
    )

    writer.delete_thread("target", memory_case.actor)

    assert writer.get_tuple(target) is None
    assert writer.get_tuple(other) is not None
    client = writer.checkpoint_event_client.client
    for checkpoint_id, namespace in [
        (checkpoint["id"], ""),
        (orphan_id, ""),
        (checkpoint["id"], "subgraph:one"),
    ]:
        response = client.list_events(
            memoryId=memory_case.memory_id,
            actorId=memory_case.actor,
            sessionId=writes_session_id("target", checkpoint_id, namespace),
            maxResults=100,
        )
        assert response["events"] == []
