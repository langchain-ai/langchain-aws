import datetime
import os
import random
import string
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint, uuid6
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_aws.checkpoint.agentcore.constants import (
    CheckpointReadLimitError,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver


def generate_valid_session_id():
    """Generate a valid session ID that matches AgentCore pattern [a-zA-Z0-9][a-zA-Z0-9-_]*"""  # noqa: E501
    # Start with letter, then 6 random alphanumeric chars
    chars = string.ascii_letters + string.digits
    return "test" + "".join(random.choices(chars, k=6))


def generate_valid_actor_id():
    """Generate a valid actor ID that matches AgentCore pattern [a-zA-Z0-9][a-zA-Z0-9-_]*"""  # noqa: E501
    # Start with letter, then 6 random alphanumeric chars
    chars = string.ascii_letters + string.digits
    return "actor" + "".join(random.choices(chars, k=6))


@tool
def add(a: int, b: int):
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int):
    """Multiply two integers and return the result."""
    return a * b


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


class TestAgentCoreMemorySaver:
    @pytest.fixture
    def tools(self):
        return [add, multiply, get_weather]

    @pytest.fixture
    def model(self):
        return ChatBedrock(model="us.anthropic.claude-sonnet-4-6", region="us-west-2")

    @pytest.fixture
    def memory_id(self):
        memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
        if not memory_id:
            pytest.skip("AGENTCORE_MEMORY_ID environment variable not set")
        return memory_id

    @pytest.fixture
    def memory_saver(self, memory_id):
        return AgentCoreMemorySaver(memory_id=memory_id, region_name="us-west-2")

    @pytest.fixture
    def boto_agentcore_client(self, memory_saver):
        return memory_saver.checkpoint_event_client.client

    def test_tool_responses(self):
        assert add.invoke({"a": 5, "b": 3}) == 8
        assert multiply.invoke({"a": 4, "b": 6}) == 24
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    def test_checkpoint_save_and_retrieve(self, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": "test_namespace",
            }
        }

        checkpoint = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-2)),
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            channel_values={
                "messages": ["test message"],
                "results": {"status": "completed"},
            },
            channel_versions={"messages": "v1", "results": "v1"},
            versions_seen={"node1": {"messages": "v1"}},
            pending_sends=[],
        )

        checkpoint_metadata = {
            "source": "input",
            "step": 1,
            "writes": {"node1": ["write1", "write2"]},
        }

        try:
            saved_config = memory_saver.put(
                config,
                checkpoint,
                checkpoint_metadata,
                checkpoint["channel_versions"],
            )

            assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
            assert saved_config["configurable"]["thread_id"] == thread_id
            assert saved_config["configurable"]["actor_id"] == actor_id
            assert saved_config["configurable"]["checkpoint_ns"] == "test_namespace"

            checkpoint_tuple = memory_saver.get_tuple(saved_config)
            assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]
            assert (
                checkpoint_tuple.checkpoint["channel_values"]
                == checkpoint["channel_values"]
            )

            # Metadata includes original metadata plus actor_id from config
            expected_metadata = checkpoint_metadata.copy()
            expected_metadata["actor_id"] = actor_id
            assert checkpoint_tuple.metadata == expected_metadata
            assert checkpoint_tuple.config == saved_config

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_math_agent_with_checkpointing(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            response = graph.invoke(
                {
                    "messages": [
                        ("human", "What is 15 times 23? Then add 100 to the result.")
                    ]
                },
                config,
            )
            assert response, "Response should not be empty"
            assert "messages" in response
            assert len(response["messages"]) > 1

            checkpoint = memory_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            checkpoint_tuples = list(memory_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list)

            # Continue conversation to test state persistence
            response2 = graph.invoke(
                {
                    "messages": [
                        (
                            "human",
                            "What was the final result from my previous calculation?",
                        )
                    ]
                },
                config,
            )
            assert response2, "Second response should not be empty"

            # Verify we have more checkpoints after second interaction
            checkpoint_tuples_after = list(memory_saver.list(config))
            assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_weather_query_with_checkpointing(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            response = graph.invoke(
                {"messages": [("human", "What's the weather in sf and nyc?")]}, config
            )
            assert response, "Response should not be empty"

            checkpoint = memory_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            checkpoint_tuples = list(memory_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_multiple_sessions_isolation(self, tools, model, memory_saver):
        thread_id_1 = generate_valid_session_id()
        thread_id_2 = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)

            config_1 = {
                "configurable": {
                    "thread_id": thread_id_1,
                    "actor_id": actor_id,
                }
            }

            config_2 = {
                "configurable": {
                    "thread_id": thread_id_2,
                    "actor_id": actor_id,
                }
            }

            # First session
            response_1 = graph.invoke(
                {"messages": [("human", "Calculate 10 times 5")]}, config_1
            )
            assert response_1, "First session response should not be empty"

            # Second session
            response_2 = graph.invoke(
                {"messages": [("human", "What's the weather in sf?")]}, config_2
            )
            assert response_2, "Second session response should not be empty"

            # Verify sessions are isolated
            checkpoints_1 = list(memory_saver.list(config_1))
            checkpoints_2 = list(memory_saver.list(config_2))

            assert len(checkpoints_1) > 0
            assert len(checkpoints_2) > 0

            # Verify different checkpoint IDs
            checkpoint_ids_1 = {
                cp.config["configurable"]["checkpoint_id"] for cp in checkpoints_1
            }
            checkpoint_ids_2 = {
                cp.config["configurable"]["checkpoint_id"] for cp in checkpoints_2
            }
            assert checkpoint_ids_1.isdisjoint(checkpoint_ids_2)

        finally:
            memory_saver.delete_thread(thread_id_1, actor_id)
            memory_saver.delete_thread(thread_id_2, actor_id)

    def test_checkpoint_listing_with_limit(self, tools, model, memory_saver):
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()

        try:
            graph = create_agent(model, tools=tools, checkpointer=memory_saver)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "actor_id": actor_id,
                }
            }

            # Create multiple interactions to generate several checkpoints
            for i in range(3):
                graph.invoke(
                    {"messages": [("human", f"Calculate {i + 1} times 2")]}, config
                )

            # Test listing with limit
            all_checkpoints = list(memory_saver.list(config))
            limited_checkpoints = list(memory_saver.list(config, limit=2))

            assert len(all_checkpoints) >= 3
            assert len(limited_checkpoints) == 2

            # Verify limited checkpoints are the most recent ones
            assert (
                limited_checkpoints[0].config["configurable"]["checkpoint_id"]
                == all_checkpoints[0].config["configurable"]["checkpoint_id"]
            )
            assert (
                limited_checkpoints[1].config["configurable"]["checkpoint_id"]
                == all_checkpoints[1].config["configurable"]["checkpoint_id"]
            )

        finally:
            memory_saver.delete_thread(thread_id, actor_id)


@dataclass
class MemoryCase:
    memory_id: str
    region: str
    actor: str = field(default_factory=lambda: f"read-safety-{uuid4().hex}")
    threads: set[str] = field(default_factory=set)
    calls: Counter[str] = field(default_factory=Counter)

    def config(self, thread: str = "thread") -> RunnableConfig:
        self.threads.add(thread)
        return {"configurable": {"actor_id": self.actor, "thread_id": thread}}

    def saver(
        self,
        **kwargs: Any,
    ) -> AgentCoreMemorySaver:
        saver = AgentCoreMemorySaver(
            self.memory_id,
            region_name=self.region,
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


class TestAgentCoreReadLimits:
    """Live read regressions using synthetic state without a language model."""

    def test_limit_one_blocks_resume_and_explicit_unlimited_retry_retains_state(
        self,
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
        second = memory_case.saver(limit=1)
        created = memory_case.calls["CreateEvent"]
        with pytest.raises(CheckpointReadLimitError):
            builder.compile(checkpointer=second).invoke(
                {"messages": [HumanMessage("What seat do I prefer?")]}, config
            )
        assert memory_case.calls["CreateEvent"] == created

        result = builder.compile(checkpointer=memory_case.saver()).invoke(
            {"messages": [HumanMessage("What seat do I prefer?")]}, config
        )

        assert len(result["messages"]) == 4
        assert result["messages"][-1].content == "My preferred seat is a window."

    @pytest.mark.parametrize("page_size", [1, None])
    def test_missing_unchanged_channel_fails_but_larger_prefix_succeeds(
        self, memory_case: MemoryCase, page_size: int | None
    ) -> None:
        config = memory_case.config()
        writer = memory_case.saver()
        for index in range(1, 4):
            state = make_checkpoint(index, {"counter": index, "static": "unchanged"})
            state["channel_versions"]["static"] = "2" if index >= 2 else "1"
            writer.put(
                config,
                state,
                {"source": "loop", "step": index},
                state["channel_versions"] if index <= 2 else {"counter": "3"},
            )
        with pytest.raises(CheckpointReadLimitError, match="missing 1 referenced"):
            memory_case.saver(limit=2, max_results=page_size).get_tuple(config)
        # Latest record + changed counter, then counter and static from checkpoint 2.
        with pytest.warns(UserWarning, match="pending writes may be incomplete"):
            restored = memory_case.saver(limit=4, max_results=page_size).get_tuple(
                config
            )
        assert restored is not None
        assert restored.checkpoint["channel_values"] == state["channel_values"]

    def test_complete_channels_do_not_certify_older_pending_writes(
        self,
        memory_case: MemoryCase,
    ) -> None:
        config = memory_case.config()
        writer = memory_case.saver()
        state = make_checkpoint(1, {"counter": 1})
        write_config: RunnableConfig = {
            "configurable": {**config["configurable"], "checkpoint_id": state["id"]}
        }
        writer.put_writes(write_config, [("answer", "saved")], "task")
        writer.put(
            config, state, {"source": "loop", "step": 1}, state["channel_versions"]
        )
        with pytest.warns(UserWarning, match="pending writes may be incomplete"):
            limited = memory_case.saver(limit=2).get_tuple(config)
        assert limited is not None
        assert limited.checkpoint["channel_values"] == state["channel_values"]
        assert limited.pending_writes == []
        full = writer.get_tuple(config)
        assert full is not None
        assert full.pending_writes == [("task", "answer", "saved")]

    def test_history_limit_counts_checkpoints_after_filtering(
        self,
        memory_case: MemoryCase,
    ) -> None:
        config = memory_case.config()
        saver = memory_case.saver()
        saved = []
        for index in range(1, 6):
            checkpoint = make_checkpoint(
                index, {"counter": index, "static": "unchanged"}
            )
            saved.append(
                saver.put(
                    config,
                    checkpoint,
                    {"source": "loop", "step": index},
                    checkpoint["channel_versions"],
                )
            )
        history = list(saver.list(config, limit=3))
        assert [item.checkpoint["channel_values"]["counter"] for item in history] == [
            5,
            4,
            3,
        ]
        filtered = list(
            saver.list(config, limit=1, filter={"step": 3}, before=saved[-1])
        )
        assert len(filtered) == 1
        assert filtered[0].checkpoint["channel_values"]["counter"] == 3
