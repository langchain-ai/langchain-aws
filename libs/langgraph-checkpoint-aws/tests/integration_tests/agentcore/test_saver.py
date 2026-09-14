import datetime
import os
import random
import string
from typing import Literal, TypedDict

import pytest
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.checkpoint.base import Checkpoint, uuid6
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver


class _SubgraphState(TypedDict):
    value: int


def _build_nested_graph(checkpointer, *, subgraph_levels: int):
    """Compile a parent graph containing ``subgraph_levels`` nested subgraphs.

    The innermost subgraph node interrupts, so after invoking, the parent and
    every nested subgraph hold persisted state. ``get_state(subgraphs=True)``
    then walks into each subgraph with a derived config that omits ``actor_id``,
    which is the exact path that regressed in issue #733.

    LangGraph does not provide a prebuilt nested graph, so (as in the other
    checkpointer integration tests) it is assembled here with ``StateGraph``.
    """

    def increment(state: _SubgraphState) -> _SubgraphState:
        interrupt("pause inside subgraph")
        return {"value": state["value"] + 1}

    def single_node_graph(node):
        builder = StateGraph(_SubgraphState)
        builder.add_node("node", node)
        builder.add_edge(START, "node")
        builder.add_edge("node", END)
        return builder

    # Deepest (interrupting) subgraph, then wrap it into progressively higher
    # subgraph levels; the parent finally wraps the stack with the checkpointer.
    node: object = single_node_graph(increment).compile()
    for _ in range(subgraph_levels - 1):
        node = single_node_graph(node).compile()
    return single_node_graph(node).compile(checkpointer=checkpointer)


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
                {"messages": "v2", "results": "v2"},
            )

            assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
            assert saved_config["configurable"]["thread_id"] == thread_id
            assert saved_config["configurable"]["actor_id"] == actor_id
            assert saved_config["configurable"]["checkpoint_ns"] == "test_namespace"

            checkpoint_tuple = memory_saver.get_tuple(saved_config)
            assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

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

    def test_get_state_with_subgraphs_resolves_actor_id(self, memory_saver):
        """get_state(subgraphs=True) works on a nested graph (issue #733)."""
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()
        config = {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}

        try:
            graph = _build_nested_graph(memory_saver, subgraph_levels=1)
            # Runs until the subgraph interrupts, persisting parent + subgraph state.
            graph.invoke({"value": 0}, config)

            # The derived subgraph config drops actor_id; pre-fix this raised
            # InvalidConfigError. It must now resolve the actor from the parent
            # read and return the nested state.
            state = graph.get_state(config, subgraphs=True)

            assert state.tasks, "expected a pending subgraph task"
            subgraph_state = state.tasks[0].state
            assert subgraph_state is not None
            assert "value" in subgraph_state.values

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    def test_get_state_with_deeply_nested_subgraphs_resolves_actor_id(
        self, memory_saver
    ):
        """get_state(subgraphs=True) works across 3 nested subgraph levels."""
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()
        config = {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}

        try:
            graph = _build_nested_graph(memory_saver, subgraph_levels=3)
            graph.invoke({"value": 0}, config)

            # Walk all three nested levels; each derived config omits actor_id and
            # must resolve it from the request rather than raising.
            state = graph.get_state(config, subgraphs=True)
            level_1 = state.tasks[0].state
            level_2 = level_1.tasks[0].state
            level_3 = level_2.tasks[0].state

            assert level_3 is not None
            assert "value" in level_3.values

        finally:
            memory_saver.delete_thread(thread_id, actor_id)

    async def test_aget_state_with_subgraphs_resolves_actor_id(self, memory_saver):
        """aget_state(subgraphs=True) resolves the actor across the executor hop."""
        thread_id = generate_valid_session_id()
        actor_id = generate_valid_actor_id()
        config = {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}

        try:
            graph = _build_nested_graph(memory_saver, subgraph_levels=1)
            await graph.ainvoke({"value": 0}, config)

            # Exercises the async path: the actor captured on the event loop must
            # be copied into the executor thread that runs the sync get_tuple.
            state = await graph.aget_state(config, subgraphs=True)

            assert state.tasks, "expected a pending subgraph task"
            subgraph_state = state.tasks[0].state
            assert subgraph_state is not None
            assert "value" in subgraph_state.values

        finally:
            await memory_saver.adelete_thread(thread_id, actor_id)
