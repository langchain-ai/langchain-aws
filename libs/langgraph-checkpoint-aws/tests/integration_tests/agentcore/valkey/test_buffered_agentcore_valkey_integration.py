"""Integration tests for BufferedCheckpointSaver wrapping AgentCoreValkeySaver."""
import datetime
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.base import Checkpoint, uuid6
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_aws import ChatBedrock

try:
    from valkey import Valkey  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip("Valkey class not available", allow_module_level=True)

from langgraph_checkpoint_aws import AgentCoreValkeySaver, BufferedCheckpointSaver


class _TestSequentialWorkflowState(TypedDict):
    task_a_completed: bool
    task_b_completed: bool
    completed: bool
    step_count: Annotated[int, operator.add]

def _build_sequential_workflow_graph(
    checkpointer: BufferedCheckpointSaver,
    *,
    flush_mid_workflow: bool = False,
    raise_mid_workflow: bool = False,
) -> CompiledStateGraph[_TestSequentialWorkflowState]:
    """Build a 3-step sequential workflow: task_a -> task_b -> finalize."""

    def task_a(state: _TestSequentialWorkflowState) -> dict:
        return {
            "task_a_completed": True,
            "step_count": 1,
        }

    def task_b(state: _TestSequentialWorkflowState) -> dict:
        if raise_mid_workflow:
            raise ValueError("Test error: simulated task failure")
        if flush_mid_workflow:
            checkpointer.flush()
        return {
            "task_b_completed": True,
            "step_count": 1,
        }

    def finalize(state: _TestSequentialWorkflowState) -> dict:
        return {
            "completed": True,
            "step_count": 1,
        }

    graph = StateGraph(_TestSequentialWorkflowState)
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "task_a")
    graph.add_edge("task_a", "task_b")
    graph.add_edge("task_b", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


class _TestParallelGraphState(TypedDict):
    results: Annotated[list, operator.add]
    step_count: Annotated[int, operator.add]
    completed: bool


def _build_parallel_workflow_graph(
    checkpointer: BufferedCheckpointSaver,
    *,
    flush_mid_workflow: bool = False,
    raise_mid_workflow: bool = False,
) -> CompiledStateGraph[_TestParallelGraphState]:
    """Build a parallel workflow: (task_a, task_b) -> merge."""

    def task_a(state: _TestParallelGraphState) -> dict:
        return {
            "results": ["result_a"],
            "step_count": 1,
        }

    def task_b(state: _TestParallelGraphState) -> dict:
        return {
            "results": ["result_b"],
            "step_count": 1,
        }

    def merge_step(state: _TestParallelGraphState) -> dict:
        if raise_mid_workflow:
            raise ValueError("Simulated failure for testing")
        if flush_mid_workflow:
            checkpointer.flush()
        return {
            "completed": True,
            "step_count": 1,
        }

    graph = StateGraph(_TestParallelGraphState)
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("merge", merge_step)

    graph.add_edge(START, "task_a")
    graph.add_edge(START, "task_b")
    graph.add_edge("task_a", "merge")
    graph.add_edge("task_b", "merge")
    graph.add_edge("merge", END)

    return graph.compile(checkpointer=checkpointer)


class TestBufferedAgentCoreValkeySaverIntegrationSync:

    def test_flush_on_exit_with_seq_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test that the BufferedCheckpointSaver flushes on exit."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            graph.invoke({"step_count": 0}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        assert len(list(agentcore_valkey_saver.list(config))) == 1

    def test_flush_on_exit_with_seq_workflow_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            for _ in graph.stream({"step_count": 0}, config):
                assert not buffered_agentcore_valkey_saver.is_empty
                assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                assert buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

    def test_flush_on_exit_with_parallel_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            graph.invoke({"step_count": 0, "results": []}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}
        assert len(list(agentcore_valkey_saver.list(config))) == 1

    def test_flush_on_exit_with_parallel_workflow_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            for _ in graph.stream({"step_count": 0, "results": []}, config):
                assert not buffered_agentcore_valkey_saver.is_empty
                assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                assert buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}

    def test_flush_mid_seq_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            graph.invoke({"step_count": 0}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        # One from mid-workflow flush, one from flush_on_exit
        assert len(list(agentcore_valkey_saver.list(config))) == 2

    def test_flush_mid_seq_workflow_with_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            streamed_nodes = []
            for event in graph.stream({"step_count": 0}, config):
                node_name = list(event.keys())[0]
                streamed_nodes.append(node_name)

                if node_name == "task_a":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                    assert agentcore_valkey_saver.get_tuple(config) is None

                elif node_name == "task_b":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

                    # Mid-workflow flush persisted state BEFORE task_b completed
                    mid_flush_persisted = agentcore_valkey_saver.get_tuple(config)
                    assert mid_flush_persisted is not None
                    assert mid_flush_persisted.checkpoint["channel_values"] == {
                        "task_a_completed": True,
                        "task_b_completed": False,
                        "completed": False,
                        "step_count": 1,
                    }

                elif node_name == "finalize":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

            assert streamed_nodes == ["task_a", "task_b", "finalize"]
            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        # One from mid-workflow flush, one from flush_on_exit
        assert len(list(agentcore_valkey_saver.list(config))) == 2

    def test_flush_mid_parallel_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            graph.invoke({"step_count": 0, "results": []}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}

        # One from mid-workflow flush, one from flush_on_exit
        assert len(list(agentcore_valkey_saver.list(config))) == 2

    def test_flush_mid_parallel_workflow_with_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.flush_on_exit():
            streamed_nodes = []
            for event in graph.stream({"step_count": 0, "results": []}, config):
                node_name = list(event.keys())[0]
                streamed_nodes.append(node_name)

                if node_name in ("task_a", "task_b"):
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                    assert agentcore_valkey_saver.get_tuple(config) is None

                elif node_name == "merge":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

                    # Mid-workflow flush persisted state BEFORE merge completed
                    mid_flush_persisted = agentcore_valkey_saver.get_tuple(config)
                    assert mid_flush_persisted is not None
                    mid_state = mid_flush_persisted.checkpoint["channel_values"]
                    assert mid_state["step_count"] == 2
                    assert mid_state["completed"] is False
                    assert set(mid_state["results"]) == {"result_a", "result_b"}

            assert set(streamed_nodes) == {"task_a", "task_b", "merge"}
            assert streamed_nodes[-1] == "merge"
            assert graph.get_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}
        assert len(list(agentcore_valkey_saver.list(config))) == 2

    def test_context_manager_flushes_on_exception_and_propagates(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test that a buffered checkpoint saver flushes state even when an exception occurs within a context manager."""

        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            raise_mid_workflow=True,
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with pytest.raises(ValueError, match="Test error: simulated task failure"):
            with buffered_agentcore_valkey_saver.flush_on_exit():
                graph.invoke({"step_count": 0}, config)

        assert buffered_agentcore_valkey_saver.is_empty
        assert not buffered_agentcore_valkey_saver.has_buffered_checkpoint
        assert not buffered_agentcore_valkey_saver.has_buffered_writes

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": False,
            "completed": False,
            "step_count": 1,
        }

    def test_manual_flush_without_context_manager(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        result = graph.invoke({"step_count": 0}, config)
        assert result["completed"] is True
        assert not buffered_agentcore_valkey_saver.is_empty
        assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
        assert agentcore_valkey_saver.get_tuple(config) is None

        flush_result = buffered_agentcore_valkey_saver.flush()
        assert flush_result is not None
        assert buffered_agentcore_valkey_saver.is_empty

        persisted = agentcore_valkey_saver.get_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

    def test_state_persistence_across_invocations(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        with buffered_agentcore_valkey_saver.flush_on_exit():
            result1 = graph.invoke({"step_count": 0}, config)
            assert result1["step_count"] == 3

        assert len(list(agentcore_valkey_saver.list(config))) == 1

        with buffered_agentcore_valkey_saver.flush_on_exit():
            result2 = graph.invoke(None, config)
            assert result2["step_count"] == 6

        assert len(list(agentcore_valkey_saver.list(config))) == 2

    def test_multiple_sessions_isolation(
        self,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id_1 = agentcore_session_id + "-1"
        thread_id_2 = agentcore_session_id + "-2"
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)

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

        with buffered_agentcore_valkey_saver.flush_on_exit():
            result_1 = graph.invoke({"step_count": 0}, config_1)

        assert result_1["step_count"] == 3
        assert result_1["completed"] is True

        with buffered_agentcore_valkey_saver.flush_on_exit():
            result_2 = graph.invoke({"step_count": 0}, config_2)

        assert result_2["step_count"] == 3
        assert result_2["completed"] is True

        checkpoints_1 = list(agentcore_valkey_saver.list(config_1))
        checkpoints_2 = list(agentcore_valkey_saver.list(config_2))
        assert len(checkpoints_1) == 1
        assert len(checkpoints_2) == 1

        checkpoint_id_1 = checkpoints_1[0].config["configurable"]["checkpoint_id"]
        checkpoint_id_2 = checkpoints_2[0].config["configurable"]["checkpoint_id"]
        assert checkpoint_id_1 != checkpoint_id_2

        assert checkpoints_1[0].config["configurable"]["thread_id"] == thread_id_1
        assert checkpoints_2[0].config["configurable"]["thread_id"] == thread_id_2
        assert checkpoints_1[0].checkpoint["channel_values"]["step_count"] == 3
        assert checkpoints_2[0].checkpoint["channel_values"]["step_count"] == 3

    def test_checkpoint_listing_with_limit(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        num_invocations = 3
        for i in range(num_invocations):
            with buffered_agentcore_valkey_saver.flush_on_exit():
                if i == 0:
                    result = graph.invoke({"step_count": 0}, config)
                else:
                    result = graph.invoke(None, config)
            assert result["step_count"] == (i + 1) * 3

        all_checkpoints = list(agentcore_valkey_saver.list(config))
        assert len(all_checkpoints) == num_invocations

        limit = 2
        limited_checkpoints = list(agentcore_valkey_saver.list(config, limit=limit))
        assert len(limited_checkpoints) == limit

        # Limited results should match the first N from the full list
        for i in range(limit):
            limited_id = limited_checkpoints[i].config["configurable"]["checkpoint_id"]
            full_id = all_checkpoints[i].config["configurable"]["checkpoint_id"]
            assert limited_id == full_id

        # Most recent checkpoint should have highest step_count
        most_recent = all_checkpoints[0].checkpoint["channel_values"]["step_count"]
        oldest = all_checkpoints[-1].checkpoint["channel_values"]["step_count"]
        assert most_recent > oldest

    def test_checkpoint_save_and_retrieve(
        self,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

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

        with buffered_agentcore_valkey_saver.flush_on_exit():
            saved_config = buffered_agentcore_valkey_saver.put(
                config,
                checkpoint,
                checkpoint_metadata,
                {"messages": "v2", "results": "v2"},
            )

            assert (
                saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
            )
            assert saved_config["configurable"]["thread_id"] == thread_id
            assert saved_config["configurable"]["actor_id"] == actor_id
            assert (
                saved_config["configurable"]["checkpoint_ns"] == "test_namespace"
            )

            checkpoint_tuple = buffered_agentcore_valkey_saver.get_tuple(
                saved_config
            )
            assert checkpoint_tuple is not None
            assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

        persisted_tuple = agentcore_valkey_saver.get_tuple(saved_config)
        assert persisted_tuple is not None
        assert persisted_tuple.checkpoint["id"] == checkpoint["id"]

        expected_metadata = checkpoint_metadata.copy()
        expected_metadata["actor_id"] = actor_id
        assert persisted_tuple.metadata == expected_metadata

    def test_math_agent_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        with buffered_agentcore_valkey_saver.flush_on_exit():
            response = graph.invoke(
                {
                    "messages": [
                        (
                            "human",
                            "What is 15 times 23? Then add 100 to the result.",
                        )
                    ]
                },
                config,
            )
            assert response
            assert "messages" in response
            assert len(response["messages"]) > 1

            checkpoint = buffered_agentcore_valkey_saver.get(config)
            assert checkpoint

        persisted_checkpoint = agentcore_valkey_saver.get(config)
        assert persisted_checkpoint

        checkpoint_tuples = list(agentcore_valkey_saver.list(config))
        assert checkpoint_tuples
        assert len(checkpoint_tuples) == 1

        with buffered_agentcore_valkey_saver.flush_on_exit():
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
            assert response2

        checkpoint_tuples_after = list(agentcore_valkey_saver.list(config))
        assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

    def test_weather_query_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        with buffered_agentcore_valkey_saver.flush_on_exit():
            response = graph.invoke(
                {"messages": [("human", "What's the weather in sf and nyc?")]},
                config,
            )
            assert response

            checkpoint = buffered_agentcore_valkey_saver.get(config)
            assert checkpoint

        persisted_checkpoint = agentcore_valkey_saver.get(config)
        assert persisted_checkpoint

        checkpoint_tuples = list(agentcore_valkey_saver.list(config))
        assert checkpoint_tuples
        assert len(checkpoint_tuples) == 1

    def test_checkpoint_listing_with_limit(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        for i in range(3):
            with buffered_agentcore_valkey_saver.flush_on_exit():
                graph.invoke(
                    {"messages": [("human", f"Calculate {i + 1} times 2")]}, config
                )

        all_checkpoints = list(agentcore_valkey_saver.list(config))
        limited_checkpoints = list(agentcore_valkey_saver.list(config, limit=2))

        assert len(all_checkpoints) >= 3
        assert len(limited_checkpoints) == 2

        assert (
            limited_checkpoints[0].config["configurable"]["checkpoint_id"]
            == all_checkpoints[0].config["configurable"]["checkpoint_id"]
        )
        assert (
            limited_checkpoints[1].config["configurable"]["checkpoint_id"]
            == all_checkpoints[1].config["configurable"]["checkpoint_id"]
        )

    def test_writes_storage(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test storing and retrieving writes."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        with buffered_agentcore_valkey_saver.flush_on_exit():
            # First create a checkpoint
            checkpoint = Checkpoint(
                v=1,
                id=str(uuid6(clock_seq=-2)),
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                channel_values={"messages": []},
                channel_versions={"messages": "v1"},
                versions_seen={},
                pending_sends=[],
            )

            metadata = {"user": "test"}
            new_versions = {"messages": "1.0"}

            result_config = buffered_agentcore_valkey_saver.put(
                config, checkpoint, metadata, new_versions
            )

            # Add writes
            writes = [
                ("messages", {"role": "assistant", "content": "Response 1"}),
                ("messages", {"role": "assistant", "content": "Response 2"}),
            ]

            buffered_agentcore_valkey_saver.put_writes(
                result_config, writes, "task-1"
            )

            # Retrieve checkpoint with writes from buffer
            retrieved = buffered_agentcore_valkey_saver.get_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.pending_writes) == 2

            # Check write content
            assert retrieved.pending_writes[0][0] == "task-1"  # task_id
            assert retrieved.pending_writes[0][1] == "messages"  # channel
            assert retrieved.pending_writes[0][2]["role"] == "assistant"  # value

            assert retrieved.pending_writes[1][2]["content"] == "Response 2"

        # Verify writes were persisted
        persisted = agentcore_valkey_saver.get_tuple(result_config)
        assert persisted is not None
        assert len(persisted.pending_writes) == 2

    def test_metadata_filtering(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test filtering checkpoints by metadata."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        # Create checkpoints with different metadata
        for i in range(3):
            with buffered_agentcore_valkey_saver.flush_on_exit():
                checkpoint = Checkpoint(
                    v=1,
                    id=str(uuid6(clock_seq=-2)),
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    channel_values={"messages": []},
                    channel_versions={"messages": "v1"},
                    versions_seen={},
                    pending_sends=[],
                )

                metadata = {
                    "user": "test_user" if i % 2 == 0 else "other_user",
                    "step": i,
                }
                new_versions = {"messages": f"{i + 1}.0"}

                buffered_agentcore_valkey_saver.put(
                    config, checkpoint, metadata, new_versions
                )

        # Filter by metadata
        filtered_checkpoints = list(
            agentcore_valkey_saver.list(config, filter={"user": "test_user"})
        )

        # Should only get checkpoints for test_user (indices 0 and 2)
        assert len(filtered_checkpoints) == 2

        for checkpoint_tuple in filtered_checkpoints:
            assert checkpoint_tuple.metadata["user"] == "test_user"

    def test_thread_deletion(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test deleting all data for a thread."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        # Create multiple checkpoints and writes
        for i in range(2):
            with buffered_agentcore_valkey_saver.flush_on_exit():
                checkpoint = Checkpoint(
                    v=1,
                    id=str(uuid6(clock_seq=-2)),
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    channel_values={"messages": []},
                    channel_versions={"messages": "v1"},
                    versions_seen={},
                    pending_sends=[],
                )

                metadata = {"step": i}
                new_versions = {"messages": f"{i + 1}.0"}

                result_config = buffered_agentcore_valkey_saver.put(
                    config, checkpoint, metadata, new_versions
                )

                # Add writes
                writes = [("messages", {"content": f"write-{i}"})]
                buffered_agentcore_valkey_saver.put_writes(
                    result_config, writes, f"task-{i}"
                )

        # Verify data exists
        checkpoints = list(agentcore_valkey_saver.list(config))
        assert len(checkpoints) == 2

        # Delete thread
        agentcore_valkey_saver.delete_thread(thread_id, actor_id)

        # Verify all data is deleted
        checkpoints_after = list(agentcore_valkey_saver.list(config))
        assert len(checkpoints_after) == 0

        # Try to get latest checkpoint
        latest = agentcore_valkey_saver.get_tuple(config)
        assert latest is None

    def test_concurrent_operations(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test concurrent operations on the same session."""
        import threading

        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        results = []
        errors = []

        def create_checkpoint(index):
            try:
                # Each thread gets its own buffered saver to avoid conflicts
                thread_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
                with thread_saver.flush_on_exit():
                    checkpoint = Checkpoint(
                        v=1,
                        id=str(uuid6(clock_seq=-2)),
                        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        channel_values={
                            "messages": [{"content": f"msg-{index}"}]
                        },
                        channel_versions={"messages": "v1"},
                        versions_seen={},
                        pending_sends=[],
                    )

                    metadata = {"thread_index": index}
                    new_versions = {"messages": f"{index + 1}.0"}

                    result_config = thread_saver.put(
                        config, checkpoint, metadata, new_versions
                    )
                    results.append(result_config)
            except Exception as e:
                errors.append(e)

        # Create multiple checkpoints concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_checkpoint, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all checkpoints were created
        checkpoints = list(agentcore_valkey_saver.list(config))
        assert len(checkpoints) == 5

        # Verify each checkpoint is unique
        checkpoint_ids = {cp.checkpoint["id"] for cp in checkpoints}
        assert len(checkpoint_ids) == 5

    def test_large_checkpoint_data(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test handling of large checkpoint data."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        with buffered_agentcore_valkey_saver.flush_on_exit():
            # Create a large checkpoint
            large_messages = []
            for i in range(100):
                large_messages.append(
                    {
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": f"This is message number {i} " * 50,
                        "metadata": {"index": i, "data": list(range(50))},
                    }
                )

            checkpoint = Checkpoint(
                v=1,
                id=str(uuid6(clock_seq=-2)),
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                channel_values={
                    "messages": large_messages,
                    "context": {
                        "large_array": list(range(1000)),
                    },
                },
                channel_versions={"messages": "v1", "context": "v1"},
                versions_seen={},
                pending_sends=[],
            )

            metadata = {"size": "large", "message_count": len(large_messages)}
            new_versions = {"messages": "1.0", "context": "1.0"}

            # Store large checkpoint
            result_config = buffered_agentcore_valkey_saver.put(
                config, checkpoint, metadata, new_versions
            )

            # Retrieve from buffer and verify
            retrieved = buffered_agentcore_valkey_saver.get_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.checkpoint["channel_values"]["messages"]) == 100
            assert (
                len(retrieved.checkpoint["channel_values"]["context"]["large_array"])
                == 1000
            )
            assert retrieved.metadata["message_count"] == 100

        # Verify persisted data
        persisted = agentcore_valkey_saver.get_tuple(result_config)
        assert persisted is not None
        assert len(persisted.checkpoint["channel_values"]["messages"]) == 100
        assert (
            len(persisted.checkpoint["channel_values"]["context"]["large_array"])
            == 1000
        )


class TestBufferedAgentCoreValkeySaverIntegrationAsync:

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_seq_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test that the BufferedCheckpointSaver flushes on exit."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            await graph.ainvoke({"step_count": 0}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        assert len(await list(agentcore_valkey_saver.alist(config))) == 1

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_seq_workflow_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            async for _ in graph.astream({"step_count": 0}, config):
                assert not buffered_agentcore_valkey_saver.is_empty
                assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                assert buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_parallel_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        
        graph = _build_parallel_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            await graph.ainvoke({"step_count": 0, "results": []}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}
        assert len(list(agentcore_valkey_saver.list(config))) == 1

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_parallel_workflow_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        
        graph = _build_parallel_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            async for _ in graph.astream({"step_count": 0, "results": []}, config):
                assert not buffered_agentcore_valkey_saver.is_empty
                assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                assert buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}

    @pytest.mark.asyncio
    async def test_async_flush_mid_seq_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with buffered_agentcore_valkey_saver.aflush_on_exit():
            await graph.ainvoke({"step_count": 0}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        # One from mid-workflow flush, one from flush_on_exit
        assert len(await list(agentcore_valkey_saver.alist(config))) == 2

    @pytest.mark.asyncio
    async def test_async_flush_mid_seq_workflow_with_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            streamed_nodes = []
            async for event in graph.astream({"step_count": 0}, config):
                node_name = list(event.keys())[0]
                streamed_nodes.append(node_name)

                if node_name == "task_a":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                    assert await agentcore_valkey_saver.aget_tuple(config) is None

                elif node_name == "task_b":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

                    # Mid-workflow flush persisted state BEFORE task_b completed
                    mid_flush_persisted = await agentcore_valkey_saver.aget_tuple(config)
                    assert mid_flush_persisted is not None
                    assert mid_flush_persisted.checkpoint["channel_values"] == {
                        "task_a_completed": True,
                        "task_b_completed": False,
                        "completed": False,
                        "step_count": 1,
                    }

                elif node_name == "finalize":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

            assert streamed_nodes == ["task_a", "task_b", "finalize"]
            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }

        # One from mid-workflow flush, one from flush_on_exit
        assert len(await list(agentcore_valkey_saver.alist(config))) == 2

    @pytest.mark.asyncio
    async def test_async_flush_mid_parallel_workflow(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            await graph.ainvoke({"step_count": 0, "results": []}, config)

            assert not buffered_agentcore_valkey_saver.is_empty
            assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
            assert not buffered_agentcore_valkey_saver.has_buffered_writes

            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}

        # One from mid-workflow flush, one from flush_on_exit
        assert len(await list(agentcore_valkey_saver.alist(config))) == 2

    @pytest.mark.asyncio
    async def test_async_flush_mid_parallel_workflow_with_streaming(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_parallel_workflow_graph(
            buffered_agentcore_valkey_saver,
            flush_mid_workflow=True,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            streamed_nodes = []
            async for event in graph.astream({"step_count": 0, "results": []}, config):
                node_name = list(event.keys())[0]
                streamed_nodes.append(node_name)

                if node_name in ("task_a", "task_b"):
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
                    assert await agentcore_valkey_saver.aget_tuple(config) is None

                elif node_name == "merge":
                    assert not buffered_agentcore_valkey_saver.is_empty
                    assert buffered_agentcore_valkey_saver.has_buffered_checkpoint

                    # Mid-workflow flush persisted state BEFORE merge completed
                    mid_flush_persisted = await agentcore_valkey_saver.aget_tuple(config)
                    assert mid_flush_persisted is not None
                    mid_state = mid_flush_persisted.checkpoint["channel_values"]
                    assert mid_state["step_count"] == 2
                    assert mid_state["completed"] is False
                    assert set(mid_state["results"]) == {"result_a", "result_b"}

            assert set(streamed_nodes) == {"task_a", "task_b", "merge"}
            assert streamed_nodes[-1] == "merge"
            assert await graph.aget_state(config) is not None

        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None

        channel_values = persisted.checkpoint["channel_values"]
        assert channel_values["step_count"] == 3
        assert channel_values["completed"] is True
        assert set(channel_values["results"]) == {"result_a", "result_b"}
        assert len(await list(agentcore_valkey_saver.alist(config))) == 2

    @pytest.mark.asyncio
    async def test_async_context_manager_flushes_on_exception_and_propagates(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test that a buffered checkpoint saver flushes state even when an exception occurs within a context manager."""

        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
            raise_mid_workflow=True,
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        with pytest.raises(ValueError, match="Test error: simulated task failure"):
            async with buffered_agentcore_valkey_saver.aflush_on_exit():
                await graph.ainvoke({"step_count": 0}, config)

        assert buffered_agentcore_valkey_saver.is_empty
        assert not buffered_agentcore_valkey_saver.has_buffered_checkpoint
        assert not buffered_agentcore_valkey_saver.has_buffered_writes

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": False,
            "completed": False,
            "step_count": 1,
        }

    @pytest.mark.asyncio
    async def test_async_manual_flush_without_context_manager(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(
            buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        assert buffered_agentcore_valkey_saver.is_empty

        result = await graph.ainvoke({"step_count": 0}, config)
        assert result["completed"] is True
        assert not buffered_agentcore_valkey_saver.is_empty
        assert buffered_agentcore_valkey_saver.has_buffered_checkpoint
        assert await agentcore_valkey_saver.aget_tuple(config) is None

        flush_result = await buffered_agentcore_valkey_saver.aflush()
        assert flush_result is not None
        assert buffered_agentcore_valkey_saver.is_empty

        persisted = await agentcore_valkey_saver.aget_tuple(config)
        assert persisted is not None
        assert persisted.checkpoint is not None
        assert persisted.checkpoint["channel_values"] == {
            "task_a_completed": True,
            "task_b_completed": True,
            "completed": True,
            "step_count": 3,
        }


    @pytest.mark.asyncio
    async def test_async_state_persistence_across_invocations(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            result1 = await graph.ainvoke({"step_count": 0}, config)
            assert result1["step_count"] == 3

        assert len(list(agentcore_valkey_saver.list(config))) == 1

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            result2 = await graph.ainvoke(None, config)
            assert result2["step_count"] == 6

        assert len(await list(agentcore_valkey_saver.alist(config))) == 2

    @pytest.mark.asyncio
    async def test_async_multiple_sessions_isolation(
        self,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id_1 = agentcore_session_id + "-1"
        thread_id_2 = agentcore_session_id + "-2"
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)

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

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            result_1 = await graph.ainvoke({"step_count": 0}, config_1)

        assert result_1["step_count"] == 3
        assert result_1["completed"] is True

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            result_2 = await graph.ainvoke({"step_count": 0}, config_2)

        assert result_2["step_count"] == 3
        assert result_2["completed"] is True

        checkpoints_1 = await list(agentcore_valkey_saver.alist(config_1))
        checkpoints_2 = await list(agentcore_valkey_saver.alist(config_2))
        assert len(checkpoints_1) == 1
        assert len(checkpoints_2) == 1

        checkpoint_id_1 = checkpoints_1[0].config["configurable"]["checkpoint_id"]
        checkpoint_id_2 = checkpoints_2[0].config["configurable"]["checkpoint_id"]
        assert checkpoint_id_1 != checkpoint_id_2

        assert checkpoints_1[0].config["configurable"]["thread_id"] == thread_id_1
        assert checkpoints_2[0].config["configurable"]["thread_id"] == thread_id_2
        assert checkpoints_1[0].checkpoint["channel_values"]["step_count"] == 3
        assert checkpoints_2[0].checkpoint["channel_values"]["step_count"] == 3

    @pytest.mark.asyncio
    async def test_async_checkpoint_listing_with_limit(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id
        
        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
        graph = _build_sequential_workflow_graph(buffered_agentcore_valkey_saver)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        num_invocations = 3
        for i in range(num_invocations):
            async with buffered_agentcore_valkey_saver.aflush_on_exit():
                if i == 0:
                    result = await graph.ainvoke({"step_count": 0}, config)
                else:
                    result = await graph.ainvoke(None, config)
            assert result["step_count"] == (i + 1) * 3

        all_checkpoints = await list(agentcore_valkey_saver.alist(config))
        assert len(all_checkpoints) == num_invocations

        limit = 2
        limited_checkpoints = await list(agentcore_valkey_saver.alist(config, limit=limit))
        assert len(limited_checkpoints) == limit

        # Limited results should match the first N from the full list
        for i in range(limit):
            limited_id = limited_checkpoints[i].config["configurable"]["checkpoint_id"]
            full_id = all_checkpoints[i].config["configurable"]["checkpoint_id"]
            assert limited_id == full_id

        # Most recent checkpoint should have highest step_count
        most_recent = all_checkpoints[0].checkpoint["channel_values"]["step_count"]
        oldest = all_checkpoints[-1].checkpoint["channel_values"]["step_count"]
        assert most_recent > oldest

    @pytest.mark.asyncio
    async def test_async_checkpoint_save_and_retrieve(
        self,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

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

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            saved_config = await buffered_agentcore_valkey_saver.aput(
                config,
                checkpoint,
                checkpoint_metadata,
                {"messages": "v2", "results": "v2"},
            )

            assert (
                saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
            )
            assert saved_config["configurable"]["thread_id"] == thread_id
            assert saved_config["configurable"]["actor_id"] == actor_id
            assert (
                saved_config["configurable"]["checkpoint_ns"] == "test_namespace"
            )

            checkpoint_tuple = await buffered_agentcore_valkey_saver.aget_tuple(
                saved_config
            )
            assert checkpoint_tuple is not None
            assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

        persisted_tuple = await agentcore_valkey_saver.aget_tuple(saved_config)
        assert persisted_tuple is not None
        assert persisted_tuple.checkpoint["id"] == checkpoint["id"]

        expected_metadata = checkpoint_metadata.copy()
        expected_metadata["actor_id"] = actor_id
        assert persisted_tuple.metadata == expected_metadata

    @pytest.mark.asyncio
    async def test_async_math_agent_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            response = await graph.ainvoke(
                {
                    "messages": [
                        (
                            "human",
                            "What is 15 times 23? Then add 100 to the result.",
                        )
                    ]
                },
                config,
            )
            assert response
            assert "messages" in response
            assert len(response["messages"]) > 1

            checkpoint = await buffered_agentcore_valkey_saver.aget(config)
            assert checkpoint

        persisted_checkpoint = await agentcore_valkey_saver.aget(config)
        assert persisted_checkpoint

        checkpoint_tuples = list(await agentcore_valkey_saver.alist(config))
        assert checkpoint_tuples
        assert len(checkpoint_tuples) == 1

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            response2 = await graph.ainvoke(
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
            assert response2

        checkpoint_tuples_after = await list(agentcore_valkey_saver.alist(config))
        assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

    @pytest.mark.asyncio
    async def test_async_weather_query_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            response = await graph.ainvoke(
                {"messages": [("human", "What's the weather in sf and nyc?")]},
                config,
            )
            assert response

            checkpoint = await buffered_agentcore_valkey_saver.aget(config)
            assert checkpoint

        persisted_checkpoint = await agentcore_valkey_saver.aget(config)
        assert persisted_checkpoint

        checkpoint_tuples = await list(agentcore_valkey_saver.alist(config))
        assert checkpoint_tuples
        assert len(checkpoint_tuples) == 1

    @pytest.mark.asyncio
    async def test_async_checkpoint_listing_with_limit(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        agentcore_actor_id: str,
        agentcore_valkey_saver: AgentCoreValkeySaver,
    ):
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        graph = create_agent(
            bedrock_model,
            tools=agent_tools,
            checkpointer=buffered_agentcore_valkey_saver,
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        for i in range(3):
            async with buffered_agentcore_valkey_saver.aflush_on_exit():
                await graph.ainvoke(
                    {"messages": [("human", f"Calculate {i + 1} times 2")]}, config
                )

        all_checkpoints = await list(agentcore_valkey_saver.alist(config))
        limited_checkpoints = await list(agentcore_valkey_saver.alist(config, limit=2))

        assert len(all_checkpoints) >= 3
        assert len(limited_checkpoints) == 2

        assert (
            limited_checkpoints[0].config["configurable"]["checkpoint_id"]
            == all_checkpoints[0].config["configurable"]["checkpoint_id"]
        )
        assert (
            limited_checkpoints[1].config["configurable"]["checkpoint_id"]
            == all_checkpoints[1].config["configurable"]["checkpoint_id"]
        )

    @pytest.mark.asyncio
    async def test_async_writes_storage(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test storing and retrieving writes asynchronously."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            # First create a checkpoint
            checkpoint = Checkpoint(
                v=1,
                id=str(uuid6(clock_seq=-2)),
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                channel_values={"messages": []},
                channel_versions={"messages": "v1"},
                versions_seen={},
                pending_sends=[],
            )

            metadata = {"user": "test"}
            new_versions = {"messages": "1.0"}

            result_config = await buffered_agentcore_valkey_saver.aput(
                config, checkpoint, metadata, new_versions
            )

            # Add writes
            writes = [
                ("messages", {"role": "assistant", "content": "Response 1"}),
                ("messages", {"role": "assistant", "content": "Response 2"}),
            ]

            await buffered_agentcore_valkey_saver.aput_writes(
                result_config, writes, "task-1"
            )

            # Retrieve checkpoint with writes from buffer
            retrieved = await buffered_agentcore_valkey_saver.aget_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.pending_writes) == 2

            # Check write content
            assert retrieved.pending_writes[0][0] == "task-1"  # task_id
            assert retrieved.pending_writes[0][1] == "messages"  # channel
            assert retrieved.pending_writes[0][2]["role"] == "assistant"  # value

            assert retrieved.pending_writes[1][2]["content"] == "Response 2"

        # Verify writes were persisted
        persisted = await agentcore_valkey_saver.aget_tuple(result_config)
        assert persisted is not None
        assert len(persisted.pending_writes) == 2

    @pytest.mark.asyncio
    async def test_async_metadata_filtering(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test filtering checkpoints by metadata asynchronously."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        # Create checkpoints with different metadata
        for i in range(3):
            async with buffered_agentcore_valkey_saver.aflush_on_exit():
                checkpoint = Checkpoint(
                    v=1,
                    id=str(uuid6(clock_seq=-2)),
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    channel_values={"messages": []},
                    channel_versions={"messages": "v1"},
                    versions_seen={},
                    pending_sends=[],
                )

                metadata = {
                    "user": "test_user" if i % 2 == 0 else "other_user",
                    "step": i,
                }
                new_versions = {"messages": f"{i + 1}.0"}

                await buffered_agentcore_valkey_saver.aput(
                    config, checkpoint, metadata, new_versions
                )

        # Filter by metadata
        filtered_checkpoints = []
        async for cp in agentcore_valkey_saver.alist(
            config, filter={"user": "test_user"}
        ):
            filtered_checkpoints.append(cp)

        # Should only get checkpoints for test_user (indices 0 and 2)
        assert len(filtered_checkpoints) == 2

        for checkpoint_tuple in filtered_checkpoints:
            assert checkpoint_tuple.metadata["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_async_thread_deletion(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test deleting all data for a thread asynchronously."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        buffered_agentcore_valkey_saver = BufferedCheckpointSaver(agentcore_valkey_saver)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        # Create multiple checkpoints and writes
        for i in range(2):
            async with buffered_agentcore_valkey_saver.aflush_on_exit():
                checkpoint = Checkpoint(
                    v=1,
                    id=str(uuid6(clock_seq=-2)),
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    channel_values={"messages": []},
                    channel_versions={"messages": "v1"},
                    versions_seen={},
                    pending_sends=[],
                )

                metadata = {"step": i}
                new_versions = {"messages": f"{i + 1}.0"}

                result_config = await buffered_agentcore_valkey_saver.aput(
                    config, checkpoint, metadata, new_versions
                )

                # Add writes
                writes = [("messages", {"content": f"write-{i}"})]
                await buffered_agentcore_valkey_saver.aput_writes(
                    result_config, writes, f"task-{i}"
                )

        # Verify data exists
        checkpoints = []
        async for cp in agentcore_valkey_saver.alist(config):
            checkpoints.append(cp)
        assert len(checkpoints) == 2

        # Delete thread
        await agentcore_valkey_saver.adelete_thread(thread_id, actor_id)

        # Verify all data is deleted
        checkpoints_after = []
        async for cp in agentcore_valkey_saver.alist(config):
            checkpoints_after.append(cp)
        assert len(checkpoints_after) == 0

        # Try to get latest checkpoint
        latest = await agentcore_valkey_saver.aget_tuple(config)
        assert latest is None

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test concurrent async operations on the same session."""
        import asyncio

        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        results = []
        errors = []

        async def create_checkpoint(index):
            try:
                # Each task gets its own buffered saver to avoid conflicts
                task_saver = BufferedCheckpointSaver(agentcore_valkey_saver)
                async with task_saver.aflush_on_exit():
                    checkpoint = Checkpoint(
                        v=1,
                        id=str(uuid6(clock_seq=-2)),
                        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        channel_values={
                            "messages": [{"content": f"msg-{index}"}]
                        },
                        channel_versions={"messages": "v1"},
                        versions_seen={},
                        pending_sends=[],
                    )

                    metadata = {"task_index": index}
                    new_versions = {"messages": f"{index + 1}.0"}

                    result_config = await task_saver.aput(
                        config, checkpoint, metadata, new_versions
                    )
                    results.append(result_config)
            except Exception as e:
                errors.append(e)

        # Create multiple checkpoints concurrently
        tasks = [create_checkpoint(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all checkpoints were created
        checkpoints = []
        async for cp in agentcore_valkey_saver.alist(config):
            checkpoints.append(cp)
        assert len(checkpoints) == 5

        # Verify each checkpoint is unique
        checkpoint_ids = {cp.checkpoint["id"] for cp in checkpoints}
        assert len(checkpoint_ids) == 5

    @pytest.mark.asyncio
    async def test_async_large_checkpoint_data(
        self,
        agentcore_valkey_saver: AgentCoreValkeySaver,
        buffered_agentcore_valkey_saver: BufferedCheckpointSaver,
        agentcore_session_id: str,
        agentcore_actor_id: str,
    ):
        """Test handling of large checkpoint data asynchronously."""
        thread_id = agentcore_session_id
        actor_id = agentcore_actor_id

        config = {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
            }
        }

        async with buffered_agentcore_valkey_saver.aflush_on_exit():
            # Create a large checkpoint
            large_messages = []
            for i in range(100):
                large_messages.append(
                    {
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": f"This is message number {i} " * 50,
                        "metadata": {"index": i, "data": list(range(50))},
                    }
                )

            checkpoint = Checkpoint(
                v=1,
                id=str(uuid6(clock_seq=-2)),
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                channel_values={
                    "messages": large_messages,
                    "context": {
                        "large_array": list(range(1000)),
                    },
                },
                channel_versions={"messages": "v1", "context": "v1"},
                versions_seen={},
                pending_sends=[],
            )

            metadata = {"size": "large", "message_count": len(large_messages)}
            new_versions = {"messages": "1.0", "context": "1.0"}

            # Store large checkpoint
            result_config = await buffered_agentcore_valkey_saver.aput(
                config, checkpoint, metadata, new_versions
            )

            # Retrieve from buffer and verify
            retrieved = await buffered_agentcore_valkey_saver.aget_tuple(result_config)

            assert retrieved is not None
            assert len(retrieved.checkpoint["channel_values"]["messages"]) == 100
            assert (
                len(retrieved.checkpoint["channel_values"]["context"]["large_array"])
                == 1000
            )
            assert retrieved.metadata["message_count"] == 100

        # Verify persisted data
        persisted = await agentcore_valkey_saver.aget_tuple(result_config)
        assert persisted is not None
        assert len(persisted.checkpoint["channel_values"]["messages"]) == 100
        assert (
            len(persisted.checkpoint["channel_values"]["context"]["large_array"])
            == 1000
        )