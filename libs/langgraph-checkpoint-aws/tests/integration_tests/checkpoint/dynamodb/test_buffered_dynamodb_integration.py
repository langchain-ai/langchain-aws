"""Integration tests for BufferedCheckpointSaver wrapping DynamoDBSaver."""

import datetime
import logging
import operator
import time
from typing import Annotated, Literal, TypedDict

import pytest
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, uuid6
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from langgraph_checkpoint_aws import BufferedCheckpointSaver, DynamoDBSaver
from tests.integration_tests.checkpoint.dynamodb.utils import (
    clean_dynamodb,
)
from tests.integration_tests.utils import (
    generate_large_data,
)

logger = logging.getLogger(__name__)


class _TestSequentialWorkflowState(TypedDict, total=False):
    task_a_completed: bool
    task_b_completed: bool
    completed: bool
    step_count: Annotated[int, operator.add]


def _build_sequential_workflow_graph(
    checkpointer: BufferedCheckpointSaver,
    *,
    flush_mid_workflow: bool = False,
    raise_mid_workflow: bool = False,
):
    """Build a 3-step sequential workflow: init -> task_a -> task_b -> finalize."""

    def init(state: _TestSequentialWorkflowState) -> _TestSequentialWorkflowState:
        return {
            "task_a_completed": False,
            "task_b_completed": False,
            "completed": False,
            "step_count": 0,
        }

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
    graph.add_node("init", init)
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "init")
    graph.add_edge("init", "task_a")
    graph.add_edge("task_a", "task_b")
    graph.add_edge("task_b", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


class _TestParallelGraphState(TypedDict, total=False):
    results: Annotated[list, operator.add]
    step_count: Annotated[int, operator.add]
    completed: bool


def _build_parallel_workflow_graph(
    checkpointer: BufferedCheckpointSaver,
    *,
    flush_mid_workflow: bool = False,
    raise_mid_workflow: bool = False,
):
    """Build a parallel workflow: init -> (task_a, task_b) -> merge."""

    def init(state: _TestParallelGraphState) -> _TestParallelGraphState:
        return {
            "results": [],
            "step_count": 0,
            "completed": False,
        }

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
    graph.add_node("init", init)
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("merge", merge_step)

    graph.add_edge(START, "init")
    graph.add_edge("init", "task_a")
    graph.add_edge("init", "task_b")
    graph.add_edge("task_a", "merge")
    graph.add_edge("task_b", "merge")
    graph.add_edge("merge", END)

    return graph.compile(checkpointer=checkpointer)


class _TestConditionalWorkflowState(TypedDict):
    """State for comprehensive workflow testing."""

    messages: Annotated[list, add_messages]
    step_count: int
    processing_complete: bool
    large_payload: str
    validation_result: str
    metadata: dict


def _build_conditional_workflow_graph(
    checkpointer: BufferedCheckpointSaver | DynamoDBSaver,
):
    """Build a workflow with conditional routing based on step_count."""

    def initialize_workflow(state: _TestConditionalWorkflowState) -> dict:
        step_count = state.get("step_count", 1)
        return {
            "messages": [{"role": "system", "content": "Workflow initialized"}],
            "step_count": step_count,
            "processing_complete": False,
            "metadata": {"start_time": time.time()},
        }

    def process_step(state: _TestConditionalWorkflowState) -> dict:
        current_step = state.get("step_count", 0)
        return {
            "messages": [
                {"role": "assistant", "content": f"Processed step {current_step}"}
            ],
            "step_count": current_step + 1,
        }

    def generate_large_checkpoint_data(state: _TestConditionalWorkflowState) -> dict:
        large_data = generate_large_data(600)  # 600KB
        return {
            "messages": [{"role": "system", "content": "Large data generated"}],
            "large_payload": large_data,
        }

    def validate_state(state: _TestConditionalWorkflowState) -> dict:
        step_count = state.get("step_count", 0)
        has_large_data = len(state.get("large_payload", "")) > 0
        validation = f"Steps: {step_count}, Large data: {has_large_data}"
        return {
            "messages": [{"role": "system", "content": "Validation complete"}],
            "validation_result": validation,
        }

    def finalize_workflow(state: _TestConditionalWorkflowState) -> dict:
        return {
            "messages": [{"role": "system", "content": "Workflow completed"}],
            "processing_complete": True,
            "metadata": {**state.get("metadata", {}), "end_time": time.time()},
        }

    def should_generate_large_data(
        state: _TestConditionalWorkflowState,
    ) -> Literal["yes", "no"]:
        return "yes" if state.get("step_count", 0) >= 2 else "no"

    graph = StateGraph(_TestConditionalWorkflowState)
    graph.add_node("init", initialize_workflow)
    graph.add_node("process", process_step)
    graph.add_node("large_data", generate_large_checkpoint_data)
    graph.add_node("validate", validate_state)
    graph.add_node("finalize", finalize_workflow)

    graph.add_edge(START, "init")
    graph.add_edge("init", "process")
    graph.add_conditional_edges(
        "process",
        should_generate_large_data,
        {"yes": "large_data", "no": "validate"},
    )
    graph.add_edge("large_data", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


class _TestParallelResumableState(TypedDict):
    """State for parallel execution with resumability testing."""

    messages: Annotated[list, add_messages]
    parallel_results: Annotated[list[str], operator.add]
    attempt_count: int
    retry_count: int


def _build_parallel_resumable_graph(
    checkpointer: BufferedCheckpointSaver | DynamoDBSaver,
):
    """Build a parallel workflow with resumability:
    (task_a, task_b, task_c) -> merge."""

    def task_a(state: _TestParallelResumableState) -> dict:
        return {
            "messages": [{"role": "system", "content": "Task A completed"}],
            "parallel_results": ["result_a"],
        }

    def task_b(state: _TestParallelResumableState) -> dict:
        retry_count = state.get("retry_count", 0)
        if retry_count < 1:
            raise ValueError("Task B failed (simulated partial failure)")
        return {
            "messages": [{"role": "system", "content": "Task B completed"}],
            "parallel_results": ["result_b"],
        }

    def task_c(state: _TestParallelResumableState) -> dict:
        return {
            "messages": [{"role": "system", "content": "Task C completed"}],
            "parallel_results": ["result_c"],
        }

    def merge_results(state: _TestParallelResumableState) -> dict:
        result_count = len(state.get("parallel_results", []))
        return {
            "messages": [
                {"role": "system", "content": f"Merged {result_count} results"}
            ],
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    graph = StateGraph(_TestParallelResumableState)
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("task_c", task_c)
    graph.add_node("merge", merge_results)

    graph.add_edge(START, "task_a")
    graph.add_edge(START, "task_b")
    graph.add_edge(START, "task_c")
    graph.add_edge("task_a", "merge")
    graph.add_edge("task_b", "merge")
    graph.add_edge("task_c", "merge")
    graph.add_edge("merge", END)

    return graph.compile(checkpointer=checkpointer)


class _TestSubgraphWorkflowState(TypedDict):
    """State for subgraph workflow testing."""

    input_data: str
    analysis_result: str
    validation_result: str
    final_output: str


def _build_analysis_subgraph(
    checkpointer: BufferedCheckpointSaver | DynamoDBSaver,
):
    """Build analysis subgraph: extract -> transform."""

    def extract_data(state: _TestSubgraphWorkflowState) -> dict:
        data = state.get("input_data", "")
        return {"analysis_result": f"extracted[{data}]"}

    def transform_data(state: _TestSubgraphWorkflowState) -> dict:
        result = state.get("analysis_result", "")
        return {"analysis_result": f"{result}→transformed"}

    graph = StateGraph(_TestSubgraphWorkflowState)
    graph.add_node("extract", extract_data)
    graph.add_node("transform", transform_data)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "transform")
    graph.add_edge("transform", END)

    return graph.compile(checkpointer=checkpointer)


def _build_validation_subgraph(
    checkpointer: BufferedCheckpointSaver | DynamoDBSaver,
):
    """Build validation subgraph: check -> verify."""

    def check_data(state: _TestSubgraphWorkflowState) -> dict:
        data = state.get("input_data", "")
        return {"validation_result": f"checked[{data}]"}

    def verify_data(state: _TestSubgraphWorkflowState) -> dict:
        result = state.get("validation_result", "")
        return {"validation_result": f"{result}→verified"}

    graph = StateGraph(_TestSubgraphWorkflowState)
    graph.add_node("check", check_data)
    graph.add_node("verify", verify_data)
    graph.add_edge(START, "check")
    graph.add_edge("check", "verify")
    graph.add_edge("verify", END)

    return graph.compile(checkpointer=checkpointer)


def _build_parent_graph_with_subgraphs(
    checkpointer: BufferedCheckpointSaver | DynamoDBSaver,
):
    """Build parent graph: analysis_subgraph -> validation_subgraph -> combine."""

    def combine_results(state: _TestSubgraphWorkflowState) -> dict:
        analysis = state.get("analysis_result", "")
        validation = state.get("validation_result", "")
        return {"final_output": f"COMBINED[{analysis} + {validation}]"}

    analysis_subgraph = _build_analysis_subgraph(checkpointer)
    validation_subgraph = _build_validation_subgraph(checkpointer)

    graph = StateGraph(_TestSubgraphWorkflowState)
    graph.add_node("analysis", analysis_subgraph)
    graph.add_node("validation", validation_subgraph)
    graph.add_node("combine", combine_results)
    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "validation")
    graph.add_edge("validation", "combine")
    graph.add_edge("combine", END)

    return graph.compile(checkpointer=checkpointer)


class TestBufferedDynamoDBSaverIntegrationSync:
    def test_flush_on_exit_with_seq_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """Test that the BufferedCheckpointSaver flushes on exit."""
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            assert buffered_dynamodb_saver.is_empty

            with buffered_dynamodb_saver.flush_on_exit():
                graph.invoke({}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            assert len(list(dynamodb_saver.list(config))) == 1

            history = list(graph.get_state_history(config))
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_on_exit_with_seq_workflow_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                for _ in graph.stream({}, config):
                    assert not buffered_dynamodb_saver.is_empty
                    assert buffered_dynamodb_saver.has_buffered_checkpoint
                    assert buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            history = list(graph.get_state_history(config))
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_on_exit_with_parallel_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                graph.invoke({"step_count": 0, "results": []}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}
            assert len(list(dynamodb_saver.list(config))) == 1

            history = list(graph.get_state_history(config))
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_on_exit_with_parallel_workflow_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                for _ in graph.stream({"step_count": 0, "results": []}, config):
                    assert not buffered_dynamodb_saver.is_empty
                    assert buffered_dynamodb_saver.has_buffered_checkpoint
                    assert buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}

            history = list(graph.get_state_history(config))
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_mid_seq_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                graph.invoke({}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            # One from mid-workflow flush, one from flush_on_exit
            assert len(list(dynamodb_saver.list(config))) == 2

            history = list(graph.get_state_history(config))
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_mid_seq_workflow_with_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                streamed_nodes = []
                for event in graph.stream({}, config):
                    node_name = list(event.keys())[0]
                    streamed_nodes.append(node_name)

                    if node_name == "task_a":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint
                        assert dynamodb_saver.get_tuple(config) is None

                    elif node_name == "task_b":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                        # Mid-workflow flush persisted state BEFORE task_b completed
                        mid_flush_persisted = dynamodb_saver.get_tuple(config)
                        assert mid_flush_persisted is not None
                        assert mid_flush_persisted.checkpoint["channel_values"] == {
                            "task_a_completed": True,
                            "task_b_completed": False,
                            "completed": False,
                            "step_count": 1,
                        }

                    elif node_name == "finalize":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                assert streamed_nodes == ["task_a", "task_b", "finalize"]
                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            # One from mid-workflow flush, one from flush_on_exit
            assert len(list(dynamodb_saver.list(config))) == 2

            history = list(graph.get_state_history(config))
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_mid_parallel_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                graph.invoke({"step_count": 0, "results": []}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}

            # One from mid-workflow flush, one from flush_on_exit
            assert len(list(dynamodb_saver.list(config))) == 2

            history = list(graph.get_state_history(config))
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_flush_mid_parallel_workflow_with_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                streamed_nodes = []
                for event in graph.stream({"step_count": 0, "results": []}, config):
                    node_name = list(event.keys())[0]
                    streamed_nodes.append(node_name)

                    if node_name in ("task_a", "task_b"):
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint
                        assert dynamodb_saver.get_tuple(config) is None

                    elif node_name == "merge":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                        # Mid-workflow flush persisted state BEFORE merge completed
                        mid_flush_persisted = dynamodb_saver.get_tuple(config)
                        assert mid_flush_persisted is not None
                        mid_state = mid_flush_persisted.checkpoint["channel_values"]
                        assert mid_state["step_count"] == 2
                        assert mid_state["completed"] is False
                        assert set(mid_state["results"]) == {"result_a", "result_b"}

                assert set(streamed_nodes) == {"task_a", "task_b", "merge"}
                assert streamed_nodes[-1] == "merge"
                assert graph.get_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}
            assert len(list(dynamodb_saver.list(config))) == 2

            history = list(graph.get_state_history(config))
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    def test_context_manager_flushes_on_exception_and_propagates(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """Test that a buffered checkpoint saver flushes state even
        when an exception occurs within a context manager."""

        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            raise_mid_workflow=True,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with pytest.raises(ValueError, match="Test error: simulated task failure"):
                with buffered_dynamodb_saver.flush_on_exit():
                    graph.invoke({}, config)

            assert buffered_dynamodb_saver.is_empty
            assert not buffered_dynamodb_saver.has_buffered_checkpoint
            assert not buffered_dynamodb_saver.has_buffered_writes

            persisted = dynamodb_saver.get_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": False,
                "completed": False,
                "step_count": 1,
            }

    def test_manual_flush_without_context_manager(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            result = graph.invoke({}, config)
            assert result["completed"] is True
            assert not buffered_dynamodb_saver.is_empty
            assert buffered_dynamodb_saver.has_buffered_checkpoint
            assert dynamodb_saver.get_tuple(config) is None

            flush_result = buffered_dynamodb_saver.flush()
            assert flush_result is not None
            assert buffered_dynamodb_saver.is_empty

            persisted = dynamodb_saver.get_tuple(config)
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
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                result1 = graph.invoke({}, config)
                assert result1["step_count"] == 3

            assert len(list(dynamodb_saver.list(config))) == 1

            with buffered_dynamodb_saver.flush_on_exit():
                result2 = graph.invoke(None, config)
                assert result2["step_count"] == 6

            assert len(list(dynamodb_saver.list(config))) == 2

    def test_multiple_sessions_isolation(
        self,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id_1 = agentcore_session_id + "-1"
        thread_id_2 = agentcore_session_id + "-2"

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)

        config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id_1,
            }
        }
        config_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id_2,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id_1, thread_id_2],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                result_1 = graph.invoke({}, config_1)

            assert result_1["step_count"] == 3
            assert result_1["completed"] is True

            with buffered_dynamodb_saver.flush_on_exit():
                result_2 = graph.invoke({}, config_2)

            assert result_2["step_count"] == 3
            assert result_2["completed"] is True

            checkpoints_1 = list(dynamodb_saver.list(config_1))
            checkpoints_2 = list(dynamodb_saver.list(config_2))
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
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            num_invocations = 3
            for i in range(num_invocations):
                with buffered_dynamodb_saver.flush_on_exit():
                    if i == 0:
                        result = graph.invoke({}, config)
                    else:
                        result = graph.invoke(None, config)
                assert result["step_count"] == (i + 1) * 3

            all_checkpoints = list(dynamodb_saver.list(config))
            assert len(all_checkpoints) == num_invocations

            limit = 2
            limited_checkpoints = list(dynamodb_saver.list(config, limit=limit))
            assert len(limited_checkpoints) == limit

            # Limited results should match the first N from the full list
            for i in range(limit):
                limited_id = limited_checkpoints[i].config["configurable"][
                    "checkpoint_id"
                ]
                full_id = all_checkpoints[i].config["configurable"]["checkpoint_id"]
                assert limited_id == full_id

            # Most recent checkpoint should have highest step_count
            most_recent = all_checkpoints[0].checkpoint["channel_values"]["step_count"]
            oldest = all_checkpoints[-1].checkpoint["channel_values"]["step_count"]
            assert most_recent > oldest

    def test_checkpoint_save_and_retrieve(
        self,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
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
            updated_channels=[],
        )

        checkpoint_metadata: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"node1": ["write1", "write2"]},  # type: ignore[typeddict-item]
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with buffered_dynamodb_saver.flush_on_exit():
                saved_config = buffered_dynamodb_saver.put(
                    config,
                    checkpoint,
                    checkpoint_metadata,
                    {"messages": "v2", "results": "v2"},
                )

                assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
                assert saved_config["configurable"]["thread_id"] == thread_id
                assert saved_config["configurable"]["checkpoint_ns"] == "test_namespace"

                checkpoint_tuple = buffered_dynamodb_saver.get_tuple(saved_config)
                assert checkpoint_tuple is not None
                assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

            persisted_tuple = dynamodb_saver.get_tuple(saved_config)
            assert persisted_tuple is not None
            assert persisted_tuple.checkpoint["id"] == checkpoint["id"]

            expected_metadata = checkpoint_metadata.copy()
            assert persisted_tuple.metadata == expected_metadata

    def test_math_agent_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            graph = create_agent(
                bedrock_model,
                tools=agent_tools,
                checkpointer=buffered_dynamodb_saver,
            )
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            with buffered_dynamodb_saver.flush_on_exit():
                response = graph.invoke(
                    {
                        "messages": [
                            (
                                "human",
                                "What is 15 times 23? Then add 100 to the result.",
                            )
                        ]
                    },  # type: ignore[arg-type]
                    config,
                )
                assert response
                assert "messages" in response
                assert len(response["messages"]) > 1

                checkpoint = buffered_dynamodb_saver.get(config)
                assert checkpoint

            persisted_checkpoint = dynamodb_saver.get(config)
            assert persisted_checkpoint

            checkpoint_tuples = list(dynamodb_saver.list(config))
            assert checkpoint_tuples
            assert len(checkpoint_tuples) == 1

            with buffered_dynamodb_saver.flush_on_exit():
                response2 = graph.invoke(
                    {
                        "messages": [
                            (
                                "human",
                                "What was the final result from my previous calculation?",  # noqa: E501
                            )
                        ]
                    },  # type: ignore[arg-type]
                    config,
                )
                assert response2

            checkpoint_tuples_after = list(dynamodb_saver.list(config))
            assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

    def test_weather_query_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            graph = create_agent(
                bedrock_model,
                tools=agent_tools,
                checkpointer=buffered_dynamodb_saver,
            )
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            with buffered_dynamodb_saver.flush_on_exit():
                response = graph.invoke(
                    {"messages": [("human", "What's the weather in sf and nyc?")]},  # type: ignore[arg-type]
                    config,
                )
                assert response

                checkpoint = buffered_dynamodb_saver.get(config)
                assert checkpoint

            persisted_checkpoint = dynamodb_saver.get(config)
            assert persisted_checkpoint

            checkpoint_tuples = list(dynamodb_saver.list(config))
            assert checkpoint_tuples
            assert len(checkpoint_tuples) == 1

    def test_conditional_routing_and_checkpoint_branching(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test conditional routing with checkpoint validation at branch points.

        Tests two scenarios:
        - Path 1: step_count < 2 after process → skips large_data
        - Path 2: step_count >= 2 after process → includes large_data

        Validates:
        - Conditional routing based on state
        - Different execution paths produce different final states
        - Single checkpoint persisted per flush (due to buffering)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        # Scenario 1: Path WITHOUT large data (step_count < 2 after process)
        thread_id_path1 = f"{thread_id}_path1"
        config1 = {"configurable": {"thread_id": thread_id_path1}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id_path1]):
            with buffered_dynamodb_saver.flush_on_exit():
                app = _build_conditional_workflow_graph(buffered_dynamodb_saver)
                result1 = app.invoke({}, config1)

                # Verify NO large payload (took "no" path)
                has_large_payload = (
                    "large_payload" in result1
                    and len(result1.get("large_payload", "")) > 0
                )
                assert not has_large_payload, (
                    "Path 1 should NOT generate large data (took 'no' path)"
                )
                assert result1["processing_complete"] is True
                assert result1["step_count"] == 1

            # Validate single checkpoint persisted (due to buffering)
            app = _build_conditional_workflow_graph(dynamodb_saver)
            history1 = list(app.get_state_history(config1))
            assert len(history1) == 1
            checkpoint_state = history1[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

        # Scenario 2: Path WITH large data (step_count >= 2 after process)
        thread_id_path2 = f"{thread_id}_path2"
        config2 = {"configurable": {"thread_id": thread_id_path2}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id_path2]):
            with buffered_dynamodb_saver.flush_on_exit():
                app = _build_conditional_workflow_graph(buffered_dynamodb_saver)
                result2 = app.invoke({"step_count": 2}, config2)

                # Verify HAS large payload (took "yes" path)
                has_large_payload = (
                    "large_payload" in result2
                    and len(result2.get("large_payload", "")) > 0
                )
                assert has_large_payload, (
                    "Path 2 SHOULD generate large data (took 'yes' path)"
                )
                assert result2["processing_complete"] is True
                assert result2["step_count"] == 3

            # Validate single checkpoint persisted (due to buffering)
            app = _build_conditional_workflow_graph(dynamodb_saver)
            history2 = list(app.get_state_history(config2))
            assert len(history2) == 1
            checkpoint_state = history2[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]
            # Verify the final state reflects the "yes" path was taken
            assert len(checkpoint_state.values.get("large_payload", "")) > 0

    def test_parallel_execution_with_resumability(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test parallel execution with failure recovery and resumability.

        Validates:
        - Partial failure preserves completed task results
        - Resume from failure with Command works
        - Two checkpoints total (one per flush_on_exit)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id]):
            # ATTEMPT 1: Partial failure (task_b fails)
            with buffered_dynamodb_saver.flush_on_exit():
                app = _build_parallel_resumable_graph(buffered_dynamodb_saver)
                try:
                    app.invoke({"retry_count": 0, "attempt_count": 0}, config)
                    raise AssertionError("Should have raised ValueError from task_b")
                except ValueError:
                    pass  # Expected

            # Validate state after partial failure (1 checkpoint from first flush)
            app = _build_parallel_resumable_graph(dynamodb_saver)
            state_after_failure = app.get_state(config)
            assert state_after_failure is not None

            # Check partial results - task_a and task_c should have completed
            partial_results = state_after_failure.values.get("parallel_results", [])
            assert len(partial_results) == 2, (
                f"Should have 2 partial results (task_a, task_c), got {len(partial_results)}"  # noqa: E501
            )

            # Check pending nodes
            pending_nodes = state_after_failure.next
            assert len(pending_nodes) == 1

            history_after_failure = list(app.get_state_history(config))
            assert len(history_after_failure) == 1

            # ATTEMPT 2: Resume with Command - sets retry_count=1 so task_b succeeds
            with buffered_dynamodb_saver.flush_on_exit():
                app = _build_parallel_resumable_graph(buffered_dynamodb_saver)
                result = app.invoke(
                    Command(update={"retry_count": 1}, goto="task_b"), config
                )

                # Validate successful completion
                assert "parallel_results" in result
                parallel_results = result["parallel_results"]
                assert len(parallel_results) == 3
                assert "result_a" in parallel_results
                assert "result_b" in parallel_results
                assert "result_c" in parallel_results
                assert result["attempt_count"] >= 1

            # Validate checkpoint history shows both attempts (2 checkpoints total)
            app = _build_parallel_resumable_graph(dynamodb_saver)
            history_after_success = list(app.get_state_history(config))
            assert len(history_after_success) == 2

            # Validate checkpoint structure
            for checkpoint_state in history_after_success:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

            # Validate current state reflects successful completion
            current_state = app.get_state(config)
            assert current_state.values["attempt_count"] >= 1
            assert len(current_state.next) == 0

    def test_subgraph_execution_with_checkpoints(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test nested subgraph execution with checkpointing.

        Validates:
        - Both subgraphs execute and produce expected results
        - Final output combines results from both subgraphs
        - Single checkpoint persisted (due to buffering)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id]):
            with buffered_dynamodb_saver.flush_on_exit():
                app = _build_parent_graph_with_subgraphs(buffered_dynamodb_saver)
                result = app.invoke({"input_data": "test_data"}, config)

                # Validate both subgraphs executed
                assert "extracted" in result["analysis_result"]
                assert "transformed" in result["analysis_result"]
                assert "checked" in result["validation_result"]
                assert "verified" in result["validation_result"]

                # Validate final output combines both
                assert "COMBINED" in result["final_output"]
                assert "extracted" in result["final_output"]
                assert "checked" in result["final_output"]

            # Validate single checkpoint persisted (due to buffering)
            app = _build_parent_graph_with_subgraphs(dynamodb_saver)
            history = list(app.get_state_history(config))
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]
            # Verify final state contains results from both subgraphs
            assert "extracted" in checkpoint_state.values["analysis_result"]
            assert "checked" in checkpoint_state.values["validation_result"]


class TestBufferedDynamoDBSaverIntegrationAsync:
    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_seq_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """Test that the BufferedCheckpointSaver flushes on exit."""
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            assert buffered_dynamodb_saver.is_empty

            async with buffered_dynamodb_saver.aflush_on_exit():
                await graph.ainvoke({}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            assert len([c async for c in dynamodb_saver.alist(config)]) == 1

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_seq_workflow_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                async for _ in graph.astream({}, config):
                    assert not buffered_dynamodb_saver.is_empty
                    assert buffered_dynamodb_saver.has_buffered_checkpoint
                    assert buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_parallel_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                await graph.ainvoke({"step_count": 0, "results": []}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}
            assert len([c async for c in dynamodb_saver.alist(config)]) == 1

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_on_exit_with_parallel_workflow_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                async for _ in graph.astream({"step_count": 0, "results": []}, config):
                    assert not buffered_dynamodb_saver.is_empty
                    assert buffered_dynamodb_saver.has_buffered_checkpoint
                    assert buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_mid_seq_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                await graph.ainvoke({}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            # One from mid-workflow flush, one from flush_on_exit
            assert len([c async for c in dynamodb_saver.alist(config)]) == 2

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_mid_seq_workflow_with_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                streamed_nodes = []
                async for event in graph.astream({}, config):
                    node_name = list(event.keys())[0]
                    streamed_nodes.append(node_name)

                    if node_name == "task_a":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint
                        assert await dynamodb_saver.aget_tuple(config) is None

                    elif node_name == "task_b":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                        # Mid-workflow flush persisted state BEFORE task_b completed
                        mid_flush_persisted = await dynamodb_saver.aget_tuple(config)
                        assert mid_flush_persisted is not None
                        assert mid_flush_persisted.checkpoint["channel_values"] == {
                            "task_a_completed": True,
                            "task_b_completed": False,
                            "completed": False,
                            "step_count": 1,
                        }

                    elif node_name == "finalize":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                assert streamed_nodes == ["task_a", "task_b", "finalize"]
                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None
            assert persisted.checkpoint["channel_values"] == {
                "task_a_completed": True,
                "task_b_completed": True,
                "completed": True,
                "step_count": 3,
            }

            # One from mid-workflow flush, one from flush_on_exit
            assert len([c async for c in dynamodb_saver.alist(config)]) == 2

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_mid_parallel_workflow(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                await graph.ainvoke({"step_count": 0, "results": []}, config)

                assert not buffered_dynamodb_saver.is_empty
                assert buffered_dynamodb_saver.has_buffered_checkpoint
                assert not buffered_dynamodb_saver.has_buffered_writes

                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}

            # One from mid-workflow flush, one from flush_on_exit
            assert len([c async for c in dynamodb_saver.alist(config)]) == 2

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_flush_mid_parallel_workflow_with_streaming(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_parallel_workflow_graph(
            buffered_dynamodb_saver,
            flush_mid_workflow=True,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                streamed_nodes = []
                async for event in graph.astream(
                    {"step_count": 0, "results": []}, config
                ):
                    node_name = list(event.keys())[0]
                    streamed_nodes.append(node_name)

                    if node_name in ("task_a", "task_b"):
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint
                        assert await dynamodb_saver.aget_tuple(config) is None

                    elif node_name == "merge":
                        assert not buffered_dynamodb_saver.is_empty
                        assert buffered_dynamodb_saver.has_buffered_checkpoint

                        # Mid-workflow flush persisted state BEFORE merge completed
                        mid_flush_persisted = await dynamodb_saver.aget_tuple(config)
                        assert mid_flush_persisted is not None
                        mid_state = mid_flush_persisted.checkpoint["channel_values"]
                        assert mid_state["step_count"] == 2
                        assert mid_state["completed"] is False
                        assert set(mid_state["results"]) == {"result_a", "result_b"}

                assert set(streamed_nodes) == {"task_a", "task_b", "merge"}
                assert streamed_nodes[-1] == "merge"
                assert await graph.aget_state(config) is not None

            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
            assert persisted is not None
            assert persisted.checkpoint is not None

            channel_values = persisted.checkpoint["channel_values"]
            assert channel_values["step_count"] == 3
            assert channel_values["completed"] is True
            assert set(channel_values["results"]) == {"result_a", "result_b"}
            assert len([c async for c in dynamodb_saver.alist(config)]) == 2

            history = [c async for c in graph.aget_state_history(config)]
            assert len(history) == 2
            for checkpoint_state in history:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

    @pytest.mark.asyncio
    async def test_async_context_manager_flushes_on_exception_and_propagates(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """Test that a buffered checkpoint saver flushes state
        even when an exception occurs within a context manager."""

        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
            raise_mid_workflow=True,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            with pytest.raises(ValueError, match="Test error: simulated task failure"):
                async with buffered_dynamodb_saver.aflush_on_exit():
                    await graph.ainvoke({}, config)

            assert buffered_dynamodb_saver.is_empty
            assert not buffered_dynamodb_saver.has_buffered_checkpoint
            assert not buffered_dynamodb_saver.has_buffered_writes

            persisted = await dynamodb_saver.aget_tuple(config)
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
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(
            buffered_dynamodb_saver,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        assert buffered_dynamodb_saver.is_empty

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            result = await graph.ainvoke({}, config)
            assert result["completed"] is True
            assert not buffered_dynamodb_saver.is_empty
            assert buffered_dynamodb_saver.has_buffered_checkpoint
            assert await dynamodb_saver.aget_tuple(config) is None

            flush_result = await buffered_dynamodb_saver.aflush()
            assert flush_result is not None
            assert buffered_dynamodb_saver.is_empty

            persisted = await dynamodb_saver.aget_tuple(config)
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
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                result1 = await graph.ainvoke({}, config)
                assert result1["step_count"] == 3

            assert len(list(dynamodb_saver.list(config))) == 1

            async with buffered_dynamodb_saver.aflush_on_exit():
                result2 = await graph.ainvoke(None, config)
                assert result2["step_count"] == 6

            assert len([x async for x in dynamodb_saver.alist(config)]) == 2

    @pytest.mark.asyncio
    async def test_async_multiple_sessions_isolation(
        self,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id_1 = agentcore_session_id + "-1"
        thread_id_2 = agentcore_session_id + "-2"

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)

        config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id_1,
            }
        }
        config_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id_2,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id_1, thread_id_2],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                result_1 = await graph.ainvoke({}, config_1)

            assert result_1["step_count"] == 3
            assert result_1["completed"] is True

            async with buffered_dynamodb_saver.aflush_on_exit():
                result_2 = await graph.ainvoke({}, config_2)

            assert result_2["step_count"] == 3
            assert result_2["completed"] is True

            checkpoints_1 = [x async for x in dynamodb_saver.alist(config_1)]
            checkpoints_2 = [x async for x in dynamodb_saver.alist(config_2)]
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
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        graph = _build_sequential_workflow_graph(buffered_dynamodb_saver)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            num_invocations = 3
            for i in range(num_invocations):
                async with buffered_dynamodb_saver.aflush_on_exit():
                    if i == 0:
                        result = await graph.ainvoke({}, config)
                    else:
                        result = await graph.ainvoke(None, config)
                assert result["step_count"] == (i + 1) * 3

            all_checkpoints = [x async for x in dynamodb_saver.alist(config)]
            assert len(all_checkpoints) == num_invocations

            limit = 2
            limited_checkpoints = [
                x async for x in dynamodb_saver.alist(config, limit=limit)
            ]
            assert len(limited_checkpoints) == limit

            # Limited results should match the first N from the full list
            for i in range(limit):
                limited_id = limited_checkpoints[i].config["configurable"][
                    "checkpoint_id"
                ]
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
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
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
            updated_channels=[],
        )

        checkpoint_metadata: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"node1": ["write1", "write2"]},  # type: ignore[typeddict-item]
        }

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            async with buffered_dynamodb_saver.aflush_on_exit():
                saved_config = await buffered_dynamodb_saver.aput(
                    config,
                    checkpoint,
                    checkpoint_metadata,
                    {"messages": "v2", "results": "v2"},
                )

                assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]
                assert saved_config["configurable"]["thread_id"] == thread_id
                assert saved_config["configurable"]["checkpoint_ns"] == "test_namespace"

                checkpoint_tuple = await buffered_dynamodb_saver.aget_tuple(
                    saved_config
                )
                assert checkpoint_tuple is not None
                assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]

            persisted_tuple = await dynamodb_saver.aget_tuple(saved_config)
            assert persisted_tuple is not None
            assert persisted_tuple.checkpoint["id"] == checkpoint["id"]

            expected_metadata = checkpoint_metadata.copy()
            assert persisted_tuple.metadata == expected_metadata

    @pytest.mark.asyncio
    async def test_async_math_agent_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            graph = create_agent(
                bedrock_model,
                tools=agent_tools,
                checkpointer=buffered_dynamodb_saver,
            )
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            async with buffered_dynamodb_saver.aflush_on_exit():
                response = await graph.ainvoke(
                    {
                        "messages": [
                            (
                                "human",
                                "What is 15 times 23? Then add 100 to the result.",
                            )
                        ]
                    },  # type: ignore[arg-type]
                    config,
                )
                assert response
                assert "messages" in response
                assert len(response["messages"]) > 1

                checkpoint = await buffered_dynamodb_saver.aget(config)
                assert checkpoint

            persisted_checkpoint = await dynamodb_saver.aget(config)
            assert persisted_checkpoint

            checkpoint_tuples = [x async for x in dynamodb_saver.alist(config)]
            assert checkpoint_tuples
            assert len(checkpoint_tuples) == 1

            async with buffered_dynamodb_saver.aflush_on_exit():
                response2 = await graph.ainvoke(
                    {
                        "messages": [
                            (
                                "human",
                                "What was the final result from my previous calculation?",  # noqa: E501
                            )
                        ]
                    },  # type: ignore[arg-type]
                    config,
                )
                assert response2

            checkpoint_tuples_after = [x async for x in dynamodb_saver.alist(config)]
            assert len(checkpoint_tuples_after) > len(checkpoint_tuples)

    @pytest.mark.asyncio
    async def test_async_weather_query_with_checkpointing(
        self,
        agent_tools: list[Tool],
        bedrock_model: ChatBedrock,
        agentcore_session_id: str,
        dynamodb_saver: DynamoDBSaver,
    ):
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        with clean_dynamodb(
            dynamodb_saver,
            thread_ids=[thread_id],
        ):
            graph = create_agent(
                bedrock_model,
                tools=agent_tools,
                checkpointer=buffered_dynamodb_saver,
            )
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

            async with buffered_dynamodb_saver.aflush_on_exit():
                response = await graph.ainvoke(
                    {"messages": [("human", "What's the weather in sf and nyc?")]},  # type: ignore[arg-type]
                    config,
                )
                assert response

                checkpoint = await buffered_dynamodb_saver.aget(config)
                assert checkpoint

            persisted_checkpoint = await dynamodb_saver.aget(config)
            assert persisted_checkpoint

            checkpoint_tuples = [x async for x in dynamodb_saver.alist(config)]
            assert checkpoint_tuples
            assert len(checkpoint_tuples) == 1

    @pytest.mark.asyncio
    async def test_async_conditional_routing_and_checkpoint_branching(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test conditional routing with checkpoint validation at branch points.

        Tests two scenarios:
        - Path 1: step_count < 2 after process → skips large_data
        - Path 2: step_count >= 2 after process → includes large_data

        Validates:
        - Conditional routing based on state
        - Different execution paths produce different final states
        - Single checkpoint persisted per flush (due to buffering)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)

        # Scenario 1: Path WITHOUT large data (step_count < 2 after process)
        thread_id_path1 = f"{thread_id}_path1"
        config1 = {"configurable": {"thread_id": thread_id_path1}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id_path1]):
            async with buffered_dynamodb_saver.aflush_on_exit():
                app = _build_conditional_workflow_graph(buffered_dynamodb_saver)
                result1 = await app.ainvoke({}, config1)

                # Verify NO large payload (took "no" path)
                has_large_payload = (
                    "large_payload" in result1
                    and len(result1.get("large_payload", "")) > 0
                )
                assert not has_large_payload, (
                    "Path 1 should NOT generate large data (took 'no' path)"
                )
                assert result1["processing_complete"] is True
                assert result1["step_count"] == 1

            # Validate single checkpoint persisted (due to buffering)
            app = _build_conditional_workflow_graph(dynamodb_saver)
            history1 = [c async for c in app.aget_state_history(config1)]
            assert len(history1) == 1
            checkpoint_state = history1[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]

        # Scenario 2: Path WITH large data (step_count >= 2 after process)
        thread_id_path2 = f"{thread_id}_path2"
        config2 = {"configurable": {"thread_id": thread_id_path2}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id_path2]):
            async with buffered_dynamodb_saver.aflush_on_exit():
                app = _build_conditional_workflow_graph(buffered_dynamodb_saver)
                result2 = await app.ainvoke({"step_count": 2}, config2)

                # Verify HAS large payload (took "yes" path)
                has_large_payload = (
                    "large_payload" in result2
                    and len(result2.get("large_payload", "")) > 0
                )
                assert has_large_payload, (
                    "Path 2 SHOULD generate large data (took 'yes' path)"
                )
                assert result2["processing_complete"] is True
                assert result2["step_count"] == 3

            # Validate single checkpoint persisted (due to buffering)
            app = _build_conditional_workflow_graph(dynamodb_saver)
            history2 = [c async for c in app.aget_state_history(config2)]
            assert len(history2) == 1
            checkpoint_state = history2[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]
            # Verify the final state reflects the "yes" path was taken
            assert len(checkpoint_state.values.get("large_payload", "")) > 0

    @pytest.mark.asyncio
    async def test_async_parallel_execution_with_resumability(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test parallel execution with failure recovery and resumability.

        Validates:
        - Partial failure preserves completed task results
        - Resume from failure with Command works
        - Two checkpoints total (one per flush_on_exit)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id]):
            # ATTEMPT 1: Partial failure (task_b fails)
            async with buffered_dynamodb_saver.aflush_on_exit():
                app = _build_parallel_resumable_graph(buffered_dynamodb_saver)
                try:
                    await app.ainvoke({"retry_count": 0, "attempt_count": 0}, config)
                    raise AssertionError("Should have raised ValueError from task_b")
                except ValueError:
                    pass  # Expected

            # Validate state after partial failure (1 checkpoint from first flush)
            app = _build_parallel_resumable_graph(dynamodb_saver)
            state_after_failure = await app.aget_state(config)
            assert state_after_failure is not None

            # Check partial results - task_a and task_c should have completed
            partial_results = state_after_failure.values.get("parallel_results", [])
            assert len(partial_results) == 2, (
                f"Should have 2 partial results (task_a, task_c), got {len(partial_results)}"  # noqa: E501
            )

            # Check pending nodes
            pending_nodes = state_after_failure.next
            assert len(pending_nodes) == 1

            history_after_failure = [c async for c in app.aget_state_history(config)]
            assert len(history_after_failure) == 1

            # ATTEMPT 2: Resume with Command - sets retry_count=1 so task_b succeeds
            async with buffered_dynamodb_saver.aflush_on_exit():
                app = _build_parallel_resumable_graph(buffered_dynamodb_saver)
                result = await app.ainvoke(
                    Command(update={"retry_count": 1}, goto="task_b"), config
                )

                # Validate successful completion
                assert "parallel_results" in result
                parallel_results = result["parallel_results"]
                assert len(parallel_results) == 3
                assert "result_a" in parallel_results
                assert "result_b" in parallel_results
                assert "result_c" in parallel_results
                assert result["attempt_count"] >= 1

            # Validate checkpoint history shows both attempts (2 checkpoints total)
            app = _build_parallel_resumable_graph(dynamodb_saver)
            history_after_success = [c async for c in app.aget_state_history(config)]
            assert len(history_after_success) == 2

            # Validate checkpoint structure
            for checkpoint_state in history_after_success:
                assert checkpoint_state.values is not None
                assert checkpoint_state.config is not None
                assert "thread_id" in checkpoint_state.config["configurable"]
                assert "checkpoint_id" in checkpoint_state.config["configurable"]

            # Validate current state reflects successful completion
            current_state = await app.aget_state(config)
            assert current_state.values["attempt_count"] >= 1
            assert len(current_state.next) == 0

    @pytest.mark.asyncio
    async def test_async_subgraph_execution_with_checkpoints(
        self,
        dynamodb_saver: DynamoDBSaver,
        agentcore_session_id: str,
    ):
        """
        Test nested subgraph execution with checkpointing.

        Validates:
        - Both subgraphs execute and produce expected results
        - Final output combines results from both subgraphs
        - Single checkpoint persisted (due to buffering)
        """
        thread_id = agentcore_session_id

        buffered_dynamodb_saver = BufferedCheckpointSaver(dynamodb_saver)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with clean_dynamodb(dynamodb_saver, thread_ids=[thread_id]):
            async with buffered_dynamodb_saver.aflush_on_exit():
                app = _build_parent_graph_with_subgraphs(buffered_dynamodb_saver)
                result = await app.ainvoke({"input_data": "test_data"}, config)

                # Validate both subgraphs executed
                assert "extracted" in result["analysis_result"]
                assert "transformed" in result["analysis_result"]
                assert "checked" in result["validation_result"]
                assert "verified" in result["validation_result"]

                # Validate final output combines both
                assert "COMBINED" in result["final_output"]
                assert "extracted" in result["final_output"]
                assert "checked" in result["final_output"]

            # Validate single checkpoint persisted (due to buffering)
            app = _build_parent_graph_with_subgraphs(dynamodb_saver)
            history = [c async for c in app.aget_state_history(config)]
            assert len(history) == 1
            checkpoint_state = history[0]
            assert checkpoint_state.values is not None
            assert checkpoint_state.config is not None
            assert "thread_id" in checkpoint_state.config["configurable"]
            assert "checkpoint_id" in checkpoint_state.config["configurable"]
            # Verify final state contains results from both subgraphs
            assert "extracted" in checkpoint_state.values["analysis_result"]
            assert "checked" in checkpoint_state.values["validation_result"]
