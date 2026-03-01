import os
import random
import string
from contextlib import contextmanager, AbstractContextManager
from typing import Literal

import pytest
from langchain_aws import ChatBedrock
from langchain_core.tools import tool, Tool

from langgraph_checkpoint_aws import AgentCoreMemorySaver, BufferedCheckpointSaver
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver


########################################################
# AgentCore Memory
########################################################

@pytest.fixture(scope="function")
def agentcore_session_id() -> str:
    """Generate a valid session ID that matches AgentCore pattern."""
    chars = string.ascii_letters + string.digits
    return "test-session-id-" + "".join(random.choices(chars, k=8))


@pytest.fixture(scope="function")
def agentcore_actor_id() -> str:
    """Generate a valid actor ID that matches AgentCore pattern."""
    chars = string.ascii_letters + string.digits
    return "test-actor-id-" + "".join(random.choices(chars, k=6))


@pytest.fixture(scope="session")
def agentcore_memory_id() -> str:
    """Get memory ID from environment variable."""
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        pytest.skip("AGENTCORE_MEMORY_ID env variable not set")
    return memory_id


@pytest.fixture(scope="session")
def agentcore_memory_saver(
        agentcore_memory_id,
        aws_region,
    ) -> AgentCoreMemorySaver:
    """Create AgentCoreMemorySaver instance."""
    return AgentCoreMemorySaver(
        memory_id=agentcore_memory_id,
        region_name=aws_region,
    )

@contextmanager
def clean_agentcore_memory(
    saver: AgentCoreMemorySaver,
    /,
    *,
    actor_id: str,
    thread_ids: list[str],
) -> AbstractContextManager[AgentCoreMemorySaver]:
    """Cleanup AgentCoreMemorySaver resources on exit."""
    def _delete_threads(thread_ids: list[str]):
        for thread_id in thread_ids:
            saver.delete_thread(thread_id, actor_id)
    
    try:
        _delete_threads(thread_ids)
        yield saver
    finally:
        _delete_threads(thread_ids)

@pytest.fixture(scope="function")
def buffered_agentcore_memory_saver(agentcore_memory_saver) -> BufferedCheckpointSaver:
    """Create BufferedCheckpointSaver wrapping AgentCoreMemorySaver."""
    return BufferedCheckpointSaver(agentcore_memory_saver)

########################################################
# Valkey
########################################################

@pytest.fixture(scope="session")
def valkey_saver():
    """Create Valkey saver instance for integration tests."""
    
    uri = "valkey://localhost:6379/1"
    
    def _cleanup_valkey_test_keys(saver: AgentCoreValkeySaver):
        """Cleanup test keys from Valkey server."""
        session_keys = saver.client.keys("agentcore:session:test-*")
        checkpoint_keys = saver.client.keys("agentcore:checkpoint:test-*")
        writes_keys = saver.client.keys("agentcore:writes:test-*")
        channel_keys = saver.client.keys("agentcore:channel:test-*")

        all_keys = (
            list(session_keys)  # type: ignore[arg-type]
            + list(checkpoint_keys)  # type: ignore[arg-type]
            + list(writes_keys)  # type: ignore[arg-type]
            + list(channel_keys)  # type: ignore[arg-type]
        )
        if all_keys:
            saver.client.delete(*all_keys)

    try:
        with AgentCoreValkeySaver.from_conn_string(
            uri,
            ttl_seconds=600,
        ) as saver:
            _cleanup_valkey_test_keys(saver)
            yield saver
            _cleanup_valkey_test_keys(saver)
    except Exception as e:
        pytest.skip(f"Could not connect to Valkey server: {e}")

########################################################
# Bedrock
########################################################

@pytest.fixture(scope="session")
def bedrock_model(aws_region) -> ChatBedrock:
    """Create ChatBedrock model instance for integration tests."""
    return ChatBedrock(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region=aws_region,
    )

########################################################
# Agents
########################################################

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


@pytest.fixture(scope="session")
def agent_tools() -> list[Tool]:
    """Return the list of tools for agent tests."""
    return [add, multiply, get_weather]
