import os
import random
import string

import pytest

from langgraph_checkpoint_aws import AgentCoreMemorySaver


@pytest.fixture(scope="session")
def agentcore_memory_id() -> str:
    """Get memory ID from environment variable."""
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        pytest.skip("AGENTCORE_MEMORY_ID env variable not set")
    return memory_id


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


@pytest.fixture(scope="function")
def agentcore_memory_saver(
    agentcore_memory_id,
    aws_region,
) -> AgentCoreMemorySaver:
    """Create AgentCoreMemorySaver instance."""
    return AgentCoreMemorySaver(
        memory_id=agentcore_memory_id,
        region_name=aws_region,
    )
