"""
LangGraph Checkpoint AWS - A LangChain checkpointer implementation using Bedrock Session Management Service.
"""

from langgraph_checkpoint_aws.agentcore.saver import (
    AgentCoreMemorySaver,
)
from langgraph_checkpoint_aws.agentcore.valkey import (
    AgentCoreValkeySaver,
)

__version__ = "0.1.2"
SDK_USER_AGENT = f"LangGraphCheckpointAWS#{__version__}"

# Expose the saver class at the package level
__all__ = [
    "AgentCoreMemorySaver",
    "AgentCoreValkeySaver",
    "SDK_USER_AGENT",
]
