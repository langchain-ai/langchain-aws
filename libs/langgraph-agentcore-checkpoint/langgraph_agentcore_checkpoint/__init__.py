"""
LangGraph Checkpoint with AgentCore Memory - A LangChain checkpointer implementation using Amazon Bedrock AgentCore Memory.
"""

from langgraph_agentcore_checkpoint.saver import AgentCoreMemorySaver

__version__ = "0.1.1"
SDK_USER_AGENT = f"LangGraphAgentcoreCheckpoint#{__version__}"

__all__ = ["AgentCoreMemorySaver"]
