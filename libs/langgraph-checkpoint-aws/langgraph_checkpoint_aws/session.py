"""Backward-compatible submodule import shim."""

import warnings

from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.session import (
    BedrockAgentRuntimeSessionClient,
)

warnings.warn(
    "Importing from 'langgraph_checkpoint_aws.session' is deprecated and will be "
    "removed soon. "
    "Use 'from langgraph_checkpoint_aws import BedrockAgentRuntimeSessionClient' "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BedrockAgentRuntimeSessionClient"]
