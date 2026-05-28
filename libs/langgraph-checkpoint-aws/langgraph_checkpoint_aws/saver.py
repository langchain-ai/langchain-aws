"""Backward-compatible submodule import shim."""

import warnings

from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.saver import (
    BedrockSessionSaver,
)

warnings.warn(
    "Importing from 'langgraph_checkpoint_aws.saver' is deprecated and will be "
    "removed soon. "
    "Use 'from langgraph_checkpoint_aws import BedrockSessionSaver' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BedrockSessionSaver"]
