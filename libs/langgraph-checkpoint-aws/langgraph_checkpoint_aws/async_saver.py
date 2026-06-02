"""Backward-compatible submodule import shim."""

import warnings

from langgraph_checkpoint_aws.checkpoint.bedrock_sessions.async_saver import (
    AsyncBedrockSessionSaver,
)

warnings.warn(
    "Importing from 'langgraph_checkpoint_aws.async_saver' is deprecated and will be "
    "removed soon. "
    "Use 'from langgraph_checkpoint_aws import AsyncBedrockSessionSaver' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AsyncBedrockSessionSaver"]
