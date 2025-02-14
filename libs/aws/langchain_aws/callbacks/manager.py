from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Generator,
    Optional,
)

from langchain_core.tracers.context import register_configure_hook

from langchain_aws.callbacks.bedrock_callback import (
    BedrockTokenUsageCallbackHandler,
)

logger = logging.getLogger(__name__)


bedrock_callback_var: (ContextVar)[
    Optional[BedrockTokenUsageCallbackHandler]
] = ContextVar("bedrock_anthropic_callback", default=None)

register_configure_hook(bedrock_callback_var, True)


@contextmanager
def get_bedrock_callback() -> (
    Generator[BedrockTokenUsageCallbackHandler, None, None]
):
    """Get the Bedrock callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        BedrockTokenUsageCallbackHandler:
            The Bedrock callback handler.

    Example:
        >>> with get_bedrock_callback() as cb:
        ...     # Use the Bedrock callback handler
    """
    cb = BedrockTokenUsageCallbackHandler()
    bedrock_callback_var.set(cb)
    yield cb
    bedrock_callback_var.set(None)