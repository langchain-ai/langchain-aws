"""
Constants and exceptions for AgentCore Memory Checkpoint Saver.
"""

EMPTY_CHANNEL_VALUE = "_empty"


class AgentCoreMemoryError(Exception):
    """Base exception for AgentCore Memory errors."""

    pass


class EventDecodingError(AgentCoreMemoryError):
    """Raised when event decoding fails."""

    pass


class InvalidConfigError(AgentCoreMemoryError):
    """Raised when configuration is invalid."""

    pass


class EventNotFoundError(AgentCoreMemoryError):
    """Raised when expected event is not found."""

    pass
