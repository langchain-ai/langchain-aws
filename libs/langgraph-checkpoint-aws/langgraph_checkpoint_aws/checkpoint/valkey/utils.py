"""Utility functions for Valkey client configuration."""

from __future__ import annotations

import logging
from importlib.metadata import version
from typing import Any

logger = logging.getLogger(__name__)

# Package information
LIBRARY_NAME = "langgraph_checkpoint_aws"
try:
    LIBRARY_VERSION = version("langgraph-checkpoint-aws")
except Exception:
    # Fallback version if package is not installed
    LIBRARY_VERSION = "1.0.0"


def set_client_info(client: Any) -> None:
    """Set CLIENT SETINFO for library name and version on a Valkey client.

    This function calls the CLIENT SETINFO command to identify the client
    library and version to the Valkey server for monitoring and debugging purposes.

    Args:
        client: Valkey client instance (sync or async)
    """
    try:
        # CLIENT SETINFO lib-name <library-name>
        client.execute_command("CLIENT", "SETINFO", "lib-name", LIBRARY_NAME)
        # CLIENT SETINFO lib-ver <library-version>
        client.execute_command("CLIENT", "SETINFO", "lib-ver", LIBRARY_VERSION)
        logger.debug(f"Set client info: {LIBRARY_NAME} v{LIBRARY_VERSION}")
    except Exception as e:
        # Don't fail if CLIENT SETINFO is not supported or fails
        logger.debug(f"Failed to set client info: {e}")


async def aset_client_info(client: Any) -> None:
    """Set CLIENT SETINFO for library name and version on an async Valkey client.

    This function calls the CLIENT SETINFO command to identify the client
    library and version to the Valkey server for monitoring and debugging purposes.

    Args:
        client: Async Valkey client instance
    """
    try:
        # CLIENT SETINFO lib-name <library-name>
        await client.execute_command("CLIENT", "SETINFO", "lib-name", LIBRARY_NAME)
        # CLIENT SETINFO lib-ver <library-version>
        await client.execute_command("CLIENT", "SETINFO", "lib-ver", LIBRARY_VERSION)
        logger.debug(f"Set client info: {LIBRARY_NAME} v{LIBRARY_VERSION}")
    except Exception as e:
        # Don't fail if CLIENT SETINFO is not supported or fails
        logger.debug(f"Failed to set client info: {e}")
