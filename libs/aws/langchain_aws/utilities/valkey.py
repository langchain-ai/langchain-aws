from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glide_sync import GlideClient, GlideClusterClient

GlideClientType: TypeAlias = Union["GlideClient", "GlideClusterClient"]


def get_client(valkey_url: str, **kwargs: Any) -> GlideClientType:
    """Get a GLIDE client from the connection url.

    Args:
        valkey_url: Connection URL for Valkey server.
        **kwargs: Additional arguments to pass to GLIDE client.

    Returns:
        GLIDE client instance.

    Raises:
        ImportError: If valkey-glide-sync package is not installed.
        ConnectionError: If unable to connect to Valkey server.
        TimeoutError: If connection attempt times out.
        ValueError: If connection configuration is invalid.

    Example:
        ```python
        from langchain_aws.utilities.valkey import get_client
        valkey_client = get_client(
            valkey_url="valkey://localhost:6379"
        )
        ```
    """
    try:
        from glide_sync import (
            GlideClient,
            GlideClientConfiguration,
            GlideClusterClient,
            GlideClusterClientConfiguration,
            GlideError,
            NodeAddress,
        )
    except ImportError as e:
        msg = (
            "Could not import valkey-glide-sync python package. "
            "Please install it with `pip install langchain-aws[valkey]`."
        )
        raise ImportError(msg) from e

    # Parse URL
    host, port = _parse_valkey_url(valkey_url)
    addresses = [NodeAddress(host, port)]

    # Try cluster first, fall back to standalone
    try:
        cluster_config = GlideClusterClientConfiguration(addresses=addresses, **kwargs)
        return GlideClusterClient.create(cluster_config)
    except GlideError as e:
        logger.debug(f"Cluster connection failed, falling back to standalone: {e}")
        standalone_config = GlideClientConfiguration(addresses=addresses, **kwargs)
        return GlideClient.create(standalone_config)


def _parse_valkey_url(url: str) -> tuple[str, int]:
    """Parse Valkey URL to extract host and port."""
    # Remove protocol
    if "://" in url:
        url = url.split("://", 1)[1]

    # Remove credentials if present
    if "@" in url:
        url = url.split("@", 1)[1]

    # Extract host and port
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        # Remove any path after port
        port_str = port_str.split("/")[0]
        port = int(port_str)
    else:
        host = url
        port = 6379

    return host, port
