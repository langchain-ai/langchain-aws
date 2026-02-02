from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glide_sync import GlideClient, GlideClusterClient

    GlideClientType = GlideClient | GlideClusterClient


def get_client(valkey_url: str, **kwargs: Any) -> GlideClientType:
    """Get a GLIDE client from the connection url.

    Args:
        valkey_url: Connection URL for Valkey server.
        **kwargs: Additional arguments to pass to GLIDE client.

    Returns:
        GLIDE client instance.

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
            NodeAddress,
        )
    except ImportError:
        raise ImportError(
            "Could not import valkey-glide-sync python package. "
            "Please install it with `pip install valkey-glide-sync>=2.0.0`."
        )

    # Parse URL
    host, port = _parse_valkey_url(valkey_url)
    addresses = [NodeAddress(host, port)]

    # Try cluster first
    try:
        config = GlideClusterClientConfiguration(addresses=addresses, **kwargs)
        client = GlideClusterClient.create(config)
        return client
    except Exception:
        # Fall back to standalone
        config = GlideClientConfiguration(addresses=addresses, **kwargs)
        client = GlideClient.create(config)
        return client


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
