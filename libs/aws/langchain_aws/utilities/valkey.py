from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeAlias, Union, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glide_sync import GlideClient, GlideClusterClient

GlideClientType: TypeAlias = Union["GlideClient", "GlideClusterClient"]


def get_client(
    valkey_url: str, cluster_mode: Optional[bool] = None, **kwargs: Any
) -> GlideClientType:
    """Get a GLIDE client from the connection url.

    Args:
        valkey_url: Connection URL for Valkey server.
        cluster_mode: If True, create cluster client. If False, create standalone
            client. If None (default), try cluster first and fall back to standalone.
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
            ServerCredentials,
        )
    except ImportError as e:
        msg = (
            "Could not import valkey-glide-sync python package. "
            "Please install it with `pip install langchain-aws[valkey]`."
        )
        raise ImportError(msg) from e

    # Parse URL
    host, port, use_tls, username, password = _parse_valkey_url(valkey_url)
    addresses = [NodeAddress(host, port)]
    
    # Set TLS if specified in URL
    if use_tls and "use_tls" not in kwargs:
        kwargs["use_tls"] = True
    
    # Set credentials if specified in URL
    if username and password and "credentials" not in kwargs:
        kwargs["credentials"] = ServerCredentials(password, username)

    # Create client based on cluster_mode
    if cluster_mode is True:
        # User explicitly wants cluster mode
        cluster_config = GlideClusterClientConfiguration(addresses=addresses, **kwargs)
        return GlideClusterClient.create(cluster_config)
    elif cluster_mode is False:
        # User explicitly wants standalone mode
        standalone_config = GlideClientConfiguration(addresses=addresses, **kwargs)
        return GlideClient.create(standalone_config)
    else:
        # Auto-detect: try cluster first, fall back to standalone
        try:
            cluster_config = GlideClusterClientConfiguration(
                addresses=addresses, **kwargs
            )
            return GlideClusterClient.create(cluster_config)
        except GlideError as e:
            logger.debug(f"Cluster connection failed, falling back to standalone: {e}")
            standalone_config = GlideClientConfiguration(addresses=addresses, **kwargs)
            return GlideClient.create(standalone_config)


def _parse_valkey_url(url: str) -> tuple[str, int, bool, str | None, str | None]:
    """Parse Valkey URL to extract host, port, TLS flag, and credentials."""
    use_tls = False
    username = None
    password = None
    
    # Extract protocol
    if "://" in url:
        protocol, url = url.split("://", 1)
        use_tls = protocol.endswith("ss")  # valkeyss or rediss
    
    # Extract credentials if present (split from right to handle @ in password)
    if "@" in url:
        credentials, url = url.rsplit("@", 1)
        if ":" in credentials:
            username, password = credentials.split(":", 1)
        else:
            password = credentials

    # Extract host and port
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        port_str = port_str.split("/")[0]
        port = int(port_str)
    else:
        host = url.split("/")[0]
        port = 6379

    return host, port, use_tls, username, password
