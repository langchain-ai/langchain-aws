from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from valkey import Valkey as ValkeyType  # type: ignore[import-untyped]


def get_client(valkey_url: str, **kwargs: Any) -> ValkeyType:
    """Get a Valkey client from the connection url.

    Args:
        valkey_url: Connection URL for Valkey server.
        **kwargs: Additional arguments to pass to Valkey client.

    Returns:
        Valkey client instance.

    Example:
        ```python
        from langchain_aws.utilities.valkey import get_client
        valkey_client = get_client(
            valkey_url="valkey://username:password@localhost:6379"
        )
        ```
    """
    try:
        import valkey  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "Could not import valkey python package. "
            "Please install it with `pip install valkey>=6.0.0`."
        )

    # Connect to Valkey server from url, reconnect with cluster client if needed
    valkey_client = valkey.from_url(valkey_url, **kwargs)
    if _check_for_cluster(valkey_client):
        valkey_client.close()
        valkey_client = _valkey_cluster_client(valkey_url, **kwargs)

    return valkey_client


def _check_for_cluster(valkey_client: ValkeyType) -> bool:
    import valkey

    try:
        cluster_info = valkey_client.info("cluster")
        return cluster_info["cluster_enabled"] == 1
    except valkey.exceptions.ValkeyError:
        return False


def _valkey_cluster_client(valkey_url: str, **kwargs: Any) -> ValkeyType:
    from valkey.cluster import ValkeyCluster  # type: ignore[import-untyped]

    return ValkeyCluster.from_url(valkey_url, **kwargs)  # type: ignore[return-value]
