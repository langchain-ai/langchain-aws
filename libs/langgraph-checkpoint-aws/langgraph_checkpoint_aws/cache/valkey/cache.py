"""Valkey cache implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol
from valkey import Valkey
from valkey.connection import ConnectionPool

from ...checkpoint.valkey.utils import set_client_info

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ValkeyCache(BaseCache[ValueT]):
    """Valkey-based cache implementation with TTL support.

    Features:
    - TTL support
    - Connection pool support for better performance
    - Namespace organization using basic prefix search (no vector index required)
    - Async operations
    - Batch operations

    Example:
        ```python
        # Using connection string
        with ValkeyCache.from_conn_string(
            "valkey://localhost:6379",
            ttl_seconds=3600.0,  # 1 hour TTL
            pool_size=10  # Connection pool size
        ) as cache:
            # Use cache...

        # Using connection pool
        pool = ConnectionPool(
            "valkey://localhost:6379",
            min_size=5,
            max_connections=20,
            timeout=30
        )
        with ValkeyCache.from_pool(
            pool,
            prefix="langgraph:cache:"
        ) as cache:
            # Use cache with custom pool...

        # Or direct initialization
        cache = ValkeyCache(
            Valkey("valkey://localhost:6379"),
            prefix="langgraph:cache:"
        )
        ```
    """

    def __init__(
        self,
        client: Valkey,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
        ttl: float | None = None,
    ) -> None:
        """Initialize the cache with a Valkey client.

        Args:
            client: Valkey client instance
            serde: Serializer to use for values
            prefix: Key prefix for all cached values
            ttl: Optional default TTL in seconds
        """
        super().__init__(serde=serde)
        self.client = client
        self.prefix = prefix
        self.ttl = int(ttl) if ttl else None
        set_client_info(client)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        prefix: str = "langgraph:cache:",
        ttl_seconds: float | None = None,
        serde: SerializerProtocol | None = None,
        pool_size: int | None = None,
        pool_timeout: float | None = None,
    ) -> Generator[ValkeyCache[ValueT], None, None]:
        """Create a ValkeyCache from a connection string.

        Args:
            conn_string: Valkey connection string (e.g. "valkey://localhost:6379")
            prefix: Key prefix for all cached values
            ttl_seconds: Optional TTL in seconds for cached values
            serde: Optional serializer for values
            pool_size: Optional connection pool size
            pool_timeout: Optional pool timeout in seconds

        Example:
            ```python
            with ValkeyCache.from_conn_string(
                "valkey://localhost:6379",
                ttl_seconds=3600.0,  # 1 hour TTL
                pool_size=10  # Use connection pool
            ) as cache:
                # Use cache with automatic cleanup
                await cache.aset({
                    (("ns1",), "key1"): ({"value": 1}, 60)
                })
            ```
        """
        try:
            if pool_size:
                # Create connection pool
                pool = ConnectionPool.from_url(  # type: ignore[no-untyped-call]
                    url=conn_string,
                    max_connections=pool_size,
                    timeout=pool_timeout or 30.0,
                )
                # When using a pool, don't pass the connection string again
                client = Valkey.from_pool(connection_pool=pool)
            else:
                # Single connection
                client = Valkey.from_url(conn_string)

            # Don't call set_client_info here - let __init__ handle it
            cache = cls(client, serde=serde, prefix=prefix, ttl=ttl_seconds)
            yield cache
        finally:
            # Cleanup will be handled by pool/client
            pass

    @classmethod
    @contextmanager
    def from_pool(
        cls,
        pool: ConnectionPool,
        *,
        ssl: bool = False,
        prefix: str = "langgraph:cache:",
        ttl_seconds: float | None = None,
        serde: SerializerProtocol | None = None,
    ) -> Generator[ValkeyCache[ValueT], None, None]:
        """Create a ValkeyCache from an existing connection pool.

        This allows reusing an existing pool across multiple caches or
        sharing a pool with other components.

        Args:
            pool: Existing Valkey connection pool
            prefix: Key prefix for all cached values
            ttl_seconds: Optional TTL in seconds for cached values
            serde: Optional serializer for values

        Example:
            ```python
            # Create custom pool
            pool = ConnectionPool(
                "valkey://localhost:6379",
                min_size=5,
                max_size=20,
                timeout=30
            )

            # Use pool with cache
            with ValkeyCache.from_pool(
                pool,
                ttl_seconds=3600.0
            ) as cache:
                await cache.aset({
                    (("ns1",), "key1"): ({"value": 1}, 60)
                })
            ```
        """
        try:
            client = Valkey(connection_pool=pool, ssl=ssl)
            # Don't call set_client_info here - let __init__ handle it
            cache = cls(client, serde=serde, prefix=prefix, ttl=ttl_seconds)
            yield cache
        finally:
            # Pool cleanup handled by owner
            pass

    def _make_key(self, ns: Namespace, key: str) -> str:
        """Create a Valkey key from namespace and key."""
        ns_str = "/".join(ns) if ns else ""
        return f"{self.prefix}{ns_str}/{key}" if ns_str else f"{self.prefix}{key}"

    def _parse_key(self, valkey_key: str) -> tuple[Namespace, str]:
        """Parse a Valkey key back to namespace and key."""
        if not valkey_key.startswith(self.prefix):
            raise ValueError(
                f"Key {valkey_key} does not start with prefix {self.prefix}"
            )

        remaining = valkey_key[len(self.prefix) :]
        if "/" in remaining:
            parts = remaining.split("/")
            key = parts[-1]
            ns_parts = parts[:-1]
            return (tuple(ns_parts), key)
        else:
            return (tuple(), remaining)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        return asyncio.run(self.aget(keys))

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        if not keys:
            return {}

        # Build Valkey keys
        valkey_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values using batch get
        try:
            raw_values = cast(
                list[bytes | None],
                await asyncio.to_thread(self.client.mget, valkey_keys),
            )
        except Exception as e:
            logger.error(f"Error getting cached values: {e}")
            return {}

        # Ensure raw_values is a list (handle potential type issues)
        if raw_values is None:
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # Deserialize the value
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception as e:
                    logger.error(f"Error deserializing cached value: {e}")
                    continue

        return values

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        asyncio.run(self.aset(pairs))

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        if not pairs:
            return

        # Get pipeline with proper typing
        pipe = self.client.pipeline(transaction=True)
        # Process each key-value pair
        for (ns, key), (value, ttl) in pairs.items():
            try:
                valkey_key = self._make_key(ns, key)
                encoding, data = self.serde.dumps_typed(value)
                serialized_value = f"{encoding}:".encode() + data

                if ttl is not None or self.ttl is not None:
                    # Use provided TTL (in seconds) or default TTL (already in seconds)
                    final_ttl = ttl if ttl is not None else self.ttl
                    if final_ttl is not None:
                        pipe.setex(valkey_key, final_ttl, serialized_value)
                    else:
                        pipe.set(valkey_key, serialized_value)
                else:
                    pipe.set(valkey_key, serialized_value)
            except Exception as e:
                logger.error(f"Error setting cached value: {e}")
                continue

        # Execute all commands in the pipeline
        try:
            await asyncio.to_thread(pipe.execute)
        except Exception as e:
            logger.error(f"Error executing pipeline: {e}")

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.

        Uses Valkey's keys pattern matching to find and delete keys.
        """
        asyncio.run(self.aclear(namespaces))

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.

        Uses Valkey's keys pattern matching to find and delete keys.
        """
        try:
            if namespaces is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix}*"
                keys = cast(
                    list[str], await asyncio.to_thread(self.client.keys, pattern)
                )
                if keys:
                    await asyncio.to_thread(self.client.delete, *keys)
            else:
                # Clear specific namespaces
                keys_to_delete = []
                for ns in namespaces:
                    ns_str = "/".join(ns) if ns else ""
                    pattern = (
                        f"{self.prefix}{ns_str}/*" if ns_str else f"{self.prefix}*"
                    )
                    keys = cast(
                        list[str], await asyncio.to_thread(self.client.keys, pattern)
                    )
                    keys_to_delete.extend(keys)

                if keys_to_delete:
                    await asyncio.to_thread(self.client.delete, *keys_to_delete)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
