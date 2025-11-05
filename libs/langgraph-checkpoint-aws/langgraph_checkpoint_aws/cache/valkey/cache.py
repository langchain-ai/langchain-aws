"""Valkey cache implementation with enhanced performance and reliability."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

try:
    from valkey import Valkey
    from valkey.connection import ConnectionPool
    from valkey.exceptions import ConnectionError, TimeoutError
except ImportError as e:
    raise ImportError(
        "The 'valkey' package is required to use ValkeyCache. "
        "Install it with: pip install 'langgraph-checkpoint-aws[valkey]'"
    ) from e

from ...checkpoint.valkey.utils import set_client_info

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants for better maintainability
DEFAULT_PREFIX = "langgraph:cache:"
MAX_SET_BATCH_SIZE = 100
DEFAULT_POOL_TIMEOUT = 30.0
ENCODING_SEPARATOR = b":"


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
        prefix: str = DEFAULT_PREFIX,
        ttl: float | None = None,
    ) -> None:
        """Initialize the cache with a Valkey client.

        Args:
            client: Valkey client instance
            serde: Serializer to use for values
            prefix: Key prefix for all cached values (must end with ':' or '/')
            ttl: Optional default TTL in seconds (must be positive)

        Raises:
            ValueError: If TTL is negative or prefix is invalid
        """
        super().__init__(serde=serde)

        # Validate inputs
        if ttl is not None and ttl <= 0:
            raise ValueError(f"TTL must be positive, got {ttl}")

        if not prefix:
            raise ValueError("Prefix cannot be empty")

        # Ensure prefix ends with separator for consistent key structure
        if not prefix.endswith((":", "/")):
            prefix += ":"

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
        prefix: str = DEFAULT_PREFIX,
        ttl_seconds: float | None = None,
        serde: SerializerProtocol | None = None,
    ) -> Generator[ValkeyCache[ValueT], None, None]:
        """Create a ValkeyCache from an existing connection pool.

        This allows reusing an existing pool across multiple caches or
        sharing a pool with other components.

        Args:
            pool: Existing Valkey connection pool
            ssl: Whether to use SSL connection
            prefix: Key prefix for all cached values
            ttl_seconds: Optional TTL in seconds for cached values
            serde: Optional serializer for values

        Raises:
            ValueError: If pool is None

        Example:
            ```python
            # Create custom pool
            pool = ConnectionPool.from_url(
                "valkey://localhost:6379",
                max_connections=20,
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
        if pool is None:
            raise ValueError("Connection pool cannot be None")

        try:
            client = Valkey(connection_pool=pool, ssl=ssl)
            cache = cls(client, serde=serde, prefix=prefix, ttl=ttl_seconds)
            yield cache
        except Exception as e:
            logger.error(f"Failed to create cache from pool: {e}")
            raise

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
        """Asynchronously get the cached values for the given keys.

        Args:
            keys: Sequence of (namespace, key) tuples to retrieve

        Returns:
            Dictionary mapping keys to their cached values

        Raises:
            ConnectionError: If unable to connect to Valkey
            TimeoutError: If operation times out
        """
        if not keys:
            return {}

        # Build Valkey keys with validation
        valkey_keys = []
        for ns, key in keys:
            try:
                valkey_keys.append(self._make_key(ns, key))
            except ValueError as e:
                logger.error(f"Invalid key {key} in namespace {ns}: {e}")
                continue

        if not valkey_keys:
            return {}

        # Get values using batch get with retry logic
        try:
            raw_values = cast(
                list[bytes | None],
                await asyncio.to_thread(self.client.mget, valkey_keys),
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection/timeout error getting cached values: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting cached values for {len(keys)} keys: {e}")
            return {}

        # Ensure raw_values is a list (handle potential type issues)
        if raw_values is None:
            logger.warning("Received None from mget operation")
            return {}

        values: dict[FullKey, ValueT] = {}
        successful_deserializations = 0

        for i, raw_value in enumerate(raw_values):
            if raw_value is not None and i < len(keys):
                try:
                    # Deserialize the value - handle malformed data gracefully
                    if ENCODING_SEPARATOR not in raw_value:
                        logger.error(
                            "Malformed cached value for key %s: "
                            "missing encoding separator",
                            keys[i],
                        )
                        continue

                    encoding, data = raw_value.split(ENCODING_SEPARATOR, 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                    successful_deserializations += 1
                except Exception as e:
                    logger.error(
                        "Error deserializing cached value for key %s: %s", keys[i], e
                    )
                    continue

        logger.debug(
            "Successfully retrieved %d/%d cached values",
            successful_deserializations,
            len(keys),
        )
        return values

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        asyncio.run(self.aset(pairs))

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs.

        Args:
            pairs: Mapping of (namespace, key) tuples to (value, ttl) tuples

        Raises:
            ConnectionError: If unable to connect to Valkey
            TimeoutError: If operation times out
            ValueError: If TTL values are invalid
        """
        if not pairs:
            return

        # Process in batches to avoid overwhelming the server
        pairs_list = list(pairs.items())

        for i in range(0, len(pairs_list), MAX_SET_BATCH_SIZE):
            batch = pairs_list[i : i + MAX_SET_BATCH_SIZE]
            await self._set_batch(batch)

    async def _set_batch(
        self, batch: list[tuple[FullKey, tuple[ValueT, int | None]]]
    ) -> None:
        """Set a batch of key-value pairs.

        Args:
            batch: List of ((namespace, key), (value, ttl)) tuples
        """
        # Get pipeline with proper typing
        pipe = self.client.pipeline(transaction=True)
        successful_operations = 0

        # Process each key-value pair
        for (ns, key), (value, ttl) in batch:
            try:
                valkey_key = self._make_key(ns, key)
                encoding, data = self.serde.dumps_typed(value)
                serialized_value = encoding.encode() + ENCODING_SEPARATOR + data

                # Determine final TTL - validate it's positive
                final_ttl = ttl if ttl is not None else self.ttl
                if final_ttl is not None:
                    if final_ttl <= 0:
                        logger.error(f"Invalid TTL {final_ttl} for key {key}, skipping")
                        continue
                    pipe.setex(valkey_key, int(final_ttl), serialized_value)
                else:
                    pipe.set(valkey_key, serialized_value)

                successful_operations += 1
            except Exception as e:
                logger.error(f"Error preparing cached value for key {key}: {e}")
                continue

        # Execute all commands in the pipeline
        if successful_operations > 0:
            try:
                await asyncio.to_thread(pipe.execute)
                logger.debug("Successfully set %d cache entries", successful_operations)
            except (ConnectionError, TimeoutError) as e:
                logger.error(
                    "Connection/timeout error executing pipeline with %d "
                    "operations: %s",
                    successful_operations,
                    e,
                )
                raise
            except Exception as e:
                logger.error(
                    "Error executing pipeline with %d operations: %s",
                    successful_operations,
                    e,
                )
                raise
        else:
            logger.warning("No valid cache operations to execute in batch")

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.

        Uses Valkey's keys pattern matching to find and delete keys.
        """
        asyncio.run(self.aclear(namespaces))

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.

        Uses Valkey's keys pattern matching to find and delete keys.
        Handles large key sets by chunking deletions to avoid command limits.
        """
        # Maximum number of keys to delete in a single command
        # Valkey/Redis typically supports up to ~1M arguments, but we use a
        # conservative limit
        max_delete_batch_size = 1000

        try:
            if namespaces is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix}*"
                keys = cast(
                    list[str], await asyncio.to_thread(self.client.keys, pattern)
                )
                if keys:
                    deleted_count = await self._delete_keys_in_batches(
                        keys, max_delete_batch_size
                    )
                    logger.debug(
                        "Cleared %d keys with pattern '%s'", deleted_count, pattern
                    )
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
                    # Remove duplicates while preserving order
                    unique_keys = list(dict.fromkeys(keys_to_delete))
                    deleted_count = await self._delete_keys_in_batches(
                        unique_keys, max_delete_batch_size
                    )
                    logger.debug(
                        "Cleared %d keys from %d namespace(s)",
                        deleted_count,
                        len(namespaces),
                    )
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    async def _delete_keys_in_batches(self, keys: list[str], batch_size: int) -> int:
        """Delete keys in batches to avoid command argument limits.

        Args:
            keys: List of keys to delete
            batch_size: Maximum number of keys to delete per batch

        Returns:
            Total number of keys deleted
        """
        if not keys:
            return 0

        total_deleted = 0

        # Process keys in batches
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            try:
                deleted = cast(int, await asyncio.to_thread(self.client.delete, *batch))
                total_deleted += deleted
            except Exception as e:
                logger.error(f"Error deleting batch of {len(batch)} keys: {e}")
                # Continue with next batch instead of failing completely
                continue

        return total_deleted
