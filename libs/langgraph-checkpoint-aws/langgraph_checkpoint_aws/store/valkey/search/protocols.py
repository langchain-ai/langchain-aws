"""Protocol definitions for ValkeyStore search."""

from __future__ import annotations

from typing import Any, Protocol


class ValkeyClientProtocol(Protocol):
    """Protocol for Valkey client interface."""

    def hgetall(self, name: str) -> Any: ...
    def scan(
        self, cursor: int, match: str | None = None, count: int | None = None
    ) -> Any: ...
    def keys(self, pattern: str) -> Any: ...
    def get(self, name: Any) -> Any: ...
    def ft(self, index_name: str) -> Any: ...
    def expire(self, name: str, time: int) -> Any: ...


__all__ = ["ValkeyClientProtocol"]
