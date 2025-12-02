"""Search strategies and utilities for ValkeyStore.

This package provides modular search strategies for both synchronous and
asynchronous operations, along with shared adapters and helpers.
"""

# Adapters
from .adapters import (
    AsyncClientAdapter,
    ClientAdapter,
    DocumentAdapter,
    EmbeddingAdapter,
    SyncClientAdapter,
)

# Async Strategies
from .async_strategies import (
    AsyncHashSearchStrategy,
    AsyncKeyPatternSearchStrategy,
    AsyncSearchStrategy,
    AsyncSearchStrategyManager,
    AsyncVectorSearchStrategy,
)

# Helpers
from .helpers import BaseSearchHelper

# Protocols
from .protocols import ValkeyClientProtocol

# Sync Strategies
from .sync_strategies import (
    HashSearchStrategy,
    KeyPatternSearchStrategy,
    SearchStrategy,
    SearchStrategyManager,
    VectorSearchStrategy,
)

__all__ = [
    # Adapters
    "EmbeddingAdapter",
    "ClientAdapter",
    "SyncClientAdapter",
    "AsyncClientAdapter",
    "DocumentAdapter",
    # Helpers
    "BaseSearchHelper",
    # Protocols
    "ValkeyClientProtocol",
    # Sync Strategies
    "SearchStrategy",
    "VectorSearchStrategy",
    "HashSearchStrategy",
    "KeyPatternSearchStrategy",
    "SearchStrategyManager",
    # Async Strategies
    "AsyncSearchStrategy",
    "AsyncVectorSearchStrategy",
    "AsyncHashSearchStrategy",
    "AsyncKeyPatternSearchStrategy",
    "AsyncSearchStrategyManager",
]
