"""Utility functions for testing."""

import pytest


def is_valkey_available() -> bool:
    try:
        import valkey  # noqa: F401

        return True
    except ImportError:
        return False


def skip_if_valkey_not_available() -> None:
    if not is_valkey_available():
        pytest.skip(
            "valkey dependency is not available",
            allow_module_level=True,
        )
