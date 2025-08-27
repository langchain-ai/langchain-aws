"""Shared pytest fixtures for unit tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_aws_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock AWS client creation to prevent network calls in unit tests."""
    mock_client = MagicMock()
    monkeypatch.setattr(
        "langchain_aws.utils.create_aws_client", lambda **_: mock_client
    )
    return mock_client