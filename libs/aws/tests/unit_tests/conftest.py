"""Shared pytest fixtures for unit tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def disable_openai_socket_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip langchain-openai's TCP keepalive probe, which opens a socket."""
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")


@pytest.fixture(autouse=True)
def mock_boto3_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock boto3 client creation to prevent network calls in unit tests."""
    mock_client = MagicMock()

    # Mock boto3.client directly
    monkeypatch.setattr("boto3.client", lambda **_: mock_client)

    # Also mock the create_aws_client function
    monkeypatch.setattr(
        "langchain_aws.utils.create_aws_client", lambda **_: mock_client
    )

    # Mock boto3.Session.client as well
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    monkeypatch.setattr("boto3.Session", lambda **_: mock_session)

    return mock_client
