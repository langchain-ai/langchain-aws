"""Unit tests for :func:`langchain_backend_aws.s3._config.build_client`.

``build_client`` documents a precondition that ``S3BackendConfig`` has
flowed through ``__post_init__`` (so SSRF / proxy validation has run).
The tests below exercise the happy path with a normally-constructed
config and confirm the explicit fields override ``extra_boto_config``
duplicates without raising ``TypeError`` from a duplicated kwarg.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from langchain_backend_aws.s3._config import S3BackendConfig, build_client


def _capture_boto3_client() -> tuple[Any, dict[str, Any]]:
    """Patch :func:`boto3.client` and capture the kwargs forwarded to it."""
    captured: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> str:
        captured["service"] = service
        captured["kwargs"] = kwargs
        return "fake-client"

    return fake_client, captured


class TestBuildClient:
    def test_forwards_explicit_fields(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            region_name="us-east-1",
            aws_access_key_id="AKIA",
            aws_secret_access_key="secret",
            aws_session_token="token",
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            client = build_client(config)
        assert client == "fake-client"
        assert captured["service"] == "s3"
        kwargs = captured["kwargs"]
        assert kwargs["region_name"] == "us-east-1"
        assert kwargs["aws_access_key_id"] == "AKIA"
        assert kwargs["aws_secret_access_key"] == "secret"
        assert kwargs["aws_session_token"] == "token"
        assert "config" in kwargs

    def test_drops_extra_boto_keys_overlapping_explicit_fields(self) -> None:
        # ``retries`` etc. are dropped by ``build_client`` so botocore
        # never sees a duplicated kwarg. Use ``extra_boto_config`` keys
        # that pass ``_validate_extra_boto_keys`` to confirm the drop is
        # silent rather than rejected.
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "retries": {"max_attempts": 99, "mode": "standard"},
                "connect_timeout": 999.0,
                "user_agent_extra": "test/1.0",
            },
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            build_client(config)
        boto_config = captured["kwargs"]["config"]
        # Explicit field wins for retries.
        assert boto_config.retries["max_attempts"] == config.max_retries
        assert boto_config.retries["mode"] == "adaptive"
        # The allow-listed extra (not in ``_EXPLICIT_BOTO_KEYS``) is forwarded.
        assert boto_config.user_agent_extra == "test/1.0"

    def test_endpoint_url_forwarded_when_set(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="https://s3.example.com",
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            build_client(config)
        assert captured["kwargs"]["endpoint_url"] == "https://s3.example.com"
