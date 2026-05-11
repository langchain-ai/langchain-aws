"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_backend_aws import S3Backend

from ._helpers import _make_backend, _s3_object_response

# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    """Tests for S3Backend initialization."""

    def test_keyword_only_init(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(bucket="b", client=mock_client)
        assert backend._bucket == "b"
        assert backend._prefix == ""

    def test_prefix_normalization(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", prefix="foo/bar/", client=mock_client
        )
        assert backend._prefix == "foo/bar/"

    def test_prefix_normalization_no_trailing_slash(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", prefix="foo/bar", client=mock_client
        )
        assert backend._prefix == "foo/bar/"

    def test_config_normalized_prefix_property(self) -> None:
        # ``S3BackendConfig.normalized_prefix`` is the single source of
        # truth for the slash-shape contract; ``S3Backend.__init__``
        # consumes it directly so the boundary string cannot drift
        # between callers.
        from langchain_backend_aws.s3._config import S3BackendConfig

        assert S3BackendConfig(bucket="b", prefix="").normalized_prefix == ""
        assert (
            S3BackendConfig(bucket="b", prefix="foo/bar").normalized_prefix
            == "foo/bar/"
        )
        assert (
            S3BackendConfig(bucket="b", prefix="/foo/bar/").normalized_prefix
            == "foo/bar/"
        )

    def test_empty_prefix(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(bucket="b", prefix="", client=mock_client)
        assert backend._prefix == ""

    @patch("langchain_backend_aws.s3._config.boto3")
    def test_creates_client_with_credentials(self, mock_boto3: MagicMock) -> None:
        S3Backend.from_kwargs(
            bucket="b",
            region_name="us-west-2",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        mock_boto3.client.assert_called_once()
        call = mock_boto3.client.call_args
        assert call.args == ("s3",)
        assert call.kwargs["region_name"] == "us-west-2"
        assert call.kwargs["aws_access_key_id"] == "AKID"
        assert call.kwargs["aws_secret_access_key"] == "SECRET"
        assert "config" in call.kwargs

    def test_accepts_config_object(self) -> None:
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(bucket="bkt", prefix="px")
        mock_client = MagicMock()
        backend = S3Backend(config, client=mock_client)
        assert backend._bucket == "bkt"
        assert backend._prefix == "px/"

    def test_constructor_requires_config(self) -> None:
        # ``S3Backend()`` without a config raises ``TypeError`` from the
        # signature itself (no default), which is the expected Python
        # behavior for a required positional argument.
        with pytest.raises(TypeError):
            S3Backend()  # type: ignore[call-arg]

    def test_repr_does_not_leak_credentials(self) -> None:
        """Secret-bearing fields are excluded from the dataclass repr."""
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(
            bucket="bkt",
            aws_access_key_id="AKID-SHOULD-NOT-LEAK",
            aws_secret_access_key="SUPER-SECRET",
            aws_session_token="SESSION-TOKEN",
        )
        rendered = repr(config)
        assert "AKID-SHOULD-NOT-LEAK" not in rendered
        assert "SUPER-SECRET" not in rendered
        assert "SESSION-TOKEN" not in rendered

    @patch("langchain_backend_aws.s3._config.boto3")
    def test_extra_boto_config_explicit_keys_win(self, mock_boto3: MagicMock) -> None:
        """Explicit config attributes override colliding ``extra_boto_config`` keys."""
        from langchain_backend_aws import S3BackendConfig

        S3Backend(
            S3BackendConfig(
                bucket="b",
                max_retries=7,
                read_timeout=99.0,
                extra_boto_config={
                    # Should be silently filtered out — explicit fields win.
                    "retries": {"max_attempts": 1, "mode": "standard"},
                    "read_timeout": 1.0,
                    # An orthogonal option should still be forwarded.
                    "signature_version": "s3v4",
                },
            )
        )
        mock_boto3.client.assert_called_once()
        boto_config = mock_boto3.client.call_args.kwargs["config"]
        assert boto_config.retries["max_attempts"] == 7
        assert boto_config.read_timeout == 99.0
        assert boto_config.signature_version == "s3v4"

    def test_extra_boto_config_user_agent_full_override_rejected(self) -> None:
        """``user_agent`` full override is excluded from the allowlist.

        The parent ``langchain-aws`` package patches the boto3 client
        with a framework user-agent header for AWS attribution; a full
        ``user_agent`` override would silently strip that identifier.
        Only ``user_agent_extra`` is permitted.
        """
        from langchain_backend_aws import S3BackendConfig

        with pytest.raises(ValueError, match="unsupported keys"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"user_agent": "evil/1.0"},
            )

    def test_extra_boto_config_user_agent_extra_accepted(self) -> None:
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"user_agent_extra": "myapp/1.0"},
        )
        assert config.extra_boto_config["user_agent_extra"] == "myapp/1.0"


# ------------------------------------------------------------------
# Async API (inherited from BackendProtocol)
# ------------------------------------------------------------------


class TestAsyncAPI:
    """Tests for the async methods inherited from BackendProtocol.

    BackendProtocol provides default async implementations that delegate
    to the sync methods via ``asyncio.to_thread``. These tests verify the
    delegation works through S3Backend without override.
    """

    async def test_aread_delegates_to_read(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"hello\nworld")

        result = await backend.aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "hello\nworld"

    async def test_awrite_delegates_to_write(self) -> None:
        backend, mock = _make_backend()
        result = await backend.awrite("/file.txt", "content")
        assert result.error is None
        assert result.path == "/file.txt"
        mock.put_object.assert_called_once()

    async def test_als_delegates_to_ls(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [], "CommonPrefixes": []}]
        mock.get_paginator.return_value = paginator

        result = await backend.als("/")
        assert result.error is None
        assert result.entries == []
