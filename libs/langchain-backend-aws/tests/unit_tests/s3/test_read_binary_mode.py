"""Tests for ``S3BackendConfig.binary_read_mode`` (base64 vs error)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig

from ._helpers import _s3_object_response


def test_binary_read_mode_error_returns_error() -> None:
    mock_client = MagicMock()
    mock_client.get_object.return_value = _s3_object_response(b"\xff\xfe\x00binary")
    config = S3BackendConfig(bucket="b", prefix="p/", binary_read_mode="error")
    backend = S3Backend(config, client=mock_client)

    result = backend.read("/bin.dat")

    assert result.error is not None
    assert "not UTF-8" in result.error
    assert "download_files" in result.error
    assert result.file_data is None


def test_binary_read_mode_base64_default() -> None:
    mock_client = MagicMock()
    mock_client.get_object.return_value = _s3_object_response(b"\xff\xfe\x00binary")
    backend = S3Backend.from_kwargs(bucket="b", prefix="p/", client=mock_client)

    result = backend.read("/bin.dat")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"
