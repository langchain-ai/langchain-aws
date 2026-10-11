"""Shared helpers for S3Backend unit tests.

Extracted from the original monolithic ``test_backend.py`` so each
operation can live in its own focused test module without duplicating
fixture code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from langchain_backend_aws import S3Backend, S3BackendConfig


def _client_error(code: str = "NoSuchKey", message: str = "Not Found") -> ClientError:
    """Create a botocore ClientError with the given code."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "TestOperation",
    )


def _make_backend(
    bucket: str = "test-bucket",
    prefix: str = "",
) -> tuple[S3Backend, MagicMock]:
    """Create a backend with a mocked S3 client."""
    mock_client = MagicMock()
    config = S3BackendConfig(bucket=bucket, prefix=prefix)
    backend = S3Backend(config, client=mock_client)
    return backend, mock_client


def _s3_object_response(
    body: bytes,
    last_modified: datetime | None = None,
    etag: str = '"deadbeef"',
) -> dict[str, Any]:
    """Build a mock S3 get_object response."""
    stream = MagicMock()
    stream.read.return_value = body
    return {
        "Body": stream,
        "ContentLength": len(body),
        "ETag": etag,
        "LastModified": last_modified or datetime(2025, 3, 7, 12, 0, 0, tzinfo=UTC),
    }
