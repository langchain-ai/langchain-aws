"""``read`` warns when ``offset``/``limit`` are ignored on a non-UTF-8 body.

With ``binary_read_mode="base64"`` the base64 encoding path returns the
full capped body regardless of the requested slice. Operators must see a
WARNING so the silent fall-back does not look like a successful slice.
"""

from __future__ import annotations

import io
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig


def _make_get_object(body: bytes) -> Any:
    client = MagicMock()
    client.get_object.return_value = {
        "Body": io.BytesIO(body),
        "ContentLength": len(body),
        "LastModified": MagicMock(),
        "ETag": '"deadbeef"',
    }
    client.get_object.return_value["LastModified"].astimezone.return_value = MagicMock()
    return client


class TestBinaryReadModeOffsetLimitWarning:
    """Non-default ``offset``/``limit`` warn on the base64 path."""

    def test_non_default_offset_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _make_get_object(b"\xff\xfe\x00\x01binary")
        backend = S3Backend(
            S3BackendConfig(bucket="b", binary_read_mode="base64"),
            client=client,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_backend_aws.s3._read"):
            result = backend.read("/binary.bin", offset=10, limit=5)
        assert result.file_data is not None
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING"
            and "non-UTF-8" in rec.getMessage()
            and "offset=10" in rec.getMessage()
        ]
        assert warnings, "expected a warning when offset/limit ignored on binary body"

    def test_default_offset_limit_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _make_get_object(b"\xff\xfe\x00\x01binary")
        backend = S3Backend(
            S3BackendConfig(bucket="b", binary_read_mode="base64"),
            client=client,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_backend_aws.s3._read"):
            backend.read("/binary.bin")  # use defaults
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING" and "non-UTF-8" in rec.getMessage()
        ]
        assert not warnings, "default offset/limit must not warn"
