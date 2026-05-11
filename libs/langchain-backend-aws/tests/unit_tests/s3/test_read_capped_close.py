"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import MagicMock

# ------------------------------------------------------------------
# read_capped_object close-path coverage: the streaming body must be
# closed for every exit path (including a malformed ContentLength
# header) so the underlying connection is returned to the pool.
# ------------------------------------------------------------------


class TestReadCappedObjectClose:
    def test_body_closed_when_content_length_invalid(self) -> None:
        from langchain_backend_aws.s3._internal import (
            OversizeError,
            read_capped_object,
        )

        body = MagicMock()
        body.read.return_value = b"x"
        client = MagicMock()
        client.get_object.return_value = {
            "Body": body,
            # Intentionally non-numeric: forces ``int(...)`` to raise
            # ValueError. ``read_capped_object`` wraps that into
            # ``OversizeError(None)`` so a hostile S3-compatible server
            # cannot smuggle an uncategorized exception past the size
            # cap. The body must still be closed before raising.
            "ContentLength": "not-a-number",
            "ETag": '"x"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        with contextlib.suppress(OversizeError):
            read_capped_object(client, "bucket", "key", max_bytes=1024)

        body.close.assert_called_once()

    def test_body_closed_when_oversize_by_header(self) -> None:
        from langchain_backend_aws.s3._internal import (
            OversizeError,
            read_capped_object,
        )

        body = MagicMock()
        body.read.return_value = b""
        client = MagicMock()
        client.get_object.return_value = {
            "Body": body,
            "ContentLength": 9999,
            "ETag": '"x"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        with contextlib.suppress(OversizeError):
            read_capped_object(client, "bucket", "key", max_bytes=10)

        body.close.assert_called_once()

    def test_body_without_close_method_is_tolerated(self) -> None:
        """Some S3-compatible stubs return a body without ``close()``;
        the helper must not crash on them.
        """
        from langchain_backend_aws.s3._internal import read_capped_object

        class _BodyWithoutClose:
            def read(self, _: int) -> bytes:
                return b"hello"

        client = MagicMock()
        client.get_object.return_value = {
            "Body": _BodyWithoutClose(),
            "ContentLength": 5,
            "ETag": '"x"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }
        result = read_capped_object(client, "bucket", "key", max_bytes=1024)
        assert result.raw_bytes == b"hello"
