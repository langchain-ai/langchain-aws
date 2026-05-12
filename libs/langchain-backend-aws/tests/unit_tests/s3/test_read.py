"""Unit tests for S3Backend read path (split from monolithic test_backend.py).

Covers ``read``, ``binary_read_mode``, the underlying ``read_capped_object``
helper (body close-paths and non-bytes body defense), and the
offset/limit warning on the base64 fall-back.
"""

from __future__ import annotations

import contextlib
import io
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws.s3._internal import OversizeError, read_capped_object

from ._helpers import _client_error, _make_backend, _s3_object_response

# ------------------------------------------------------------------
# read()
# ------------------------------------------------------------------


class TestRead:
    """Tests for the read method."""

    def test_read_text_file(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"line1\nline2\nline3")

        result = backend.read("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "line1\nline2\nline3"
        assert result.file_data["encoding"] == "utf-8"

    def test_read_with_offset_and_limit(self) -> None:
        backend, mock = _make_backend()
        content = "\n".join(f"line{i}" for i in range(10))
        mock.get_object.return_value = _s3_object_response(content.encode())

        result = backend.read("/file.txt", offset=2, limit=3)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "line2\nline3\nline4"

    def test_read_offset_exceeds_length(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"short")

        result = backend.read("/file.txt", offset=100)
        assert result.error is not None
        assert "exceeds file length" in result.error

    def test_read_offset_equal_to_length_returns_empty(self) -> None:
        # ``offset == len(file)`` means "skip exactly the whole file";
        # the requested skip is satisfied so the result is an empty
        # selection rather than an error. ``offset > len(file)`` still
        # errors (covered by ``test_read_offset_exceeds_length``).
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"a\nb\nc")

        result = backend.read("/file.txt", offset=3)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    def test_read_empty_file(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"")

        result = backend.read("/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    def test_read_empty_file_with_offset(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"")

        result = backend.read("/empty.txt", offset=5)
        assert result.error is not None
        assert "exceeds file length" in result.error

    def test_read_negative_offset_rejected(self) -> None:
        """Negative ``offset`` returns an error rather than slicing.

        Regression guard: Python's slice semantics interpret
        ``lines[-5:]`` as "last five lines". Without an explicit
        non-negative check, ``read(offset=-5, limit=2000)`` would
        silently return the file's tail instead of failing — exactly
        the kind of silent-wrong-data behavior the ``limit <= 0`` guard
        was added to prevent on the other axis.
        """
        backend, mock = _make_backend()
        content = "\n".join(f"line{i}" for i in range(10))
        mock.get_object.return_value = _s3_object_response(content.encode())

        result = backend.read("/file.txt", offset=-5)
        assert result.error is not None
        assert "non-negative" in result.error
        assert result.file_data is None

    def test_read_generic_client_error(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("InternalError")

        result = backend.read("/file.txt")
        assert result.error is not None
        assert "Error reading" in result.error

    def test_read_binary_file(self) -> None:
        backend, mock = _make_backend()
        binary_data = b"\x89PNG\r\n\x1a\n\x00"
        mock.get_object.return_value = _s3_object_response(binary_data)

        result = backend.read("/image.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"

    def test_read_not_found(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("NoSuchKey")

        result = backend.read("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error

    def test_read_uses_correct_key_with_prefix(self) -> None:
        backend, mock = _make_backend(prefix="data")
        mock.get_object.return_value = _s3_object_response(b"content")

        backend.read("/file.txt")
        mock.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/file.txt"
        )

    def test_read_timestamps(self) -> None:
        backend, mock = _make_backend()
        ts = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        mock.get_object.return_value = _s3_object_response(b"content", last_modified=ts)

        result = backend.read("/file.txt")
        assert result.file_data is not None
        assert result.file_data["created_at"] == ts.isoformat()

    def test_read_trailing_newline_line_count(self) -> None:
        """Trailing-newline files report a line count that matches the
        text the user sees, not :class:`io.StringIO`'s split.

        ``"a\\nb\\n"`` is conceptually two lines; ``StringIO`` iteration
        yields ``["a\\n", "b\\n"]`` (two lines), which is what we want.
        Asking for ``offset=2`` is "skip both lines" and must return an
        empty selection rather than erroring as "out of range".
        """
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"a\nb\n")

        # offset=2 == seen → empty selection.
        result = backend.read("/file.txt", offset=2)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

        # offset=3 > seen → out-of-range error.
        result_over = backend.read("/file.txt", offset=3)
        assert result_over.error is not None
        assert "exceeds file length" in result_over.error

    def test_read_no_trailing_newline_line_count(self) -> None:
        """Without a trailing newline the count still matches user view."""
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"a\nb")

        result = backend.read("/file.txt", offset=2)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

        result_over = backend.read("/file.txt", offset=3)
        assert result_over.error is not None
        assert "exceeds file length" in result_over.error

    def test_read_oversize_body_with_understated_content_length(self) -> None:
        """A liar ``ContentLength`` cannot bypass the in-memory cap."""
        backend, mock = _make_backend()
        max_bytes = backend._config.max_file_size_mb * 1024 * 1024
        # Header claims small, body delivers more than the cap.
        body = b"x" * (max_bytes + 1024)
        stream = MagicMock()
        stream.read.return_value = body
        mock.get_object.return_value = {
            "Body": stream,
            "ContentLength": 100,  # Lies — actual body is far larger.
            "ETag": '"etag"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        result = backend.read("/big.txt")
        assert result.error is not None
        assert "max_file_size_mb" in result.error


# ------------------------------------------------------------------
# binary_read_mode (base64 vs error)
# ------------------------------------------------------------------


class TestBinaryReadMode:
    """``S3BackendConfig.binary_read_mode`` selects base64 vs explicit error."""

    def test_binary_read_mode_error_returns_error(self) -> None:
        mock_client = MagicMock()
        mock_client.get_object.return_value = _s3_object_response(
            b"\xff\xfe\x00binary"
        )
        config = S3BackendConfig(bucket="b", prefix="p/", binary_read_mode="error")
        backend = S3Backend(config, client=mock_client)

        result = backend.read("/bin.dat")

        assert result.error is not None
        assert "not UTF-8" in result.error
        assert "download_files" in result.error
        assert result.file_data is None

    def test_binary_read_mode_base64_default(self) -> None:
        mock_client = MagicMock()
        mock_client.get_object.return_value = _s3_object_response(
            b"\xff\xfe\x00binary"
        )
        backend = S3Backend.from_kwargs(bucket="b", prefix="p/", client=mock_client)

        result = backend.read("/bin.dat")

        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"


# ------------------------------------------------------------------
# read_capped_object: non-bytes body defense
# ------------------------------------------------------------------


def _response_with_str_body(body: str) -> dict[str, Any]:
    """Build a get_object response whose ``Body.read()`` returns a str."""
    stream = MagicMock()
    stream.read.return_value = body
    return {
        "Body": stream,
        "ContentLength": len(body),
        "ETag": '"x"',
        "LastModified": datetime(2025, 3, 7, tzinfo=UTC),
    }


class TestNonBytesBodyDefense:
    """``read_capped_object`` must surface a ``TypeError`` for non-bytes.

    Defense-in-depth: a custom boto stub that returns a ``str`` body would
    silently corrupt the downstream encoding path without this check.
    """

    def test_str_body_raises_type_error(self) -> None:
        client = MagicMock()
        client.get_object.return_value = _response_with_str_body("not-bytes")

        with pytest.raises(TypeError, match="bytes-like"):
            read_capped_object(client, "b", "k", max_bytes=1024)


# ------------------------------------------------------------------
# read_capped_object close-path coverage: the streaming body must be
# closed for every exit path (including a malformed ContentLength
# header) so the underlying connection is returned to the pool.
# ------------------------------------------------------------------


class TestReadCappedObjectClose:
    def test_body_closed_when_content_length_invalid(self) -> None:
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


# ------------------------------------------------------------------
# offset/limit warning on the base64 fall-back path: operators must see
# a WARNING so the silent ignoring of the requested slice does not look
# like a successful slice.
# ------------------------------------------------------------------


def _make_get_object(body: bytes) -> Any:
    client = MagicMock()
    client.get_object.return_value = {
        "Body": io.BytesIO(body),
        "ContentLength": len(body),
        "LastModified": MagicMock(),
        "ETag": '"deadbeef"',
    }
    client.get_object.return_value["LastModified"].astimezone.return_value = (
        MagicMock()
    )
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
