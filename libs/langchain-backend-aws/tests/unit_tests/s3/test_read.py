"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

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
