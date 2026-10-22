"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from ._helpers import _client_error, _make_backend

# ------------------------------------------------------------------
# write()
# ------------------------------------------------------------------


class TestWrite:
    """Tests for the write method."""

    def test_write_new_file(self) -> None:
        backend, mock = _make_backend()

        result = backend.write("/new.txt", "hello world")
        assert result.error is None
        assert result.path == "/new.txt"
        mock.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="new.txt",
            Body=b"hello world",
            IfNoneMatch="*",
        )

    def test_write_existing_file_errors(self) -> None:
        backend, mock = _make_backend()
        mock.put_object.side_effect = _client_error("PreconditionFailed")

        result = backend.write("/exists.txt", "content")
        assert result.error is not None
        assert "already exists" in result.error

    def test_write_existing_file_errors_412(self) -> None:
        # Some S3-compatible stores return raw HTTP status as code.
        backend, mock = _make_backend()
        mock.put_object.side_effect = _client_error("412")

        result = backend.write("/exists.txt", "content")
        assert result.error is not None
        assert "already exists" in result.error

    def test_write_with_prefix(self) -> None:
        backend, mock = _make_backend(prefix="workspace")

        backend.write("/file.py", "print('hi')")
        mock.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="workspace/file.py",
            Body=b"print('hi')",
            IfNoneMatch="*",
        )

    def test_write_put_error(self) -> None:
        backend, mock = _make_backend()
        mock.put_object.side_effect = _client_error("AccessDenied")

        result = backend.write("/file.txt", "content")
        assert result.error is not None
        assert "Error writing" in result.error
