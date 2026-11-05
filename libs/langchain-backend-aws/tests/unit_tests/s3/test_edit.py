"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from ._helpers import _client_error, _make_backend, _s3_object_response

# ------------------------------------------------------------------
# edit()
# ------------------------------------------------------------------


class TestEdit:
    """Tests for the edit method."""

    def test_edit_single_replacement(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(
            b"hello world", etag='"abc123"'
        )

        result = backend.edit("/file.txt", "world", "earth")
        assert result.error is None
        assert result.path == "/file.txt"
        assert result.occurrences == 1

        mock.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="file.txt",
            Body=b"hello earth",
            IfMatch='"abc123"',
        )

    def test_edit_conflict_on_precondition_failed(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"hello world", etag='"v1"')
        mock.put_object.side_effect = _client_error("PreconditionFailed")

        result = backend.edit("/file.txt", "world", "earth")
        assert result.error is not None
        assert "Conflict" in result.error
        assert "modified concurrently" in result.error

    def test_edit_replace_all(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"aaa")

        result = backend.edit("/file.txt", "a", "b", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3

    def test_edit_replace_all_positional(self) -> None:
        # BackendProtocol declares ``replace_all`` positionally; pin the
        # contract so a future keyword-only refactor breaks loudly.
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"aaa")

        result = backend.edit("/file.txt", "a", "b", True)  # noqa: FBT003
        assert result.error is None
        assert result.occurrences == 3

    def test_edit_string_not_found(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"hello")

        result = backend.edit("/file.txt", "missing", "replacement")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_multiple_without_replace_all(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"foo foo foo")

        result = backend.edit("/file.txt", "foo", "bar")
        assert result.error is not None
        assert "appears 3 times" in result.error

    def test_edit_file_not_found(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("NoSuchKey")

        result = backend.edit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_binary_file_errors(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"\x80\x81\x82")

        result = backend.edit("/binary.bin", "a", "b")
        assert result.error is not None
        assert "not a text file" in result.error.lower()

    def test_edit_fails_closed_when_etag_missing(self) -> None:
        """edit must not silently drop optimistic concurrency when ETag is absent."""
        backend, mock = _make_backend()
        # Build a response without an ETag — drop the field rather than
        # set it to None so we exercise the ``response.get("ETag")`` path.
        response = _s3_object_response(b"hello")
        del response["ETag"]
        mock.get_object.return_value = response

        result = backend.edit("/file.txt", "hello", "world")
        assert result.error is not None
        assert "ETag" in result.error
        # No PUT should have been issued.
        mock.put_object.assert_not_called()
