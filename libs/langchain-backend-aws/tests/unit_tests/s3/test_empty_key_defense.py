"""Unit tests for the empty-object-key defense.

With ``prefix=""`` and a path that normalizes to ``/`` the underlying
S3 object key would be the empty string. boto3 surfaces this as a
generic ``ParamValidationError``; reject it close to the public API
instead so callers see a clear "path is not a file" error and the
backend never issues a degenerate ``GetObject(Key="")`` request.
"""

from __future__ import annotations

from ._helpers import _make_backend


class TestEmptyKeyDefense:
    def test_read_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.read("/")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.get_object.assert_not_called()

    def test_write_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.write("/", "hi")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.put_object.assert_not_called()

    def test_edit_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.edit("/", "a", "b")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.get_object.assert_not_called()

    def test_upload_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        responses = backend.upload_files([("/", b"hi")])
        assert responses[0].error == "invalid_path"
        mock.put_object.assert_not_called()

    def test_download_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        responses = backend.download_files(["/"])
        assert responses[0].error == "invalid_path"
        mock.get_object.assert_not_called()

    def test_ls_root_path_still_works(self) -> None:
        # ls/glob/grep operate on directories; an empty key for the
        # bucket root is intentional under no-prefix mode.
        backend, mock = _make_backend(prefix="")
        mock.get_paginator.return_value.paginate.return_value = []
        result = backend.ls("/")
        assert result.error is None
