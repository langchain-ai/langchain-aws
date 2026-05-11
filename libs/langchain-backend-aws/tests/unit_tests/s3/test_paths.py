"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from ._helpers import _make_backend

# ------------------------------------------------------------------
# Path mapping helpers
# ------------------------------------------------------------------


class TestPathMapping:
    """Tests for path-to-key and key-to-path conversions."""

    def test_path_to_key_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._path_to_key("/foo/bar.txt") == "foo/bar.txt"

    def test_path_to_key_with_prefix(self) -> None:
        backend, _ = _make_backend(prefix="workspace/session1")
        assert backend._path_to_key("/foo/bar.txt") == "workspace/session1/foo/bar.txt"

    def test_path_to_key_root(self) -> None:
        backend, _ = _make_backend(prefix="data")
        assert backend._path_to_key("/") == "data/"

    def test_path_to_key_root_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._path_to_key("/") == ""

    def test_key_to_path_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._key_to_path("foo/bar.txt") == "/foo/bar.txt"

    def test_key_to_path_with_prefix(self) -> None:
        backend, _ = _make_backend(prefix="workspace")
        assert backend._key_to_path("workspace/foo/bar.txt") == "/foo/bar.txt"

    def test_round_trip(self) -> None:
        backend, _ = _make_backend(prefix="prefix")
        path = "/some/deep/file.py"
        assert backend._key_to_path(backend._path_to_key(path)) == path


# ------------------------------------------------------------------
# Path traversal prevention
# ------------------------------------------------------------------


class TestPathTraversal:
    """Tests for path traversal attack prevention."""

    def test_traversal_in_read(self) -> None:
        backend, _ = _make_backend()
        result = backend.read("/../../../etc/passwd")
        assert result.error is not None
        assert "traversal" in result.error.lower()

    def test_traversal_in_write(self) -> None:
        backend, _ = _make_backend()
        result = backend.write("/../secret.txt", "data")
        assert result.error is not None

    def test_traversal_in_edit(self) -> None:
        backend, _ = _make_backend()
        result = backend.edit("/../secret.txt", "a", "b")
        assert result.error is not None

    def test_traversal_in_ls(self) -> None:
        backend, _ = _make_backend()
        result = backend.ls("/../")
        assert result.error is not None

    def test_traversal_in_glob(self) -> None:
        backend, _ = _make_backend()
        result = backend.glob("*.py", path="/../")
        assert result.error is not None

    def test_traversal_in_upload(self) -> None:
        backend, _ = _make_backend()
        result = backend.upload_files([("/../evil.txt", b"data")])
        assert result[0].error == "invalid_path"

    def test_traversal_in_download(self) -> None:
        backend, _ = _make_backend()
        result = backend.download_files(["/../evil.txt"])
        assert result[0].error == "invalid_path"

    def test_tilde_path_rejected(self) -> None:
        backend, _ = _make_backend()
        result = backend.read("~/secret")
        assert result.error is not None
