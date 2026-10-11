"""Live-store checks for ``binary_read_mode``."""

from __future__ import annotations

from .conftest import make_backend, skip_without_credentials

pytestmark = skip_without_credentials


def test_binary_read_mode_error_against_live_store(prefix: str) -> None:
    backend = make_backend(prefix, binary_read_mode="error")
    upload = backend.upload_files([("/blob.bin", b"\xff\xfe\x00binary")])
    assert upload[0].error is None

    result = backend.read("/blob.bin")

    assert result.error is not None
    assert "not UTF-8" in result.error
    assert "download_files" in result.error


def test_binary_read_mode_base64_default_against_live_store(prefix: str) -> None:
    backend = make_backend(prefix)
    backend.upload_files([("/blob.bin", b"\xff\xfe\x00binary")])

    result = backend.read("/blob.bin")

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["encoding"] == "base64"
