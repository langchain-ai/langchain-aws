"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from ._helpers import _client_error, _make_backend, _s3_object_response

# ------------------------------------------------------------------
# download_files()
# ------------------------------------------------------------------


class TestDownloadFiles:
    """Tests for the download_files method."""

    def test_download_success(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"file content")

        result = backend.download_files(["/file.txt"])
        assert len(result) == 1
        assert result[0].content == b"file content"
        assert result[0].error is None

    def test_download_not_found(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("NoSuchKey")

        result = backend.download_files(["/missing.txt"])
        assert result[0].content is None
        assert result[0].error == "file_not_found"

    def test_download_permission_denied(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("AccessDenied")

        result = backend.download_files(["/secret.txt"])
        assert result[0].content is None
        assert result[0].error == "permission_denied"

    def test_download_throttle_logs_real_code(self, caplog: Any) -> None:
        import logging

        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("SlowDown")

        logger_name = "langchain_backend_aws.s3.backend"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result = backend.download_files(["/file.txt"])

        assert result[0].content is None
        assert result[0].error == "permission_denied"
        assert any("SlowDown" in rec.getMessage() for rec in caplog.records)

    def test_download_refuses_oversize_object(self, caplog: Any) -> None:
        """Oversized objects must not be loaded into memory by download_files."""
        import logging

        backend, mock = _make_backend()
        max_bytes = backend._config.max_file_size_mb * 1024 * 1024
        # Header reports an oversize body so the cap is hit pre-read.
        stream = MagicMock()
        stream.read.return_value = b"unused"
        mock.get_object.return_value = {
            "Body": stream,
            "ContentLength": max_bytes + 1,
            "ETag": '"etag"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        with caplog.at_level(logging.ERROR, logger="langchain_backend_aws.s3.backend"):
            result = backend.download_files(["/big.bin"])

        assert result[0].content is None
        assert result[0].error == "oversize"
        # Body should never have been read into memory.
        stream.read.assert_not_called()
        # Operator-facing log must explain the real reason.
        assert any("max_file_size_mb" in rec.getMessage() for rec in caplog.records)

    def test_download_refuses_oversize_when_header_lies(self) -> None:
        """A liar ``ContentLength`` cannot bypass the in-memory cap."""
        backend, mock = _make_backend()
        max_bytes = backend._config.max_file_size_mb * 1024 * 1024
        body = b"x" * (max_bytes + 1024)
        stream = MagicMock()
        stream.read.return_value = body
        mock.get_object.return_value = {
            "Body": stream,
            "ContentLength": 100,  # Lies — actual body is far larger.
            "ETag": '"etag"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        result = backend.download_files(["/big.bin"])
        assert result[0].content is None
        assert result[0].error == "oversize"

    def test_download_throttle_logs_at_error_level(self, caplog: Any) -> None:
        """Transient S3 codes log at ERROR so alerting catches them."""
        import logging

        backend, mock = _make_backend()
        mock.get_object.side_effect = _client_error("SlowDown")

        with caplog.at_level(
            logging.DEBUG, logger="langchain_backend_aws.s3._internal"
        ):
            backend.download_files(["/file.txt"])

        transient_records = [
            rec for rec in caplog.records if "SlowDown" in rec.getMessage()
        ]
        assert transient_records, "expected at least one log record mentioning SlowDown"
        assert any(rec.levelno == logging.ERROR for rec in transient_records), (
            "transient codes must log at ERROR, got "
            f"{[rec.levelname for rec in transient_records]}"
        )

    def test_download_multiple(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.side_effect = [
            _s3_object_response(b"first"),
            _client_error("NoSuchKey"),
        ]

        result = backend.download_files(["/a.txt", "/b.txt"])
        assert result[0].content == b"first"
        assert result[0].error is None
        assert result[1].content is None
        assert result[1].error == "file_not_found"
