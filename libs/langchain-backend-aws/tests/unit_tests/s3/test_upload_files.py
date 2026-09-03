"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend

from ._helpers import _client_error, _make_backend

# ------------------------------------------------------------------
# upload_files()
# ------------------------------------------------------------------


class TestUploadFiles:
    """Tests for the upload_files method."""

    def test_upload_success(self) -> None:
        backend, mock = _make_backend()
        result = backend.upload_files([("/a.txt", b"data"), ("/b.txt", b"more")])
        assert len(result) == 2
        assert all(r.error is None for r in result)
        assert mock.put_object.call_count == 2

    def test_upload_error(self) -> None:
        backend, mock = _make_backend()
        mock.put_object.side_effect = _client_error("AccessDenied")

        result = backend.upload_files([("/a.txt", b"data")])
        assert result[0].error == "permission_denied"
        assert result[0].path == "/a.txt"

    def test_upload_throttle_logs_real_code(self, caplog: Any) -> None:
        """Non-permission errors must surface the real code in logs.

        The BackendProtocol constrains the response error tag to a
        Literal, so the response itself stays ``permission_denied`` —
        but operators need the underlying code (e.g. ``SlowDown``) to
        triage throttling vs. auth failures.
        """
        import logging

        backend, mock = _make_backend()
        mock.put_object.side_effect = _client_error("SlowDown")

        logger_name = "langchain_backend_aws.s3.backend"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result = backend.upload_files([("/a.txt", b"data")])

        assert result[0].error == "permission_denied"
        assert any("SlowDown" in rec.getMessage() for rec in caplog.records)

    def test_upload_empty_list(self) -> None:
        backend, mock = _make_backend()
        result = backend.upload_files([])
        assert result == []
        mock.put_object.assert_not_called()

    def test_upload_oversize_rejected(self) -> None:
        """Oversized uploads are rejected without hitting put_object.

        Mirrors the size cap applied by ``read``/``edit``/
        ``download_files`` so a caller cannot bypass ``max_file_size_mb``
        by routing the body through upload. The response surfaces the
        backend-specific ``"oversize"`` tag (sanctioned by the
        ``FileUploadResponse`` docstring); the byte count goes to the
        log for triage.
        """
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", client=mock_client, max_file_size_mb=1
        )
        body = b"x" * (2 * 1024 * 1024)  # 2 MiB > 1 MiB cap

        result = backend.upload_files([("/big.bin", body)])

        assert len(result) == 1
        assert result[0].error == "oversize"
        assert result[0].path == "/big.bin"
        mock_client.put_object.assert_not_called()

    def test_upload_oversize_does_not_block_other_files(self) -> None:
        """An oversized entry must not skip subsequent entries.

        Partial-success semantics: per-file errors are isolated, so a
        single oversized file should still let the rest of the batch
        upload normally.
        """
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", client=mock_client, max_file_size_mb=1
        )
        big = b"x" * (2 * 1024 * 1024)

        result = backend.upload_files([("/big.bin", big), ("/small.txt", b"ok")])

        assert result[0].error == "oversize"
        assert result[1].error is None
        # Only the small file reached put_object.
        assert mock_client.put_object.call_count == 1
        _, kwargs = mock_client.put_object.call_args
        assert kwargs["Key"] == "small.txt"
