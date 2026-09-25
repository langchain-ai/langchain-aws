"""Unit tests for S3Backend download_files path.

Covers ``download_files`` happy paths, concurrency behavior, robustness
against malformed S3 responses (unexpected per-path exceptions must
become per-path ``permission_denied`` rather than aborting the batch),
and teardown-signal handling for ``CancelledError``/``KeyboardInterrupt``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import CancelledError as FuturesCancelledError
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import FileDownloadResponse

from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws.s3._io import download_files

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


# ------------------------------------------------------------------
# Concurrency: ``download_concurrency`` selects sequential vs parallel.
# ------------------------------------------------------------------


def test_download_concurrency_one_runs_sequentially() -> None:
    mock_client = MagicMock()
    mock_client.get_object.return_value = _s3_object_response(b"hello")
    config = S3BackendConfig(bucket="b", prefix="p/", download_concurrency=1)
    backend = S3Backend(config, client=mock_client)

    results = backend.download_files(["/a.txt", "/b.txt", "/c.txt"])

    assert [r.error for r in results] == [None, None, None]
    assert mock_client.get_object.call_count == 3


def test_download_concurrency_runs_in_parallel() -> None:
    """Verify multiple downloads execute concurrently when allowed."""
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def slow_get(**_: object) -> dict[str, object]:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return _s3_object_response(b"x")

    mock_client = MagicMock()
    mock_client.get_object.side_effect = slow_get
    config = S3BackendConfig(bucket="b", prefix="p/", download_concurrency=4)
    backend = S3Backend(config, client=mock_client)

    paths = [f"/f{i}.txt" for i in range(8)]
    results = backend.download_files(paths)

    assert all(r.error is None for r in results)
    assert peak >= 2, f"expected parallel execution, peak in-flight={peak}"


def test_download_concurrency_empty_paths() -> None:
    backend = S3Backend.from_kwargs(bucket="b", prefix="p/", client=MagicMock())
    assert backend.download_files([]) == []


# ------------------------------------------------------------------
# Robustness: unexpected per-path exception must become per-path
# ``permission_denied`` so the batch does not abort.
# ------------------------------------------------------------------


class TestDownloadFilesRobustness:
    def test_download_unexpected_exception_becomes_permission_denied(self) -> None:
        backend, mock = _make_backend()
        # First path raises an unexpected programming error; second
        # succeeds. The batch should complete and the surviving result
        # should still be returned.
        mock.get_object.side_effect = [
            RuntimeError("malformed response from stub"),
            _s3_object_response(b"ok"),
        ]

        result = backend.download_files(["/bad.txt", "/good.txt"])

        assert len(result) == 2
        bad = next(r for r in result if r.path == "/bad.txt")
        good = next(r for r in result if r.path == "/good.txt")
        assert bad.content is None
        assert bad.error == "permission_denied"
        assert good.content == b"ok"
        assert good.error is None

    def test_download_missing_body_key_does_not_crash_batch(self) -> None:
        backend, mock = _make_backend()
        # Simulate a malformed S3 response: ``ContentLength`` present
        # but no ``Body`` key. ``read_capped_object`` classifies missing
        # required response keys as :class:`OversizeError` so the
        # caller's ClientError/Oversize disjoint hierarchy stays clean;
        # ``download_one`` therefore surfaces this as the backend-specific
        # ``"oversize"`` tag (see :data:`OVERSIZE_ERROR_TAG`). The
        # important invariant is that the batch is not torn down by a
        # bubbling :class:`KeyError`.
        mock.get_object.return_value = {"ContentLength": 5}

        result = backend.download_files(["/broken.txt"])

        assert len(result) == 1
        assert result[0].error == "oversize"
        assert result[0].content is None

    def test_download_unexpected_exception_in_sequential_path(self) -> None:
        # Force the sequential executor path (``download_concurrency=1``).
        # The same ``_make_safe_download`` wrapper applies on this branch,
        # but it is exercised inline rather than via ``ThreadPoolExecutor``;
        # a regression that only catches in the parallel path would still
        # tear down the sequential batch on an unexpected exception.
        mock_client = MagicMock()
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="p", download_concurrency=1),
            client=mock_client,
        )
        mock_client.get_object.side_effect = [
            RuntimeError("malformed response"),
            _s3_object_response(b"ok"),
        ]

        result = backend.download_files(["/bad.txt", "/good.txt"])

        assert len(result) == 2
        # Sequential path preserves input order.
        assert result[0].path == "/bad.txt"
        assert result[0].error == "permission_denied"
        assert result[0].content is None
        assert result[1].path == "/good.txt"
        assert result[1].error is None

    def test_download_unexpected_exception_in_parallel_path(self) -> None:
        # Force the parallel executor path by setting a high
        # concurrency and providing multiple paths.
        mock_client = MagicMock()
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="p", download_concurrency=4),
            client=mock_client,
        )
        mock_client.get_object.side_effect = [
            RuntimeError("boom"),
            RuntimeError("boom"),
            _s3_object_response(b"ok"),
        ]

        result = backend.download_files(["/a.txt", "/b.txt", "/c.txt"])

        assert len(result) == 3
        errors = [r.error for r in result]
        assert errors.count("permission_denied") == 2
        assert errors.count(None) == 1


# ------------------------------------------------------------------
# Teardown signals: ``_make_safe_download`` flattens ``Exception`` and
# below into per-path ``permission_denied``; ``BaseException`` subclasses
# (``asyncio.CancelledError``, ``KeyboardInterrupt``, ``SystemExit``)
# bypass that catch and are surfaced by the outer ``as_completed`` loop.
# The result-collection loop wraps ``future.result()`` so a slot is
# never left ``None``.
# ------------------------------------------------------------------


class TestDownloadBaseException:
    def test_futures_cancelled_error_becomes_permission_denied_slot(self) -> None:
        # ``concurrent.futures.CancelledError`` is an :class:`Exception`
        # subclass on Python 3.11+. It is flattened by
        # ``_make_safe_download`` into a per-path ``permission_denied``
        # response, so the batch continues and the result slot is filled.
        calls: list[str] = []

        def raising(path: str) -> FileDownloadResponse:
            calls.append(path)
            if path == "/bad.txt":
                raise FuturesCancelledError("cancelled")
            return FileDownloadResponse(path=path, content=b"ok", error=None)

        result = download_files(
            ["/bad.txt", "/good.txt"],
            download_one=raising,
            download_concurrency=4,
            max_pool_connections=4,
        )

        assert len(result) == 2
        bad = next(r for r in result if r.path == "/bad.txt")
        good = next(r for r in result if r.path == "/good.txt")
        assert bad.error == "permission_denied"
        assert bad.content is None
        assert good.error is None
        assert good.content == b"ok"

    def test_no_none_slots_when_all_paths_raise_futures_cancelled(self) -> None:
        def raising(path: str) -> FileDownloadResponse:
            raise FuturesCancelledError("cancelled")

        result = download_files(
            ["/a.txt", "/b.txt", "/c.txt"],
            download_one=raising,
            download_concurrency=4,
            max_pool_connections=4,
        )

        assert len(result) == 3
        # The post-loop assert would raise ``RuntimeError`` if any slot
        # were left ``None`` — every entry must be a populated response.
        for r in result:
            assert r is not None
            assert r.error == "permission_denied"
            assert r.content is None

    def test_asyncio_cancelled_error_is_reraised_after_drain(self) -> None:
        # ``asyncio.CancelledError`` inherits from :class:`BaseException`
        # on Python 3.11+ so ``_make_safe_download`` does not flatten it.
        # The outer collector names it in the teardown ``except`` clause
        # and re-raises it once the executor has drained.
        import asyncio

        def raising(path: str) -> FileDownloadResponse:
            if path == "/bad.txt":
                raise asyncio.CancelledError
            return FileDownloadResponse(path=path, content=b"ok", error=None)

        with pytest.raises(asyncio.CancelledError):
            download_files(
                ["/bad.txt", "/good.txt"],
                download_one=raising,
                download_concurrency=4,
                max_pool_connections=4,
            )

    def test_keyboard_interrupt_is_reraised_after_drain(self) -> None:
        # ``KeyboardInterrupt`` is a user-driven teardown signal, not a
        # per-path error: ``download_files`` must re-raise it once the
        # executor has drained instead of folding it into
        # ``permission_denied`` (which would silently continue).
        def raising(path: str) -> FileDownloadResponse:
            if path == "/bad.txt":
                raise KeyboardInterrupt
            return FileDownloadResponse(path=path, content=b"ok", error=None)

        with pytest.raises(KeyboardInterrupt):
            download_files(
                ["/bad.txt", "/good.txt"],
                download_one=raising,
                download_concurrency=4,
                max_pool_connections=4,
            )


# ------------------------------------------------------------------
# Teardown logging on multiple in-flight interrupts. Python's implicit
# ``__context__`` chaining cannot link sibling-future exceptions, so
# subsequent teardown signals are otherwise silently dropped. They must
# surface as a WARNING for operator visibility.
# ------------------------------------------------------------------


class TestDownloadFilesTeardownLogging:
    def test_second_teardown_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Two paths, both raising ``KeyboardInterrupt``: only the first is
        # re-raised, but the second must surface as a WARNING so the loss
        # is not silent.
        from langchain_backend_aws.s3 import _io

        def fail(path: str) -> Any:
            raise KeyboardInterrupt(f"interrupt on {path}")

        with (
            caplog.at_level(logging.WARNING, logger=_io.__name__),
            pytest.raises(KeyboardInterrupt),
        ):
            download_files(
                ["a", "b", "c"],
                download_one=fail,
                download_concurrency=4,
                max_pool_connections=10,
            )
        # At least one (and likely two) extra teardown signals were
        # discarded; the WARNING must surface so the loss is visible.
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING"
            and "additional teardown signal" in rec.getMessage()
        ]
        assert warnings, "expected at least one extra-teardown warning"
