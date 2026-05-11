"""Regression tests for ``download_files`` BaseException handling.

``_make_safe_download`` flattens ``Exception`` and below into a per-path
``permission_denied`` response, but ``BaseException`` (CancelledError,
KeyboardInterrupt, SystemExit) bypasses that catch. The parallel-path
result-collection loop wraps ``future.result()`` so a slot is never
left ``None`` — otherwise the final ``cast`` would lie about the
returned shape and a caller typed against ``FileDownloadResponse``
would receive ``None``.
"""

from __future__ import annotations

from concurrent.futures import CancelledError

import pytest
from deepagents.backends.protocol import FileDownloadResponse

from langchain_backend_aws.s3._io import download_files


class TestDownloadBaseException:
    def test_base_exception_becomes_permission_denied_slot(self) -> None:
        # Callable that raises CancelledError (a ``BaseException`` in
        # Python 3.8+). ``_make_safe_download`` will not catch it; the
        # parallel collector must.
        calls: list[str] = []

        def raising(path: str) -> FileDownloadResponse:
            calls.append(path)
            if path == "/bad.txt":
                raise CancelledError("cancelled")
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

    def test_no_none_slots_when_all_paths_raise_base_exception(self) -> None:
        def raising(path: str) -> FileDownloadResponse:
            raise CancelledError("cancelled")

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
