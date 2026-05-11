"""Robustness tests for ``download_files`` exception handling.

The download executor must convert any unexpected per-path exception
into a per-path ``permission_denied`` response rather than aborting the
whole batch — otherwise a single malformed S3 response (missing
``Body``, programmatic stub error) tears down concurrent fetches and
the BackendProtocol contract leaks an unhandled exception.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ._helpers import _make_backend, _s3_object_response


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
        from langchain_backend_aws import S3Backend, S3BackendConfig

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
        from langchain_backend_aws import S3Backend, S3BackendConfig

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
