"""Tests for parallel ``download_files`` (``download_concurrency``)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig

from ._helpers import _s3_object_response


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
