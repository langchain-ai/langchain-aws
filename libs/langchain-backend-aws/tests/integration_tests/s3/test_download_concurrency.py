"""Live-store checks for parallel ``download_files``."""

from __future__ import annotations

from .conftest import make_backend, skip_without_credentials

pytestmark = skip_without_credentials


def test_download_files_parallel_against_live_store(prefix: str) -> None:
    backend = make_backend(prefix, download_concurrency=4)
    paths = [f"/f{i}.txt" for i in range(8)]
    uploads = backend.upload_files([(p, f"content-{p}".encode()) for p in paths])
    assert all(u.error is None for u in uploads)

    results = backend.download_files(paths)

    assert len(results) == len(paths)
    assert all(r.error is None for r in results)
    by_path = {r.path: r.content for r in results}
    for p in paths:
        assert by_path[p] == f"content-{p}".encode()


def test_download_files_sequential_when_concurrency_one(prefix: str) -> None:
    backend = make_backend(prefix, download_concurrency=1)
    paths = [f"/seq{i}.txt" for i in range(3)]
    backend.upload_files([(p, b"seq") for p in paths])

    results = backend.download_files(paths)

    assert [r.error for r in results] == [None, None, None]
    assert {r.path for r in results} == set(paths)


def test_download_files_empty_list_live(prefix: str) -> None:
    backend = make_backend(prefix)
    assert backend.download_files([]) == []
