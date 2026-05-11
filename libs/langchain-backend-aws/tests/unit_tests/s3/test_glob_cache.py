"""Public ``S3Backend.clear_glob_cache`` empties the per-instance LRU."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend


class TestClearGlobCache:
    """Public ``clear_glob_cache`` drops every entry from the LRU."""

    def test_clear_glob_cache_empties_cache(self) -> None:
        backend = S3Backend.from_kwargs(bucket="b", client=MagicMock())
        backend._compile_glob("*.py")
        backend._compile_glob("*.md")
        info_before = backend._compile_glob.cache_info()  # type: ignore[attr-defined]
        assert info_before.currsize >= 2

        backend.clear_glob_cache()

        info_after = backend._compile_glob.cache_info()  # type: ignore[attr-defined]
        assert info_after.currsize == 0

    def test_clear_glob_cache_idempotent(self) -> None:
        backend = S3Backend.from_kwargs(bucket="b", client=MagicMock())
        backend.clear_glob_cache()
        backend.clear_glob_cache()  # second call must not raise
