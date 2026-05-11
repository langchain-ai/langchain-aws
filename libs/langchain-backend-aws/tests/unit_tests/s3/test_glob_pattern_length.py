"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from ._helpers import _make_backend

# ------------------------------------------------------------------
# Glob pattern length cap (ReDoS-style amplification guard).
# ------------------------------------------------------------------


class TestGlobMaxPatternLength:
    def test_glob_rejects_overlong_pattern(self) -> None:
        backend, mock = _make_backend()
        long_pattern = "a" * 2000
        result = backend.glob(long_pattern)
        assert result.matches is None
        assert result.error is not None
        assert "glob_max_pattern_length" in result.error
        # Cap is enforced before any S3 call.
        mock.get_paginator.assert_not_called()

    def test_grep_rejects_overlong_glob_filter(self) -> None:
        backend, mock = _make_backend()
        long_glob = "a" * 2000
        result = backend.grep("needle", glob=long_glob)
        assert result.matches is None
        assert result.error is not None
        assert "glob_max_pattern_length" in result.error
        mock.get_paginator.assert_not_called()
