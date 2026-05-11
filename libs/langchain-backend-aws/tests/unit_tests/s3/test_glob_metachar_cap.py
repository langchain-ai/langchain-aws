"""Unit tests for the glob wildcard count cap (ReDoS amplification guard).

Mirrors ``test_grep_metachar_cap.py`` but for the glob/grep-glob path.
A long ``****…`` under the source-length cap still translates to a
regex with many stacked ``[^/]*`` runs, which ``regex.Pattern.match``
backtracks on catastrophically. The cap is the orthogonal shape bound.
"""

from __future__ import annotations

from ._helpers import _make_backend


class TestGlobMaxPatternMetachars:
    def test_glob_rejects_overstuffed_wildcards(self) -> None:
        backend, mock = _make_backend()
        # 100 stars: under the 1000-char source-length cap but well over
        # the default 50-wildcard metachar cap.
        pattern = "*" * 100
        result = backend.glob(pattern)
        assert result.matches is None
        assert result.error is not None
        assert "glob_max_pattern_metachars" in result.error
        mock.get_paginator.assert_not_called()

    def test_grep_rejects_overstuffed_glob_filter(self) -> None:
        backend, mock = _make_backend()
        glob_filter = "*" * 100
        result = backend.grep("needle", glob=glob_filter)
        assert result.matches is None
        assert result.error is not None
        assert "glob_max_pattern_metachars" in result.error
        mock.get_paginator.assert_not_called()

    def test_glob_question_marks_count_too(self) -> None:
        backend, _ = _make_backend()
        pattern = "?" * 100
        result = backend.glob(pattern)
        assert result.error is not None
        assert "glob_max_pattern_metachars" in result.error

    def test_glob_under_metachar_cap_still_runs(self) -> None:
        backend, mock = _make_backend()
        mock.get_paginator.return_value.paginate.return_value = []
        # 10 wildcards is well under the 50 default cap.
        result = backend.glob("*.py")
        assert result.error is None
