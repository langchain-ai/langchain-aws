"""Unit tests for S3Backend glob path.

Covers ``glob``, the LRU cache, ``**`` translation, and the pattern
length / wildcard metachar caps (ReDoS amplification guards).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend
from langchain_backend_aws.s3._internal import _compile_glob_regex_uncached

from ._helpers import _client_error, _make_backend

# ------------------------------------------------------------------
# glob()
# ------------------------------------------------------------------


class TestGlob:
    """Tests for the glob method."""

    def test_glob_matches_pattern(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "src/main.py",
                        "Size": 200,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "src/readme.md",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py", path="/src")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/src/main.py"

    def test_glob_recursive_pattern(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "a/b/c.py",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "a/d.txt",
                        "Size": 30,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("**/*.py", path="/a")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/a/b/c.py"

    def test_glob_no_matches(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "file.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.matches is not None
        assert len(result.matches) == 0

    def test_glob_with_prefix(self) -> None:
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "workspace/src/app.py",
                        "Size": 150,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py", path="/src")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/src/app.py"

        paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="workspace/src/"
        )

    def test_glob_basename_fallback_recurses(self) -> None:
        """`*.py` (no path separators) finds matching files at any depth."""
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "deep/nested/dir/main.py",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "top.py",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.matches is not None
        paths = sorted(m["path"] for m in result.matches)
        assert paths == ["/deep/nested/dir/main.py", "/top.py"]

    def test_glob_anchored_pattern_does_not_recurse(self) -> None:
        """Patterns containing `/` are anchored and do NOT match basename-only."""
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "src/main.py",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "src/sub/main.py",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("src/*.py")
        assert result.matches is not None
        paths = [m["path"] for m in result.matches]
        # `src/*.py` must not match `src/sub/main.py` (`*` excludes `/`).
        assert paths == ["/src/main.py"]

    def test_glob_globstar_in_middle(self) -> None:
        """`a/**/c.py` matches across any number of intermediate segments."""
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "a/c.py",
                        "Size": 1,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "a/b/c.py",
                        "Size": 1,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "a/b/c/d.py",
                        "Size": 1,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("a/**/c.py")
        assert result.matches is not None
        paths = sorted(m["path"] for m in result.matches)
        assert paths == ["/a/b/c.py", "/a/c.py"]

    def test_glob_fails_closed_when_max_objects_exceeded(self) -> None:
        """Hitting `glob_max_objects` must fail closed, not return partial matches.

        The GlobResult contract has no truncation field, so silently
        returning a partial list would let callers conclude that a file
        does not exist when it merely was not scanned.
        """
        from langchain_backend_aws import S3BackendConfig

        mock_client = MagicMock()
        config = S3BackendConfig(bucket="test-bucket", glob_max_objects=2)
        backend = S3Backend(config, client=mock_client)

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": f"f{i}.py",
                        "Size": 1,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    }
                    for i in range(10)
                ]
            }
        ]
        mock_client.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.matches is None
        assert result.error is not None
        assert "glob_max_objects" in result.error

    def test_glob_within_cap_returns_matches(self) -> None:
        """When the scan stays within the cap, results are returned normally."""
        from langchain_backend_aws import S3BackendConfig

        mock_client = MagicMock()
        config = S3BackendConfig(bucket="test-bucket", glob_max_objects=10)
        backend = S3Backend(config, client=mock_client)

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": f"f{i}.py",
                        "Size": 1,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    }
                    for i in range(3)
                ]
            }
        ]
        mock_client.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3

    def test_glob_client_error(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.side_effect = _client_error("AccessDenied")
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.error is not None


# ------------------------------------------------------------------
# Public ``clear_glob_cache`` empties the per-instance LRU.
# ------------------------------------------------------------------


class TestClearGlobCache:
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


# ------------------------------------------------------------------
# ``**`` translation. ``**`` followed by ``/`` is the only form that
# crosses path separators; bare ``**`` collapses to a single ``*`` so it
# stays within one segment.
# ------------------------------------------------------------------


class TestGlobDoubleStar:
    def test_recursive_double_star_slash_matches_across_segments(self) -> None:
        compiled = _compile_glob_regex_uncached("src/**/foo.py")
        assert compiled.match("src/foo.py") is not None
        assert compiled.match("src/a/foo.py") is not None
        assert compiled.match("src/a/b/foo.py") is not None
        assert compiled.match("other/foo.py") is None

    def test_bare_double_star_does_not_cross_slash(self) -> None:
        # ``a**b`` must behave as a single-segment match — ``a/x/b``
        # used to match because ``**`` expanded to ``.*`` and crossed
        # ``/``. The fix collapses bare ``**`` to ``[^/]*`` so it
        # behaves like a single ``*``.
        compiled = _compile_glob_regex_uncached("a**b")
        assert compiled.match("ab") is not None
        assert compiled.match("aXXXb") is not None
        assert compiled.match("a/b") is None
        assert compiled.match("a/x/b") is None

    def test_bare_double_star_equivalent_to_single_star(self) -> None:
        bare = _compile_glob_regex_uncached("foo**")
        single = _compile_glob_regex_uncached("foo*")
        for sample in ("foo", "foobar", "foobarbaz"):
            assert bool(bare.match(sample)) == bool(single.match(sample))
        # Both must reject anything that crosses ``/``.
        assert bare.match("foo/bar") is None
        assert single.match("foo/bar") is None


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


# ------------------------------------------------------------------
# Glob wildcard metachar cap. A long ``****…`` pattern under the
# source-length cap still translates to a regex with many stacked
# ``[^/]*`` runs that ``regex.Pattern.match`` backtracks on
# catastrophically. The metachar cap is the orthogonal shape bound.
# ------------------------------------------------------------------


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
