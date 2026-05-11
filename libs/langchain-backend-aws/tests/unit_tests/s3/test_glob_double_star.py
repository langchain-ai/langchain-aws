"""Regression tests for the glob ``**`` translation.

``**`` followed by ``/`` is the only form that crosses path separators
(matches "zero or more path segments"). Bare ``**`` collapses to a
single ``*`` so it stays inside one segment — earlier revisions
translated it to ``.*`` which would silently cross ``/`` and conflict
with shell glob semantics.
"""

from __future__ import annotations

from langchain_backend_aws.s3._internal import _compile_glob_regex_uncached


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
