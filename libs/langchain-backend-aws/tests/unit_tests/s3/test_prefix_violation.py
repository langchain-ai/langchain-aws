"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from ._helpers import _make_backend

# ------------------------------------------------------------------
# Storage prefix violation: ls/grep/glob fail closed when the
# paginator returns keys outside the configured prefix.
# ------------------------------------------------------------------


class TestPrefixViolationFailsClosed:
    """A misbehaving S3-compatible store / proxy can return keys
    outside the requested ``Prefix``. The backend must discard partial
    results and surface an error rather than silently dropping the
    offending entries — otherwise a partial scan looks identical to a
    complete negative result.
    """

    def test_ls_fails_closed_on_out_of_prefix_key(self) -> None:
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "workspace/legit.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "OTHER_TENANT/leak.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.ls("/")
        assert result.entries is None
        assert result.error is not None
        assert "prefix violation" in result.error

    def test_ls_fails_closed_on_out_of_prefix_common_prefix(self) -> None:
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [],
                "CommonPrefixes": [
                    {"Prefix": "OTHER_TENANT/dir/"},
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.ls("/")
        assert result.entries is None
        assert result.error is not None
        assert "prefix violation" in result.error

    def test_glob_fails_closed_on_out_of_prefix_key(self) -> None:
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "workspace/a.py",
                        "Size": 5,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "OTHER_TENANT/b.py",
                        "Size": 5,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("**/*.py")
        assert result.matches is None
        assert result.error is not None
        assert "prefix violation" in result.error

    def test_glob_fails_closed_on_out_of_prefix_key_with_pattern_miss(
        self,
    ) -> None:
        """A prefix-violating key must fail closed even when the user's
        glob pattern would not match it.

        Regression guard against ordering bug: previously the glob
        filter ran before the prefix-containment check, so a leaked
        key that happened not to match the user's pattern was silently
        skipped instead of producing an error. The trust-boundary
        invariant is "ListObjectsV2 returning a key outside the
        configured prefix is always an error", regardless of whether
        that key matches what the caller is looking for. Per the
        deepagents virtual-FS guideline, surface this as
        ``GlobResult.error`` rather than raising.
        """
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        # Outside the configured prefix, AND does not
                        # match ``*.py`` — the old order would have
                        # filtered it out before the prefix check.
                        "Key": "OTHER_TENANT/leak.txt",
                        "Size": 5,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.glob("*.py")
        assert result.matches is None
        assert result.error is not None
        assert "prefix violation" in result.error

    def test_grep_fails_closed_on_out_of_prefix_key(self) -> None:
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "OTHER_TENANT/leak.txt",
                        "Size": 5,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.grep("needle")
        assert result.matches is None
        assert result.error is not None
        assert "prefix violation" in result.error
        # No GetObject should be issued for a key we refused.
        mock.get_object.assert_not_called()

    def test_grep_fails_closed_on_out_of_prefix_key_with_glob(self) -> None:
        """A prefix-violating key must fail closed even when the
        provided glob would otherwise filter it out.

        Glob/size filtering must not be allowed to mask a paginator
        leak: the prefix-containment contract is checked first.
        """
        backend, mock = _make_backend(prefix="workspace")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "OTHER_TENANT/leak.txt",
                        "Size": 5,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        # ``glob='*.py'`` would not match ``leak.txt``; the old code
        # returned an empty match list. The new code must fail closed.
        result = backend.grep("needle", glob="*.py")
        assert result.matches is None
        assert result.error is not None
        assert "prefix violation" in result.error
        mock.get_object.assert_not_called()
