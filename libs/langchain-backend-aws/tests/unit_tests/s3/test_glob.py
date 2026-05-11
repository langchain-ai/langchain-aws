"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend

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
