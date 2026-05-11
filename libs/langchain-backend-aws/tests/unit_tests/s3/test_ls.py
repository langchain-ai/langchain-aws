"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from ._helpers import _client_error, _make_backend

# ------------------------------------------------------------------
# ls()
# ------------------------------------------------------------------


class TestLs:
    """Tests for the ls method."""

    def test_ls_returns_files_and_dirs(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "docs/readme.txt",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
                "CommonPrefixes": [
                    {"Prefix": "docs/images/"},
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.ls("/docs")
        assert result.error is None
        assert result.entries is not None
        assert len(result.entries) == 2

        file_entry = next(e for e in result.entries if not e.get("is_dir"))
        assert file_entry["path"] == "/docs/readme.txt"
        assert file_entry["size"] == 100

        dir_entry = next(e for e in result.entries if e.get("is_dir"))
        assert dir_entry["path"] == "/docs/images/"

    def test_ls_empty_directory(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]
        mock.get_paginator.return_value = paginator

        result = backend.ls("/empty")
        assert result.error is None
        assert result.entries == []

    def test_ls_skips_prefix_itself(self) -> None:
        """The prefix key itself (e.g. 'docs/') should not appear in results."""
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "docs/",
                        "Size": 0,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "docs/file.txt",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ],
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.ls("/docs")
        assert result.entries is not None
        assert len(result.entries) == 1
        assert result.entries[0]["path"] == "/docs/file.txt"

    def test_ls_client_error(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.side_effect = _client_error("AccessDenied")
        mock.get_paginator.return_value = paginator

        result = backend.ls("/secret")
        assert result.error is not None
        assert "Error listing" in result.error
