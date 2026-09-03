"""Unit tests for S3Backend ls path.

Covers ``ls`` happy paths and robustness against malformed S3 listings
(missing ``Key`` / ``LastModified`` / ``Prefix``) — these must fail
closed with a sanitized error rather than escape across the
``BackendProtocol`` boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig

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


# ------------------------------------------------------------------
# Robustness against malformed listing responses.
# ------------------------------------------------------------------


def _backend_with_pages(pages: list[dict[str, Any]]) -> S3Backend:
    mock_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = iter(pages)
    mock_client.get_paginator.return_value = paginator
    return S3Backend(
        S3BackendConfig(bucket="b", prefix="tenant/a"),
        client=mock_client,
    )


class TestLsMalformedResponse:
    def test_ls_missing_key_fails_closed(self) -> None:
        # ``Contents`` entry without ``Key`` — would raise ``KeyError``
        # inside the loop. Must surface as an ``LsResult.error`` rather
        # than escape across the protocol boundary.
        backend = _backend_with_pages([{"Contents": [{"Size": 1}]}])
        result = backend.ls("/")
        assert result.error is not None
        assert "malformed listing response" in result.error.lower()

    def test_ls_missing_last_modified_fails_closed(self) -> None:
        # ``LastModified`` missing — accessed via ``obj["LastModified"]``.
        backend = _backend_with_pages(
            [{"Contents": [{"Key": "tenant/a/notes.txt", "Size": 1}]}]
        )
        result = backend.ls("/")
        assert result.error is not None
        assert "malformed listing response" in result.error.lower()

    def test_ls_missing_common_prefix_value_fails_closed(self) -> None:
        # ``CommonPrefixes`` entry without ``Prefix`` — would raise
        # ``KeyError`` when read as ``prefix_entry["Prefix"]``.
        backend = _backend_with_pages([{"CommonPrefixes": [{}]}])
        result = backend.ls("/")
        assert result.error is not None
        assert "malformed listing response" in result.error.lower()


class TestGlobMalformedResponse:
    def test_glob_missing_key_fails_closed(self) -> None:
        backend = _backend_with_pages([{"Contents": [{"Size": 1}]}])
        result = backend.glob("*.txt")
        assert result.error is not None
        assert "malformed listing response" in result.error.lower()

    def test_glob_missing_last_modified_fails_closed(self) -> None:
        backend = _backend_with_pages(
            [{"Contents": [{"Key": "tenant/a/notes.txt", "Size": 1}]}]
        )
        result = backend.glob("*.txt")
        assert result.error is not None
        assert "malformed listing response" in result.error.lower()
