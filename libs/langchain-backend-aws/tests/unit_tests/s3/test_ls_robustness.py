"""Robustness tests for ``ls`` against malformed S3 listings.

A malformed paginator page (missing ``Key`` / ``LastModified`` /
``Prefix``) must not propagate as an unhandled exception across the
``BackendProtocol`` boundary. ``ls`` should fail closed with a sanitized
error string instead — otherwise callers cannot distinguish a crash
from an empty listing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig


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
