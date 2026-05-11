"""Robustness tests for grep against malformed S3 responses.

A malformed ``GetObject`` response (missing ``Body``/``LastModified``)
must not propagate as an unhandled exception across the
``BackendProtocol`` boundary. ``grep`` should fail closed with a
sanitized error string instead — otherwise callers cannot distinguish a
crash from "no match".
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig


def _backend_with_listing(
    objects: list[dict[str, object]],
) -> tuple[S3Backend, MagicMock]:
    """Create an S3Backend whose paginator yields ``objects``."""
    mock_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = iter([{"Contents": objects}])
    mock_client.get_paginator.return_value = paginator
    backend = S3Backend(
        S3BackendConfig(bucket="b", prefix="tenant/a"),
        client=mock_client,
    )
    return backend, mock_client


class TestGrepMalformedResponse:
    def test_grep_missing_body_fails_closed(self) -> None:
        backend, mock = _backend_with_listing(
            [{"Key": "tenant/a/notes.txt", "Size": 5}]
        )
        # ``Body`` missing — ``read_capped_object`` classifies this as
        # :class:`OversizeError` (fail-closed) so the
        # ClientError/Oversize disjoint hierarchy stays clean.
        # ``_fetch_object`` then surfaces the backend-specific
        # ``"oversize"`` tag and ``_visit_object`` fail-closes the
        # entire grep run rather than silently dropping the object.
        mock.get_object.return_value = {
            "ContentLength": 5,
            "ETag": '"x"',
            "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
        }

        result = backend.grep("foo")

        assert result.matches is None or result.matches == []
        assert result.error is not None
        # The error message must not leak the stack trace; "oversize"
        # (from the OversizeError mapping) and "unexpected error"
        # (legacy broad-guard path) are both acceptable sanitized
        # surfaces.
        lowered = result.error.lower()
        assert (
            "oversize" in lowered
            or "unexpected error" in lowered
            or "keyerror" in lowered
        )

    def test_grep_malformed_listing_entry_fails_closed(self) -> None:
        # ``Size`` missing or non-numeric on a Contents entry must be
        # caught by the broad-except guard in ``_safe_visit`` and
        # surface as a sanitized error rather than an unhandled
        # ``KeyError``/``ValueError`` across the protocol boundary.
        backend, _mock = _backend_with_listing(
            [{"Key": "tenant/a/notes.txt"}]  # missing Size
        )

        result = backend.grep("foo")

        # No matches; error must be sanitized (not a raw traceback).
        assert result.matches is None or result.matches == []
        assert result.error is not None
        assert "Traceback" not in result.error
