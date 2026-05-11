"""Tests for ``grep_max_pattern_metachars`` ReDoS guard."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_backend_aws import S3Backend, S3BackendConfig


def test_grep_metachar_cap_rejects_stacked_quantifiers() -> None:
    backend = S3Backend.from_kwargs(bucket="b", prefix="p/", client=MagicMock())
    backend._config = S3BackendConfig(
        bucket="b", prefix="p/", grep_max_pattern_metachars=10
    )
    pattern = "(" * 6 + "a" + ")" * 6 + "+"  # 13 metachars, source length 14

    result = backend.grep(pattern)

    assert result.error is not None
    assert "metacharacter" in result.error


def test_grep_metachar_cap_allows_realistic_patterns() -> None:
    mock_client = MagicMock()
    mock_client.get_paginator.return_value.paginate.return_value = iter([])
    backend = S3Backend.from_kwargs(bucket="b", prefix="p/", client=mock_client)

    result = backend.grep(r"def\s+\w+\(")

    assert result.error is None
