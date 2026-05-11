"""Tests for ``S3BackendConfig.require_prefix`` fail-closed semantics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig


def test_require_prefix_rejects_empty_prefix() -> None:
    # Validation lives on ``S3BackendConfig.__post_init__`` so the
    # invariant is enforced wherever the dataclass is constructed,
    # not only in ``S3Backend.__init__``.
    with pytest.raises(ValueError, match="require_prefix"):
        S3BackendConfig(bucket="b", prefix="", require_prefix=True)


def test_require_prefix_accepts_non_empty_prefix() -> None:
    config = S3BackendConfig(bucket="b", prefix="sessions/abc/", require_prefix=True)
    backend = S3Backend(config, client=MagicMock())
    assert backend._prefix == "sessions/abc/"


def test_require_prefix_default_preserves_backwards_compat() -> None:
    backend = S3Backend.from_kwargs(bucket="b", prefix="", client=MagicMock())
    assert backend._prefix == ""
