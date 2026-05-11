"""Construction-time rejection of traversal segments in ``prefix``.

``_key_to_path`` only enforces ``startswith(self._prefix)``; a prefix
like ``foo/../`` would normalize against the bucket root and silently
widen the isolation boundary. Validate at __init__ time so a bad
config raises rather than masquerading as a working backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig


class TestPrefixTraversal:
    def test_dot_dot_segment_rejected(self) -> None:
        with pytest.raises(ValueError, match="traversal segment"):
            S3Backend(S3BackendConfig(bucket="b", prefix="foo/../bar"))

    def test_leading_dot_dot_rejected(self) -> None:
        with pytest.raises(ValueError, match="traversal segment"):
            S3Backend(S3BackendConfig(bucket="b", prefix="../bar"))

    def test_single_dot_segment_rejected(self) -> None:
        with pytest.raises(ValueError, match="traversal segment"):
            S3Backend(S3BackendConfig(bucket="b", prefix="foo/./bar"))

    def test_double_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="traversal segment"):
            S3Backend(S3BackendConfig(bucket="b", prefix="foo//bar"))

    def test_clean_prefix_accepted(self) -> None:
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/abc"), client=MagicMock()
        )
        # ``_prefix`` always ends with a trailing slash on a non-empty
        # prefix; clean input survives normalization unchanged.
        assert backend._prefix == "tenant/abc/"  # noqa: SLF001

    def test_empty_prefix_still_allowed_with_warning(self) -> None:
        # Empty prefix is allowed unless ``require_prefix=True``; the
        # traversal guard must not trip on the empty string.
        backend = S3Backend(S3BackendConfig(bucket="b", prefix=""), client=MagicMock())
        assert backend._prefix == ""  # noqa: SLF001

    def test_empty_bucket_rejected(self) -> None:
        # ``bucket=""`` would otherwise reach boto3 and surface as a
        # generic ParamValidationError on the first call. Reject it at
        # construction so the misuse is contained to the dataclass.
        with pytest.raises(ValueError, match="bucket"):
            S3BackendConfig(bucket="")

    def test_blank_region_rejected(self) -> None:
        with pytest.raises(ValueError, match="region_name"):
            S3BackendConfig(bucket="b", region_name="   ")
