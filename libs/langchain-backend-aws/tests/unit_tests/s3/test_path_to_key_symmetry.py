"""Defense-in-depth check: ``_path_to_key`` mirrors ``_key_to_path``.

``_key_to_path`` already enforces ``startswith(self._prefix)`` on the
listing path. The matching guard in ``_path_to_key`` is the put/get
side of the same invariant — it catches the case where a future
``validate_path`` regression (Unicode normalization, percent-encoded
``..``) would otherwise let a single-object operation escape the
configured prefix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig


class TestPathToKeySymmetry:
    def test_clean_path_resolves_under_prefix(self) -> None:
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        assert backend._path_to_key("/foo/bar.txt") == "tenant/a/foo/bar.txt"  # noqa: SLF001

    def test_validate_path_regression_caught(self) -> None:
        # Simulate a hypothetical ``validate_path`` regression that lets
        # a traversal sequence slip through unchanged. The defense-in-depth
        # check in ``_path_to_key`` must reject the resulting key.
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        with (
            patch(
                "langchain_backend_aws.s3._paths.validate_path",
                return_value="/../other-tenant/secret",
            ),
            pytest.raises(ValueError, match="traversal or empty segment"),
        ):
            backend._path_to_key("/anything")  # noqa: SLF001

    def test_empty_prefix_skips_check(self) -> None:
        # With ``prefix=""`` the bucket root is the boundary, so the
        # symmetric check is trivially satisfied for any normalized key.
        backend = S3Backend(S3BackendConfig(bucket="b", prefix=""), client=MagicMock())
        assert backend._path_to_key("/foo.txt") == "foo.txt"  # noqa: SLF001

    def test_nfkc_fullwidth_dot_traversal_rejected(self) -> None:
        # ``U+FF0E FULLWIDTH FULL STOP`` (``．``) NFKC-normalizes to
        # ``.``, so a full-width ``．．`` segment would expose the same
        # traversal surface as ``..`` if a future ``validate_path``
        # regressed on Unicode normalization. The defense-in-depth
        # check must reject it.
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        with (
            patch(
                "langchain_backend_aws.s3._paths.validate_path",
                return_value="/．．/secret",
            ),
            pytest.raises(ValueError, match="traversal or empty segment"),
        ):
            backend._path_to_key("/anything")  # noqa: SLF001

    def test_percent_encoded_traversal_rejected(self) -> None:
        # If a future ``validate_path`` regressed and let
        # percent-encoded ``..`` (``%2e%2e``) slip through unchanged,
        # the defense-in-depth check must still catch it before it
        # reaches S3 — boto3/S3 may URL-decode the key on the wire.
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        with (
            patch(
                "langchain_backend_aws.s3._paths.validate_path",
                return_value="/%2e%2e/secret",
            ),
            pytest.raises(ValueError, match="percent-encoded traversal"),
        ):
            backend._path_to_key("/anything")  # noqa: SLF001

    def test_percent_encoded_single_dot_segment_rejected(self) -> None:
        # A standalone ``%2e`` segment decodes to ``.`` on the wire and
        # would otherwise survive the literal segment scan (which sees
        # ``%2e`` as a regular four-character segment, not the dot).
        # The defense-in-depth check must reject it independently of
        # the upstream decoder behavior.
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        with (
            patch(
                "langchain_backend_aws.s3._paths.validate_path",
                return_value="/foo/%2e/bar",
            ),
            pytest.raises(ValueError, match="percent-encoded dot segment"),
        ):
            backend._path_to_key("/anything")  # noqa: SLF001

    def test_percent_encoded_single_dot_segment_uppercase_rejected(self) -> None:
        # Same case but with the uppercase ``%2E`` form — both must hit
        # the same guard since percent-encoding is case-insensitive.
        backend = S3Backend(
            S3BackendConfig(bucket="b", prefix="tenant/a"), client=MagicMock()
        )
        with (
            patch(
                "langchain_backend_aws.s3._paths.validate_path",
                return_value="/foo/%2E/bar",
            ),
            pytest.raises(ValueError, match="percent-encoded dot segment"),
        ):
            backend._path_to_key("/anything")  # noqa: SLF001
