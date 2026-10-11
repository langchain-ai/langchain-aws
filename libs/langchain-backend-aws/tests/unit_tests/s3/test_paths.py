"""Unit tests for S3Backend path / prefix / traversal handling.

Covers path<->key mapping, traversal prevention (CLI + percent-encoded +
double-encoded variants), the ``_path_to_key`` symmetric defense, the
segment-aware prefix shape invariant, construction-time prefix
validation, ``require_prefix`` fail-closed semantics, the empty-object-
key defense, and the storage-prefix-violation fail-closed contract for
``ls``/``glob``/``grep``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws.s3._paths import key_to_path, path_to_key

from ._helpers import _make_backend

# ------------------------------------------------------------------
# Path mapping helpers
# ------------------------------------------------------------------


class TestPathMapping:
    """Tests for path-to-key and key-to-path conversions."""

    def test_path_to_key_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._path_to_key("/foo/bar.txt") == "foo/bar.txt"

    def test_path_to_key_with_prefix(self) -> None:
        backend, _ = _make_backend(prefix="workspace/session1")
        assert backend._path_to_key("/foo/bar.txt") == "workspace/session1/foo/bar.txt"

    def test_path_to_key_root(self) -> None:
        backend, _ = _make_backend(prefix="data")
        assert backend._path_to_key("/") == "data/"

    def test_path_to_key_root_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._path_to_key("/") == ""

    def test_key_to_path_no_prefix(self) -> None:
        backend, _ = _make_backend()
        assert backend._key_to_path("foo/bar.txt") == "/foo/bar.txt"

    def test_key_to_path_with_prefix(self) -> None:
        backend, _ = _make_backend(prefix="workspace")
        assert backend._key_to_path("workspace/foo/bar.txt") == "/foo/bar.txt"

    def test_round_trip(self) -> None:
        backend, _ = _make_backend(prefix="prefix")
        path = "/some/deep/file.py"
        assert backend._key_to_path(backend._path_to_key(path)) == path


# ------------------------------------------------------------------
# Path traversal prevention (user-facing API)
# ------------------------------------------------------------------


class TestPathTraversal:
    """Tests for path traversal attack prevention."""

    def test_traversal_in_read(self) -> None:
        backend, _ = _make_backend()
        result = backend.read("/../../../etc/passwd")
        assert result.error is not None
        assert "traversal" in result.error.lower()

    def test_traversal_in_write(self) -> None:
        backend, _ = _make_backend()
        result = backend.write("/../secret.txt", "data")
        assert result.error is not None

    def test_traversal_in_edit(self) -> None:
        backend, _ = _make_backend()
        result = backend.edit("/../secret.txt", "a", "b")
        assert result.error is not None

    def test_traversal_in_ls(self) -> None:
        backend, _ = _make_backend()
        result = backend.ls("/../")
        assert result.error is not None

    def test_traversal_in_glob(self) -> None:
        backend, _ = _make_backend()
        result = backend.glob("*.py", path="/../")
        assert result.error is not None

    def test_traversal_in_upload(self) -> None:
        backend, _ = _make_backend()
        result = backend.upload_files([("/../evil.txt", b"data")])
        assert result[0].error == "invalid_path"

    def test_traversal_in_download(self) -> None:
        backend, _ = _make_backend()
        result = backend.download_files(["/../evil.txt"])
        assert result[0].error == "invalid_path"

    def test_tilde_path_rejected(self) -> None:
        backend, _ = _make_backend()
        result = backend.read("~/secret")
        assert result.error is not None


# ------------------------------------------------------------------
# Defense-in-depth: ``_path_to_key`` mirrors ``_key_to_path``. ``_key_
# to_path`` already enforces ``startswith(self._prefix)`` on the listing
# path. The matching guard in ``_path_to_key`` is the put/get side of
# the same invariant — it catches the case where a future
# ``validate_path`` regression (Unicode normalization, percent-encoded
# ``..``) would otherwise let a single-object operation escape the
# configured prefix.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Double-percent-encoded traversal contract. ``path_to_key`` only
# decodes a single layer (``%2e%2e`` is rejected). Doubly-encoded
# variants such as ``%252e%252e`` must be treated as user data — *not*
# recursively decoded into ``..`` — because the upstream
# ``validate_path`` performs at most one decode pass.
# ------------------------------------------------------------------


class TestDoubleEncodedTraversal:
    def test_double_encoded_dot_dot_is_data_not_traversal(self) -> None:
        # ``%252e%252e`` should be treated as the literal 12-character
        # string and resolve into the key as data. It must NOT decode
        # to ``..`` and trip the traversal segment guard.
        key = path_to_key("/foo/%252e%252e/bar.txt", prefix="t/")
        assert key == "t/foo/%252e%252e/bar.txt"

    def test_double_encoded_slash_is_data_not_traversal(self) -> None:
        # Same contract for ``%252f`` (would decode to ``%2f`` then
        # ``/`` if recursively decoded).
        key = path_to_key("/foo/%252fbar.txt", prefix="t/")
        assert key == "t/foo/%252fbar.txt"

    def test_double_encoded_segment_only_dot_passes(self) -> None:
        # A segment that *would* decode to ``%2e`` (encoded dot) on a
        # single pass is data on a single pass too — the guard rejects
        # only the literal ``%2e`` segment that would itself decode to
        # ``.`` once.
        key = path_to_key("/foo/%252e/bar.txt", prefix="")
        assert key == "foo/%252e/bar.txt"


# ------------------------------------------------------------------
# Segment-aware prefix shape invariant. ``key_to_path`` uses a byte-
# prefix ``startswith(prefix)`` to verify a listed key is inside the
# namespace. With ``prefix="tenantA"`` (no trailing ``/``) the same
# check would also accept ``"tenantAB/..."``. The live path is safe
# because ``S3Backend.__init__`` always appends a trailing ``/``, but
# ``path_to_key`` / ``key_to_path`` are exported helpers; this invariant
# locks the contract so the bug cannot reappear.
# ------------------------------------------------------------------


class TestPrefixShapeInvariant:
    def test_path_to_key_rejects_prefix_without_trailing_slash(self) -> None:
        with pytest.raises(ValueError, match="must be empty or end with"):
            path_to_key("/foo.txt", prefix="tenantA")

    def test_key_to_path_rejects_prefix_without_trailing_slash(self) -> None:
        with pytest.raises(ValueError, match="must be empty or end with"):
            key_to_path("tenantA/foo.txt", prefix="tenantA")

    def test_path_to_key_accepts_empty_prefix(self) -> None:
        assert path_to_key("/foo.txt", prefix="") == "foo.txt"

    def test_path_to_key_accepts_prefix_with_trailing_slash(self) -> None:
        assert path_to_key("/foo.txt", prefix="tenantA/") == "tenantA/foo.txt"

    def test_key_to_path_accepts_empty_prefix(self) -> None:
        assert key_to_path("foo.txt", prefix="") == "/foo.txt"

    def test_key_to_path_accepts_prefix_with_trailing_slash(self) -> None:
        assert key_to_path("tenantA/foo.txt", prefix="tenantA/") == "/foo.txt"

    @pytest.mark.parametrize(
        "bad_prefix",
        ["..//tenantA/", "tenantA/../", "tenantA//inner/", "./tenantA/"],
    )
    def test_path_to_key_rejects_unsanitized_prefix(self, bad_prefix: str) -> None:
        # Defense-in-depth: ``S3BackendConfig.__post_init__`` already
        # rejects these segments at construction, but a future caller
        # that wires ``path_to_key`` up with a hand-rolled prefix would
        # otherwise bypass that check. The helper must reject the
        # boundary-string itself.
        with pytest.raises(ValueError, match="empty or traversal segment"):
            path_to_key("/foo.txt", prefix=bad_prefix)


# ------------------------------------------------------------------
# Construction-time rejection of traversal segments in ``prefix``.
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# ``S3BackendConfig.require_prefix`` fail-closed semantics.
# ------------------------------------------------------------------


class TestRequirePrefix:
    def test_require_prefix_rejects_empty_prefix(self) -> None:
        # Validation lives on ``S3BackendConfig.__post_init__`` so the
        # invariant is enforced wherever the dataclass is constructed,
        # not only in ``S3Backend.__init__``.
        with pytest.raises(ValueError, match="require_prefix"):
            S3BackendConfig(bucket="b", prefix="", require_prefix=True)

    def test_require_prefix_accepts_non_empty_prefix(self) -> None:
        config = S3BackendConfig(
            bucket="b", prefix="sessions/abc/", require_prefix=True
        )
        backend = S3Backend(config, client=MagicMock())
        assert backend._prefix == "sessions/abc/"

    def test_require_prefix_default_preserves_backwards_compat(self) -> None:
        backend = S3Backend.from_kwargs(bucket="b", prefix="", client=MagicMock())
        assert backend._prefix == ""


# ------------------------------------------------------------------
# Empty-object-key defense. With ``prefix=""`` and a path that
# normalizes to ``/`` the S3 object key would be the empty string; boto3
# surfaces this as a generic ``ParamValidationError``. Reject it close
# to the public API so callers see a clear "not a file" error and the
# backend never issues a degenerate ``GetObject(Key="")``.
# ------------------------------------------------------------------


class TestEmptyKeyDefense:
    def test_read_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.read("/")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.get_object.assert_not_called()

    def test_write_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.write("/", "hi")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.put_object.assert_not_called()

    def test_edit_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        result = backend.edit("/", "a", "b")
        assert result.error is not None
        assert "does not refer to a file" in result.error
        mock.get_object.assert_not_called()

    def test_upload_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        responses = backend.upload_files([("/", b"hi")])
        assert responses[0].error == "invalid_path"
        mock.put_object.assert_not_called()

    def test_download_root_path_rejected(self) -> None:
        backend, mock = _make_backend(prefix="")
        responses = backend.download_files(["/"])
        assert responses[0].error == "invalid_path"
        mock.get_object.assert_not_called()

    def test_ls_root_path_still_works(self) -> None:
        # ls/glob/grep operate on directories; an empty key for the
        # bucket root is intentional under no-prefix mode.
        backend, mock = _make_backend(prefix="")
        mock.get_paginator.return_value.paginate.return_value = []
        result = backend.ls("/")
        assert result.error is None


# ------------------------------------------------------------------
# Storage prefix violation: ls/grep/glob fail closed when the paginator
# returns keys outside the configured prefix.
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
