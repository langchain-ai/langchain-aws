"""Test the segment-aware prefix invariant in :mod:`._paths`.

:func:`langchain_backend_aws.s3._paths.key_to_path` uses a byte-prefix
``startswith(prefix)`` to verify a listed key is inside the configured
namespace. With ``prefix="tenantA"`` (no trailing ``/``) the same check
would also accept ``"tenantAB/..."`` — a cross-tenant boundary bug.
The live path is safe because :class:`S3Backend.__init__` always
appends a trailing ``/`` to non-empty prefixes, but
:func:`path_to_key` / :func:`key_to_path` are exported helpers and a
future caller could wire them up directly. The invariant assert below
locks the contract so the bug cannot reappear.
"""

from __future__ import annotations

import pytest

from langchain_backend_aws.s3._paths import key_to_path, path_to_key


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
