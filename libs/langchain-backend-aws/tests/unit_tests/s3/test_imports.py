"""Public-API surface tests for ``langchain_backend_aws``.

Locks the ``__all__`` shape on both the top-level package and the
``s3`` subpackage so an accidental rename, removal, or stray export
trips a unit test before reaching downstream callers. The set
comparison is exact — adding a new symbol requires updating this test
in the same change, which forces a deliberate decision about whether
the new name is part of the supported surface.
"""

from __future__ import annotations

import langchain_backend_aws
from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws import s3 as s3_pkg


def test_top_level_all_is_exact() -> None:
    """Top-level package exports only the documented public names."""
    assert set(langchain_backend_aws.__all__) == {"S3Backend", "S3BackendConfig"}


def test_s3_subpackage_all_is_exact() -> None:
    """s3 subpackage exports only the documented public names."""
    assert set(s3_pkg.__all__) == {"BinaryReadMode", "S3Backend", "S3BackendConfig"}


def test_public_classes_resolve_to_same_object() -> None:
    """Top-level re-exports must be the same objects as in :mod:`s3`."""
    assert S3Backend is s3_pkg.S3Backend
    assert S3BackendConfig is s3_pkg.S3BackendConfig
