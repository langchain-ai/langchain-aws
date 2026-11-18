"""Live-store check that ``require_prefix=True`` fails closed."""

from __future__ import annotations

import pytest

from langchain_backend_aws import S3BackendConfig

from .conftest import (
    ACCESS_KEY,
    BUCKET,
    ENDPOINT_URL,
    REGION,
    SECRET_KEY,
    skip_without_credentials,
)

pytestmark = skip_without_credentials


def test_require_prefix_rejects_empty_prefix_live() -> None:
    # Validation runs in ``S3BackendConfig.__post_init__``, so the
    # ``ValueError`` is raised at config-construction time — before the
    # backend ever touches S3. Wrap the config call accordingly.
    with pytest.raises(ValueError, match="require_prefix"):
        S3BackendConfig(
            bucket=BUCKET,
            prefix="",
            region_name=REGION,
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            require_prefix=True,
            allow_private_endpoints=True,
        )
