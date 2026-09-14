"""Shared fixtures and helpers for the integration suite.

Centralises the env-var gate, bucket creation, per-test prefix
allocation and the ``S3BackendConfig`` builder so the per-feature test
modules stay focused on assertions.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from langchain_backend_aws import S3Backend, S3BackendConfig

ENDPOINT_URL = os.getenv("S3_BACKEND_ENDPOINT_URL")
ACCESS_KEY = os.getenv("S3_BACKEND_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_BACKEND_SECRET_KEY")
BUCKET = os.getenv("S3_BACKEND_BUCKET", "langchain-backend-aws-tests")
REGION = os.getenv("S3_BACKEND_REGION", "us-east-1")

REQUIRED_VARS_PRESENT = bool(ACCESS_KEY and SECRET_KEY)

skip_without_credentials = pytest.mark.skipif(
    not REQUIRED_VARS_PRESENT,
    reason="S3_BACKEND_ACCESS_KEY/S3_BACKEND_SECRET_KEY required.",
)


@pytest.fixture(scope="module")
def s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


@pytest.fixture(scope="module", autouse=True)
def ensure_bucket(request: pytest.FixtureRequest) -> None:
    # Skip bucket setup for compile-only tests and when credentials are absent;
    # compile tests just verify imports and must not require AWS access.
    if not REQUIRED_VARS_PRESENT:
        return
    s3_client = request.getfixturevalue("s3_client")
    try:
        s3_client.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=BUCKET)


@pytest.fixture
def prefix(s3_client: Any) -> Iterator[str]:
    p = f"it/{uuid.uuid4().hex}"
    yield p
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{p}/")
        for obj in page.get("Contents", [])
    ]
    if keys:
        s3_client.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": k} for k in keys]},
        )


def make_backend(prefix_value: str, **extra: object) -> S3Backend:
    # Integration tests run against LocalStack/MinIO on a private host;
    # opt in to private endpoints so the construction-time SSRF guard
    # does not block the tests. Production callers leave this off.
    extra.setdefault("allow_private_endpoints", True)
    config = S3BackendConfig(
        bucket=BUCKET,
        prefix=prefix_value,
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **extra,  # type: ignore[arg-type]
    )
    return S3Backend(config)
