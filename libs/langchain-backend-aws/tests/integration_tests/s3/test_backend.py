"""Integration tests for :class:`S3Backend` against a live S3-compatible store.

These tests require an S3-compatible endpoint (real S3 or MinIO/LocalStack)
and are gated on environment variables. To run against MinIO locally:

.. code-block:: bash

    docker run -d -p 9000:9000 -p 9001:9001 \\
        -e MINIO_ROOT_USER=minioadmin \\
        -e MINIO_ROOT_PASSWORD=minioadmin \\
        minio/minio:latest server /data --console-address ":9001"

    export S3_BACKEND_ENDPOINT_URL=http://localhost:9000
    export S3_BACKEND_ACCESS_KEY=minioadmin
    export S3_BACKEND_SECRET_KEY=minioadmin
    export S3_BACKEND_BUCKET=langchain-backend-aws-tests

    make integration_tests
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

pytestmark = pytest.mark.skipif(
    not REQUIRED_VARS_PRESENT,
    reason=(
        "Set S3_BACKEND_ACCESS_KEY and S3_BACKEND_SECRET_KEY (and optionally "
        "S3_BACKEND_ENDPOINT_URL for MinIO) to run integration tests."
    ),
)


@pytest.fixture(scope="module")
def s3_client() -> Any:
    """Build a low-level boto3 S3 client for fixture setup/teardown."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


@pytest.fixture(scope="module", autouse=True)
def ensure_bucket(s3_client: Any) -> None:
    """Create the test bucket if it does not exist."""
    try:
        s3_client.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=BUCKET)


@pytest.fixture
def backend(s3_client: Any) -> Iterator[S3Backend]:
    """Provide an S3Backend rooted at a unique per-test prefix.

    The prefix is cleaned up after the test, leaving the bucket reusable
    across runs.
    """
    prefix = f"it/{uuid.uuid4().hex}"
    config = S3BackendConfig(
        bucket=BUCKET,
        prefix=prefix,
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        allow_private_endpoints=True,
    )
    yield S3Backend(config)

    # Teardown: delete every key under the prefix.
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix}/")
        for obj in page.get("Contents", [])
    ]
    if keys:
        s3_client.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": k} for k in keys]},
        )


# ----------------------------------------------------------------------
# write / read / edit round trips
# ----------------------------------------------------------------------


def test_write_then_read(backend: S3Backend) -> None:
    write_result = backend.write("/hello.txt", "hello\nworld\n")
    assert write_result.error is None
    assert write_result.path == "/hello.txt"

    read_result = backend.read("/hello.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    # splitlines() drops the trailing newline, then "\n".join restores
    # interior newlines only.
    assert read_result.file_data["content"] == "hello\nworld"


def test_write_existing_errors(backend: S3Backend) -> None:
    backend.write("/conflict.txt", "v1")
    second = backend.write("/conflict.txt", "v2")
    assert second.error is not None
    assert "already exists" in second.error


def test_read_offset_and_limit(backend: S3Backend) -> None:
    content = "\n".join(f"line{i}" for i in range(20))
    backend.write("/big.txt", content)

    result = backend.read("/big.txt", offset=5, limit=3)
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "line5\nline6\nline7"


def test_edit_replaces_string(backend: S3Backend) -> None:
    backend.write("/note.txt", "the quick brown fox")
    edit_result = backend.edit("/note.txt", "quick", "slow")
    assert edit_result.error is None
    assert edit_result.occurrences == 1

    after = backend.read("/note.txt")
    assert after.file_data is not None
    assert after.file_data["content"] == "the slow brown fox"


def test_edit_conflict_on_concurrent_modification(
    backend: S3Backend, s3_client: Any
) -> None:
    """A concurrent writer between read and write must produce a conflict.

    Validates that the optimistic-concurrency contract (``IfMatch=<ETag>``
    on the conditional ``PutObject``) is honored end-to-end against the
    target store. Without this, a concurrent write would be silently
    overwritten.
    """
    backend.write("/race.txt", "v1")

    # Capture ETag the same way edit() does, then race a foreign writer
    # to mutate the object so the ETag the backend will pass back is
    # stale by the time put_object fires.
    response = s3_client.get_object(
        Bucket=BUCKET,
        Key=f"{backend._prefix}race.txt",  # noqa: SLF001
    )
    body = response["Body"]
    try:
        body.read()
    finally:
        body.close()

    # Foreign concurrent write — bumps ETag.
    s3_client.put_object(
        Bucket=BUCKET,
        Key=f"{backend._prefix}race.txt",
        Body=b"v2",  # noqa: SLF001
    )

    # The next edit() reads the (now-current) ETag, but if we patch the
    # backend to remember a stale ETag we exercise the conditional path.
    # Simpler: rely on edit()'s natural read-then-write window by issuing
    # two near-simultaneous edits via two backend instances. To keep the
    # test deterministic without threads, simulate the stale ETag path
    # by overwriting between backend.read and backend.put.
    config = S3BackendConfig(
        bucket=BUCKET,
        prefix=backend._prefix.rstrip("/"),  # noqa: SLF001
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        allow_private_endpoints=True,
    )
    racer = S3Backend(config)

    original_get_object = racer._client.get_object  # noqa: SLF001

    def get_then_overwrite(*args: object, **kwargs: object) -> dict:
        response = original_get_object(*args, **kwargs)
        # Foreign writer steps in between get_object and put_object.
        s3_client.put_object(
            Bucket=BUCKET,
            Key=f"{backend._prefix}race.txt",  # noqa: SLF001
            Body=b"v3",
        )
        return response

    racer._client.get_object = get_then_overwrite  # type: ignore[method-assign]  # noqa: SLF001

    result = racer.edit("/race.txt", "v2", "v4")
    assert result.error is not None
    assert "modified concurrently" in result.error or "Conflict" in result.error


def test_read_missing_file(backend: S3Backend) -> None:
    result = backend.read("/does-not-exist.txt")
    assert result.error is not None
    assert "not found" in result.error


# ----------------------------------------------------------------------
# ls / glob
# ----------------------------------------------------------------------


def test_ls_returns_files_and_dirs(backend: S3Backend) -> None:
    backend.write("/a/file1.txt", "x")
    backend.write("/a/file2.txt", "y")
    backend.write("/a/sub/file3.txt", "z")

    result = backend.ls("/a")
    assert result.error is None
    assert result.entries is not None

    paths = {e["path"] for e in result.entries}
    assert "/a/file1.txt" in paths
    assert "/a/file2.txt" in paths
    # Sub-directory should appear as virtual directory
    assert any(e.get("is_dir") and e["path"].endswith("/sub/") for e in result.entries)


def test_glob_matches(backend: S3Backend) -> None:
    backend.write("/code/main.py", "print('hi')")
    backend.write("/code/util.py", "x = 1")
    backend.write("/code/notes.md", "notes")

    result = backend.glob("*.py", path="/code")
    assert result.error is None
    assert result.matches is not None
    paths = sorted(m["path"] for m in result.matches)
    assert paths == ["/code/main.py", "/code/util.py"]


# ----------------------------------------------------------------------
# grep
# ----------------------------------------------------------------------


def test_grep_finds_pattern(backend: S3Backend) -> None:
    backend.write("/docs/a.txt", "alpha\nbeta\ngamma\n")
    backend.write("/docs/b.txt", "delta\nbeta-2\n")
    backend.write("/docs/c.bin", "ignored")

    result = backend.grep(r"beta", path="/docs", glob="*.txt")
    assert result.error is None
    assert result.matches is not None

    paths = {m["path"] for m in result.matches}
    assert paths == {"/docs/a.txt", "/docs/b.txt"}


def test_grep_invalid_regex(backend: S3Backend) -> None:
    result = backend.grep("[unterminated", path="/")
    assert result.error is not None
    assert "Invalid regex" in result.error


# ----------------------------------------------------------------------
# upload / download (binary)
# ----------------------------------------------------------------------


def test_upload_and_download_binary(backend: S3Backend) -> None:
    binary_payload = bytes(range(256))
    upload = backend.upload_files([("/blob.bin", binary_payload)])
    assert upload[0].error is None

    download = backend.download_files(["/blob.bin"])
    assert download[0].error is None
    assert download[0].content == binary_payload


def test_download_missing_returns_file_not_found(backend: S3Backend) -> None:
    result = backend.download_files(["/missing.bin"])
    assert result[0].error == "file_not_found"
    assert result[0].content is None


# ----------------------------------------------------------------------
# path traversal guard (security)
# ----------------------------------------------------------------------


def test_path_traversal_rejected(backend: S3Backend) -> None:
    result = backend.read("/../escape.txt")
    assert result.error is not None
    assert "traversal" in result.error.lower()
