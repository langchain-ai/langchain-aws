"""``ls`` enumeration for :class:`S3Backend`.

Extracted from ``backend.py`` to keep the protocol surface compact;
behavior is unchanged from the inlined implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from botocore.exceptions import ClientError
from deepagents.backends.protocol import FileInfo, LsResult

from langchain_backend_aws.s3._internal import (
    LOG_TRIM,
    normalize_listing_size,
    sanitize_error_code,
)

logger = logging.getLogger(__name__)


def ls_listing(
    client: Any,
    bucket: str,
    prefix: str,
    path: str,
    key_prefix: str,
    key_to_path: Callable[[str], str],
    format_timestamp: Callable[[datetime], str],
    max_objects: int,
) -> LsResult:
    """Enumerate objects and common prefixes under ``key_prefix``.

    Mirrors the contract of :meth:`S3Backend.ls`: fails closed when the
    cap is hit or when the paginator returns a key outside the
    configured prefix.
    """
    entries: list[FileInfo] = []
    scanned = 0
    truncated = False
    prefix_violation: str | None = None

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=key_prefix,
            Delimiter="/",
        ):
            for obj in page.get("Contents", []):
                obj_key: str = obj["Key"]
                if obj_key == key_prefix:
                    continue
                if scanned >= max_objects:
                    truncated = True
                    break
                scanned += 1
                try:
                    virtual_path = key_to_path(obj_key)
                except ValueError:
                    prefix_violation = obj_key
                    break
                entries.append(
                    {
                        "path": virtual_path,
                        "is_dir": False,
                        "size": normalize_listing_size(obj.get("Size")),
                        "modified_at": format_timestamp(obj["LastModified"]),
                    }
                )

            if prefix_violation is not None or truncated:
                break

            for prefix_entry in page.get("CommonPrefixes", []):
                if scanned >= max_objects:
                    truncated = True
                    break
                scanned += 1
                dir_key: str = prefix_entry["Prefix"]
                try:
                    virtual_dir = key_to_path(dir_key)
                except ValueError:
                    prefix_violation = dir_key
                    break
                entries.append(
                    {
                        "path": virtual_dir,
                        "is_dir": True,
                    }
                )

            if prefix_violation is not None or truncated:
                break
    except ClientError as exc:
        logger.exception("Error listing '%s'", path)
        return LsResult(error=f"Error listing '{path}': {sanitize_error_code(exc)}")
    except Exception:  # noqa: BLE001
        # Fail closed on unexpected paginator response shapes. The body
        # of the loop indexes directly into ``obj["Key"]`` / ``obj["LastModified"]``
        # / ``prefix_entry["Prefix"]``, so a malformed or stubbed S3
        # response can raise ``KeyError`` / ``TypeError``. The
        # ``BackendProtocol`` contract is "return error, do not raise",
        # so we surface a generic listing error rather than letting an
        # implementation detail propagate across the protocol boundary.
        # ``logger.exception`` captures the real cause for triage.
        logger.exception(
            "ls: unexpected response shape while listing '%s'; failing closed",
            path[:LOG_TRIM],
        )
        return LsResult(
            error=(
                f"Error listing '{path}': malformed listing response (failing closed)."
            )
        )

    if prefix_violation is not None:
        logger.error(
            "ls: storage prefix violation — S3 returned '%s' outside "
            "configured prefix '%s' while listing '%s'; failing closed",
            prefix_violation[:LOG_TRIM],
            prefix[:LOG_TRIM],
            path[:LOG_TRIM],
        )
        return LsResult(
            error=(
                f"Error listing '{path}': storage prefix violation "
                f"(returned key outside configured prefix)."
            )
        )

    if truncated:
        return LsResult(
            error=(
                f"ls scan exceeded ls_max_objects={max_objects} under "
                f"'{path}'. Narrow the path or raise the limit."
            )
        )

    # Sort by (is_dir, path) so files always precede directories at the
    # same level. S3 allows a key (``a``) and a common prefix (``a/``) to
    # coexist, and a single sort on ``path`` would produce
    # implementation-defined ordering for that pair. Pin the order
    # explicitly: files first, then directories, each group sorted by
    # path.
    entries.sort(key=lambda e: (bool(e.get("is_dir", False)), e.get("path", "")))
    return LsResult(entries=entries)
