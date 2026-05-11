"""``glob`` enumeration for :class:`S3Backend`.

Extracted from ``backend.py`` to keep the protocol surface compact;
behavior is unchanged from the inlined implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

import regex as regex_mod
from botocore.exceptions import ClientError
from deepagents.backends.protocol import FileInfo, GlobResult

from langchain_backend_aws.s3._internal import (
    LOG_TRIM,
    glob_matches,
    normalize_listing_size,
    sanitize_error_code,
)

logger = logging.getLogger(__name__)


def glob_search(
    client: Any,
    bucket: str,
    prefix: str,
    base_key: str,
    pattern: str,
    path: str,
    key_to_path: Callable[[str], str],
    format_timestamp: Callable[[datetime], str],
    max_objects: int,
    timeout: float,
    compile_glob: Callable[[str], regex_mod.Pattern[str]],
) -> GlobResult:
    """Enumerate objects under ``base_key`` and return glob matches.

    Mirrors the contract of :meth:`S3Backend.glob`: fails closed when
    the cap is hit or when the paginator returns a key outside the
    configured prefix.

    ``compile_glob`` is required: callers must provide a per-context
    glob translator (typically the per-instance compiler produced by
    :func:`make_glob_compiler`) so a process-global cache cannot bridge
    tenants.
    """
    # Defence-in-depth: ``base_key`` must be either empty (root scope) or
    # end with ``/`` so the ``startswith`` check below cannot accept a
    # sibling key. The caller (``S3Backend.glob``) enforces this shape;
    # this guard surfaces a future regression even under ``python -O``
    # (which strips ``assert`` statements).
    if base_key != "" and not base_key.endswith("/"):
        msg = f"base_key must be empty or end with '/', got {base_key!r}"
        raise AssertionError(msg)
    compiled = compile_glob(pattern)
    # Reusing the same compiled regex is sound: the translator produces
    # a path-aware pattern, but with no ``/`` in the input it is
    # equivalent against either the full relative path or the basename
    # alone.
    basename_regex = compiled if "/" not in pattern else None

    matches: list[FileInfo] = []
    scanned = 0
    truncated = False
    prefix_violation: str | None = None

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=base_key):
            for obj in page.get("Contents", []):
                if scanned >= max_objects:
                    truncated = True
                    break
                scanned += 1

                obj_key: str = obj["Key"]
                # Validate prefix containment BEFORE any glob/size
                # filtering — same ordering as ``grep`` — so a
                # paginator that leaks keys outside the configured
                # prefix cannot be silently masked when the leaked key
                # happens not to match the user's glob. Fail-closed via
                # ``GlobResult.error`` per deepagents' virtual-FS
                # contract: never raise across the protocol boundary.
                try:
                    virtual_path = key_to_path(obj_key)
                except ValueError:
                    prefix_violation = obj_key
                    break
                if not obj_key.startswith(base_key):
                    prefix_violation = obj_key
                    break
                relative = obj_key[len(base_key) :]
                if not relative:
                    continue
                if not glob_matches(
                    relative, compiled, basename_regex, timeout=timeout
                ):
                    continue
                matches.append(
                    {
                        "path": virtual_path,
                        "is_dir": False,
                        "size": normalize_listing_size(obj.get("Size")),
                        "modified_at": format_timestamp(obj["LastModified"]),
                    }
                )

            if prefix_violation is not None or truncated:
                break
    except ClientError as exc:
        logger.exception("Error during glob '%s' at '%s'", pattern, path)
        code = sanitize_error_code(exc)
        return GlobResult(error=f"Error during glob '{pattern}' at '{path}': {code}")
    except TimeoutError as exc:
        # ``regex.Pattern.match`` raised TimeoutError because a single
        # match attempt exceeded ``glob_regex_timeout``. Fail closed —
        # a partial match list would let callers conclude a path does
        # not exist when it merely was not finished scanning.
        logger.error(
            "glob: regex match exceeded glob_regex_timeout=%.3fs for "
            "pattern '%s' at '%s'",
            timeout,
            pattern,
            path,
        )
        return GlobResult(
            error=(
                f"Error during glob '{pattern}' at '{path}': regex match "
                f"exceeded glob_regex_timeout={timeout}s "
                f"(catastrophic backtracking suspected): {exc}"
            )
        )
    except Exception:  # noqa: BLE001
        # Fail closed on unexpected paginator response shapes. The loop
        # indexes directly into ``obj["Key"]`` / ``obj["LastModified"]``,
        # so a malformed or stubbed S3 response can raise
        # ``KeyError`` / ``TypeError``. The ``BackendProtocol`` contract
        # is "return error, do not raise", so we surface a generic glob
        # error rather than letting an implementation detail propagate
        # across the protocol boundary. ``logger.exception`` captures
        # the real cause for triage.
        logger.exception(
            "glob: unexpected response shape for pattern '%s' at '%s'; failing closed",
            pattern[:LOG_TRIM],
            path[:LOG_TRIM],
        )
        return GlobResult(
            error=(
                f"Error during glob '{pattern}' at '{path}': malformed "
                f"listing response (failing closed)."
            )
        )

    if prefix_violation is not None:
        # Trim the offending key/prefix/pattern/path before logging so a
        # pathologically long caller-supplied or paginator-returned
        # value cannot bloat the log line — only triage context is
        # needed.
        logger.error(
            "glob: storage prefix violation — S3 returned '%s' outside "
            "configured prefix '%s' while matching '%s' at '%s'; "
            "failing closed",
            prefix_violation[:LOG_TRIM],
            prefix[:LOG_TRIM],
            pattern[:LOG_TRIM],
            path[:LOG_TRIM],
        )
        return GlobResult(
            error=(
                f"Error during glob '{pattern}' at '{path}': storage "
                f"prefix violation (returned key outside configured "
                f"prefix)."
            )
        )

    if truncated:
        # Fail closed: the GlobResult contract has no truncation field,
        # so silently returning partial matches would let callers
        # conclude a file does not exist when it merely was not
        # scanned. Surface this as an error instead.
        return GlobResult(
            error=(
                f"glob scan exceeded glob_max_objects={max_objects} "
                f"under '{path}'. Narrow the path or raise the limit."
            )
        )

    matches.sort(key=lambda e: e.get("path", ""))
    return GlobResult(matches=matches)
