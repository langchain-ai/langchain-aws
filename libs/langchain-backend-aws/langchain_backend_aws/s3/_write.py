"""``write`` / ``edit`` implementations for :class:`S3Backend`.

Extracted from ``backend.py`` so that file stays under the project's
800-line guideline; behavior is unchanged from the inlined
implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from botocore.exceptions import ClientError
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.utils import perform_string_replacement

from langchain_backend_aws.s3._internal import (
    CappedReadResult,
    OversizeError,
    format_oversize_message,
    is_not_found,
    is_precondition_failed,
    sanitize_error_code,
)

logger = logging.getLogger(__name__)


def write_file(
    file_path: str,
    content: str,
    *,
    client: Any,
    bucket: str,
    path_to_file_key: Callable[[str], str],
) -> WriteResult:
    """Write a new file to S3 with ``IfNoneMatch="*"``.

    See :meth:`S3Backend.write` for the public contract; this helper
    holds the body so ``backend.py`` stays focused on the protocol
    surface.
    """
    try:
        key = path_to_file_key(file_path)
    except ValueError as exc:
        return WriteResult(error=str(exc))

    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),
            IfNoneMatch="*",
        )
        # NOTE: deepagents deprecated the ``files_update`` field; state
        # updates are handled internally by the backend, so we omit the
        # field entirely. Setting it (even to None) emits a
        # DeprecationWarning.
        return WriteResult(path=file_path)
    except ClientError as exc:
        if is_precondition_failed(exc):
            return WriteResult(
                error=(
                    f"Cannot write to {file_path!r} because it already exists. "
                    "Read and then make an edit, or write to a new path."
                )
            )
        logger.exception("Error writing '%s'", file_path)
        return WriteResult(
            error=f"Error writing '{file_path}': {sanitize_error_code(exc)}"
        )


def _load_for_edit(
    file_path: str,
    read_capped: Callable[[str], CappedReadResult],
    key: str,
    max_file_size_mb: int,
) -> tuple[CappedReadResult | None, EditResult | None]:
    """Read the current object body and classify read-side failures.

    Returns ``(fetched, None)`` on a successful read, or
    ``(None, error_result)`` when the read failed in a way the caller
    must surface unchanged (not-found, denied, oversize, missing ETag,
    non-UTF-8 body). Splitting this off from :func:`edit_file` keeps
    the orchestrator short and lets the read-side classification be
    unit-tested independently of the conditional-PUT path.
    """
    try:
        fetched = read_capped(key)
    except ClientError as exc:
        if is_not_found(exc):
            return None, EditResult(error=f"Error: File '{file_path}' not found")
        logger.exception("Error reading '%s'", file_path)
        return None, EditResult(
            error=f"Error reading '{file_path}': {sanitize_error_code(exc)}"
        )
    except OversizeError as exc:
        return None, EditResult(
            error=format_oversize_message(file_path, max_file_size_mb, exc)
        )

    if not fetched.etag:
        # Without an ETag we cannot enforce optimistic concurrency via
        # ``IfMatch``; falling back to an unconditional PUT would
        # silently overwrite a concurrent writer's change. Fail closed
        # instead — it is safer for the caller to retry than to lose a
        # write.
        return None, EditResult(
            error=(
                f"Cannot edit '{file_path}': S3 did not return an ETag, "
                "so the optimistic-concurrency precondition cannot be "
                "applied."
            )
        )

    try:
        fetched.raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, EditResult(error=f"Error: File '{file_path}' is not a text file")
    return fetched, None


def _apply_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,  # noqa: FBT001
) -> tuple[str, int] | EditResult:
    """Run :func:`perform_string_replacement` and lift its error shape.

    Wraps the deepagents helper so the orchestrator deals with a single
    union — either the new content + occurrence count or a fully-formed
    :class:`EditResult` carrying the helper's error message.
    """
    result = perform_string_replacement(content, old_string, new_string, replace_all)
    if isinstance(result, str):
        return EditResult(error=result)
    new_content, occurrences = result
    return new_content, int(occurrences)


def _conditional_put(
    file_path: str,
    new_content: str,
    *,
    client: Any,
    bucket: str,
    key: str,
    etag: str,
) -> EditResult | None:
    """Issue the conditional ``PutObject`` and classify write failures.

    Returns ``None`` on success (caller composes the final
    :class:`EditResult` with ``occurrences``); returns a populated
    :class:`EditResult` when the ``IfMatch`` precondition fails or the
    write hits another :class:`ClientError`.
    """
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=new_content.encode("utf-8"),
            # ETag from get_object is already a quoted string per RFC
            # 7232; pass it through unchanged so the conditional matches
            # the exact version we read.
            IfMatch=etag,
        )
    except ClientError as exc:
        if is_precondition_failed(exc):
            return EditResult(
                error=(
                    f"Conflict: {file_path!r} was modified concurrently. "
                    "Re-read the file and retry the edit."
                )
            )
        logger.exception("Error writing '%s'", file_path)
        return EditResult(
            error=f"Error writing '{file_path}': {sanitize_error_code(exc)}"
        )
    return None


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,  # noqa: FBT001
    *,
    client: Any,
    bucket: str,
    path_to_file_key: Callable[[str], str],
    read_capped: Callable[[str], CappedReadResult],
    max_file_size_mb: int,
) -> EditResult:
    """Edit a file via optimistic-concurrency ``IfMatch`` PUT.

    See :meth:`S3Backend.edit` for the public contract. The body is
    decomposed into three steps so each stage's failure modes are
    classified close to the operation that produced them:

    1. :func:`_load_for_edit` — read + classify (not-found, denied,
       oversize, missing ETag, non-UTF-8).
    2. :func:`_apply_replacement` — run the deepagents string-replacement
       helper and lift its error shape into :class:`EditResult`.
    3. :func:`_conditional_put` — issue the ``IfMatch`` PUT and classify
       precondition / write failures.
    """
    try:
        key = path_to_file_key(file_path)
    except ValueError as exc:
        return EditResult(error=str(exc))

    fetched, error = _load_for_edit(file_path, read_capped, key, max_file_size_mb)
    if fetched is None:
        if error is None:
            # _load_for_edit's union contract guarantees one branch is set;
            # surface a deterministic error rather than letting `None`
            # propagate if that contract is ever violated.
            msg = (
                f"Internal error reading '{file_path}': loader returned "
                "neither fetched bytes nor an error."
            )
            raise RuntimeError(msg)
        return error

    content = fetched.raw_bytes.decode("utf-8")
    replacement = _apply_replacement(content, old_string, new_string, replace_all)
    if isinstance(replacement, EditResult):
        return replacement
    new_content, occurrences = replacement

    write_error = _conditional_put(
        file_path,
        new_content,
        client=client,
        bucket=bucket,
        key=key,
        etag=fetched.etag,
    )
    if write_error is not None:
        return write_error
    return EditResult(path=file_path, occurrences=occurrences)
