"""``read`` implementation for :class:`S3Backend`.

Extracted from ``backend.py`` to keep that file under the 800-line
guideline; behavior is unchanged from the inlined implementation.
"""

from __future__ import annotations

import base64
import logging
from io import StringIO
from typing import Callable

from botocore.exceptions import ClientError
from deepagents.backends.protocol import ReadResult
from deepagents.backends.utils import create_file_data

from langchain_backend_aws.s3._config import BinaryReadMode
from langchain_backend_aws.s3._internal import (
    CappedReadResult,
    OversizeError,
    format_oversize_message,
    is_not_found,
    sanitize_error_code,
    strip_line_terminator,
)

logger = logging.getLogger(__name__)

# Defaults for ``read``'s ``offset``/``limit`` parameters. Kept in sync
# with :meth:`S3Backend.read` so the binary fallback below can detect
# when the caller supplied a non-default selection that will be ignored
# by the base64 path and warn — rather than silently returning the full
# body and surprising the caller with a much larger payload than the
# requested ``limit`` lines.
_DEFAULT_READ_OFFSET = 0
_DEFAULT_READ_LIMIT = 2000


def read_file(
    file_path: str,
    offset: int,
    limit: int,
    *,
    path_to_file_key: Callable[[str], str],
    read_capped: Callable[[str], CappedReadResult],
    format_timestamp: Callable[..., str],
    max_file_size_mb: int,
    binary_read_mode: BinaryReadMode,
) -> ReadResult:
    """Read file content with line-based pagination.

    See :meth:`S3Backend.read` for the public contract; this helper holds
    the body so ``backend.py`` stays focused on the protocol surface.
    """
    try:
        key = path_to_file_key(file_path)
    except ValueError as exc:
        return ReadResult(error=str(exc))

    try:
        fetched = read_capped(key)
    except ClientError as exc:
        if is_not_found(exc):
            return ReadResult(error=f"Error: File '{file_path}' not found")
        logger.exception("Error reading '%s'", file_path)
        return ReadResult(
            error=f"Error reading '{file_path}': {sanitize_error_code(exc)}"
        )
    except OversizeError as exc:
        return ReadResult(
            error=format_oversize_message(file_path, max_file_size_mb, exc)
        )

    timestamp = format_timestamp(fetched.last_modified)
    try:
        content = fetched.raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _binary_read_result(
            file_path, fetched, timestamp, binary_read_mode, offset, limit
        )

    return _paginate_text(content, offset, limit, timestamp)


def _binary_read_result(
    file_path: str,
    fetched: CappedReadResult,
    timestamp: str,
    binary_read_mode: BinaryReadMode,
    offset: int,
    limit: int,
) -> ReadResult:
    """Surface a non-UTF-8 body per the configured ``binary_read_mode``."""
    if binary_read_mode == "error":
        return ReadResult(
            error=(
                f"Error: File '{file_path}' is not UTF-8 text. "
                "Use ``download_files`` for byte-exact reads."
            )
        )
    if offset != _DEFAULT_READ_OFFSET or limit != _DEFAULT_READ_LIMIT:
        # ``offset``/``limit`` are line-based and have no meaning on a
        # non-UTF-8 body, so the base64 path returns the full (capped)
        # body. Warn so a caller that intentionally narrowed the
        # selection — e.g. ``limit=10`` to keep the LLM context small —
        # is not silently handed several MiB of base64 instead. Switch
        # ``binary_read_mode="error"`` to fail closed in that case.
        logger.warning(
            "read('%s') with offset=%d, limit=%d ignored on non-UTF-8 body "
            "(binary_read_mode='base64' returns the full capped body); "
            "set binary_read_mode='error' to fail closed.",
            file_path,
            offset,
            limit,
        )
    encoded = base64.standard_b64encode(fetched.raw_bytes).decode("ascii")
    return ReadResult(
        file_data=create_file_data(encoded, created_at=timestamp, encoding="base64")
    )


def _paginate_text(content: str, offset: int, limit: int, timestamp: str) -> ReadResult:
    """Apply line-based ``offset``/``limit`` to decoded text content.

    Iterates over the body via :class:`io.StringIO` so we avoid the
    extra resident copy a ``content.splitlines()`` list would add on
    top of ``content``. ``StringIO`` itself wraps ``content`` by
    reference, so the original string is still held while iterating —
    peak memory is roughly ``content`` (the body) plus the selected
    slice, not the ``2x`` of body + line list. At the
    ``max_file_size_mb`` cap (default 10 MiB) the saved copy is the
    dominant cost.
    """
    if offset < 0:
        return ReadResult(error=f"Line offset must be non-negative, got {offset}")
    if limit <= 0:
        return ReadResult(file_data=create_file_data("", created_at=timestamp))

    selected: list[str] = []
    seen = 0
    end_target = offset + limit
    buffer = StringIO(content)
    for raw_line in buffer:
        line = strip_line_terminator(raw_line)
        if seen >= offset:
            selected.append(line)
        seen += 1
        if seen >= end_target:
            break
    # ``offset == seen`` means the caller asked to skip exactly the
    # whole file; treat that as a valid empty selection rather than an
    # error. Only ``offset > seen`` is genuinely out of range.
    if offset > 0 and seen < offset:
        return ReadResult(
            error=f"Line offset {offset} exceeds file length ({seen} lines)"
        )
    return ReadResult(
        file_data=create_file_data("\n".join(selected), created_at=timestamp)
    )
