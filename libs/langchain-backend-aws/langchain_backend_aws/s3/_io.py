"""``upload_files`` / ``download_files`` helpers for :class:`S3Backend`.

Extracted from ``backend.py`` to keep the protocol surface compact and
to keep ``backend.py`` under the project's 800-line file-size guideline.
Behavior is unchanged from the inlined implementation.
"""

from __future__ import annotations

import hashlib
import logging
from asyncio import CancelledError as AsyncioCancelledError
from concurrent.futures import (
    CancelledError as FuturesCancelledError,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any, Callable

from botocore.exceptions import ClientError
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from langchain_backend_aws.s3._internal import (
    LOG_TRIM,
    OVERSIZE_ERROR_TAG,
    CappedReadResult,
    OversizeError,
    is_not_found,
    log_upload_download_error,
)

logger = logging.getLogger(__name__)


def _safe_path_repr(path: object) -> str:
    """Return a log-safe, length-bounded representation of ``path``.

    Defensive against non-string entries that could otherwise raise
    ``TypeError`` from slice operations inside an exception handler and
    mask the original failure.
    """
    if isinstance(path, str):
        return path[:LOG_TRIM]
    return f"<non-str path: {type(path).__name__}>"


def _fingerprint_path(path: object) -> str:
    """Return a short SHA-256 fingerprint of ``path`` for diagnostic logs.

    Used in branches that are expected to be unreachable but still need
    to surface enough information for triage. The fingerprint avoids
    leaking tenant identifiers embedded in the path while preserving
    set-equality so duplicate slot failures collapse to a single token.
    """
    if isinstance(path, str):
        digest = hashlib.sha256(path.encode("utf-8", errors="replace")).hexdigest()
        return f"sha256:{digest[:12]}"
    return f"<non-str path: {type(path).__name__}>"


def upload_files(
    client: Any,
    bucket: str,
    files: list[tuple[str, bytes]],
    *,
    path_to_file_key: Callable[[str], str],
    max_bytes: int,
    max_file_size_mb: int,
) -> list[FileUploadResponse]:
    """Upload multiple files to S3 with per-file size cap.

    Mirrors :meth:`S3Backend.upload_files`. Oversized uploads surface
    the backend-specific tag ``"oversize"`` (see
    :data:`OVERSIZE_ERROR_TAG` for the contract). The real cause is also
    logged at ERROR for triage.
    """
    responses: list[FileUploadResponse] = []
    for path, content in files:
        try:
            key = path_to_file_key(path)
        except ValueError:
            responses.append(FileUploadResponse(path=path, error="invalid_path"))
            continue
        if len(content) > max_bytes:
            logger.error(
                "S3 upload refused (oversize) for '%s': body %d bytes "
                "exceeds max_file_size_mb=%d",
                _safe_path_repr(path),
                len(content),
                max_file_size_mb,
            )
            responses.append(
                FileUploadResponse(path=path, error=OVERSIZE_ERROR_TAG)  # type: ignore[arg-type]
            )
            continue
        try:
            client.put_object(Bucket=bucket, Key=key, Body=content)
            responses.append(FileUploadResponse(path=path, error=None))
        except ClientError as exc:
            log_upload_download_error("upload", path, exc)
            responses.append(FileUploadResponse(path=path, error="permission_denied"))
    return responses


def download_files(
    paths: list[str],
    *,
    download_one: Callable[[str], FileDownloadResponse],
    download_concurrency: int,
    max_pool_connections: int,
) -> list[FileDownloadResponse]:
    """Download multiple files, bounded by the boto3 connection pool.

    A single-path request is run inline to skip executor overhead.

    Each ``download_one`` call is wrapped so an unexpected exception
    (e.g. malformed S3 response, ``KeyError`` from a stub) is converted
    into a per-path ``permission_denied`` response rather than
    propagating across the ``BackendProtocol`` boundary and aborting the
    whole batch.

    Results are collected via :func:`concurrent.futures.as_completed`
    rather than ``pool.map`` so a single slow download (or one waiting
    on a hung connection up to ``read_timeout``) cannot serialize the
    whole batch behind it. Per-path order is preserved in the returned
    list by re-indexing on the input ``paths`` so callers still see a
    stable mapping ``paths[i] -> result[i]``.

    Shutdown semantics: the ``ThreadPoolExecutor`` is entered with
    ``with``, so leaving the block waits for every submitted future to
    finish (``shutdown(wait=True)``). A :class:`KeyboardInterrupt`
    raised in the main thread therefore *does not* immediately tear
    down in-flight ``GetObject`` calls — it surfaces only after each
    worker returns or its socket trips ``read_timeout``. We accept this
    delay rather than calling ``shutdown(wait=False, cancel_futures=True)``
    because boto3's streaming body is not safe to abandon mid-read
    (the connection would be discarded instead of returned to the
    pool, and the next call would re-establish TLS), and the per-call
    ``read_timeout`` already bounds the worst-case wait. Callers that
    need hard interrupt semantics should set ``download_concurrency=1``
    so the loop runs inline on the main thread.

    ``KeyboardInterrupt`` and :class:`asyncio.CancelledError` raised
    from a worker future are teardown signals, not per-path errors, and
    must not be flattened into ``permission_denied``: callers (sync
    Ctrl-C or an outer asyncio cancel scope) must see them so they can
    stop submitting more work. Both are collected and re-raised after
    the executor finishes draining; the first observed wins. On
    Python 3.11 (the package's minimum) :class:`asyncio.CancelledError`
    inherits directly from :class:`BaseException` while
    :class:`concurrent.futures.CancelledError` inherits from
    :class:`Exception`; they are distinct classes, so the teardown
    ``except`` clause names both explicitly. ``SystemExit`` is allowed
    to propagate naturally (no clause catches it) so process-exit
    signals are not delayed past the draining loop.
    """
    if not paths:
        return []
    configured = max(1, download_concurrency)
    max_workers = min(configured, max_pool_connections, len(paths))
    safe_download = _make_safe_download(download_one)
    if max_workers <= 1:
        return [safe_download(path) for path in paths]
    # ``as_completed`` yields futures in completion order, not input
    # order, so we collect into a pre-sized list keyed by the input
    # index. Slots are initialized to ``None`` and filled in place;
    # any ``None`` remaining at the end means a future regression
    # skipped an index, which we surface loudly instead of returning
    # a short or partially-empty list. The list is local to this
    # function and never observed by callers — they only see the
    # post-validation typed list — so this stays consistent with the
    # codebase's immutable-result convention while preserving the
    # ``paths[i] -> result[i]`` mapping.
    results: list[FileDownloadResponse | None] = [None] * len(paths)
    pending_teardown: BaseException | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {
            pool.submit(safe_download, path): idx for idx, path in enumerate(paths)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except (
                KeyboardInterrupt,
                AsyncioCancelledError,
                FuturesCancelledError,
            ) as exc:
                # User-driven Ctrl-C or an outer asyncio cancel scope
                # surfacing through a worker future. We do NOT flatten
                # these into ``permission_denied`` because they
                # represent teardown signals, not per-path errors:
                # callers must see the signal so they can stop
                # submitting more work. We still drain the executor
                # (the ``with`` block calls ``shutdown(wait=True)``) so
                # in-flight fetches return their connections to the
                # pool, then re-raise once every slot has been
                # resolved. The first observed teardown wins;
                # subsequent teardown signals from other futures are
                # discarded (Python's implicit ``__context__`` chaining
                # only links exceptions raised in the same frame, not
                # results pulled from sibling ``Future`` objects), so we
                # log them at WARNING below to keep the loss visible.
                if pending_teardown is None:
                    pending_teardown = exc
                else:
                    logger.warning(
                        "download_files received an additional teardown "
                        "signal (%s) while one was already pending; only "
                        "the first will be re-raised",
                        type(exc).__name__,
                    )
                results[idx] = FileDownloadResponse(
                    path=paths[idx], content=None, error="permission_denied"
                )
            except Exception as exc:  # noqa: BLE001
                # Only ``Exception`` subclasses are flattened into a
                # ``permission_denied`` slot. The prior ``except`` already
                # handled ``KeyboardInterrupt``/``CancelledError``, and
                # ``SystemExit``/``GeneratorExit`` plus framework-control
                # bases such as ``pytest.outcomes.Skipped``/``Failed``
                # would have inherited from ``BaseException`` directly —
                # they must propagate so the user-driven (or harness-
                # driven) teardown semantics survive. The ``noqa`` is for
                # the broad ``Exception`` catch, which is intentional:
                # ``safe_download`` already classifies expected failures
                # (``ClientError``/``OversizeError``) and this branch is
                # the last-resort net for programming errors in the
                # download path.
                logger.error(
                    "S3 download for '%s' raised %s; surfacing as "
                    "permission_denied to keep batch shape stable",
                    _safe_path_repr(paths[idx]),
                    type(exc).__name__,
                )
                results[idx] = FileDownloadResponse(
                    path=paths[idx], content=None, error="permission_denied"
                )
    # Every slot SHOULD be filled at this point: ``safe_download``
    # covers ``Exception``, the ``BaseException`` guard above covers
    # ``KeyboardInterrupt``/``CancelledError``, and the surrounding
    # ``with ThreadPoolExecutor`` block awaits every submitted future
    # before returning, so ``as_completed`` is guaranteed to visit each
    # slot. The fill-in below is therefore defensive only — kept against
    # a future regression that skips an index (e.g. a code change that
    # narrows the except clauses without updating the slot assignment)
    # so the public ``BackendProtocol`` contract
    # (``len(result) == len(paths)`` with a stable ``paths[i] -> result[i]``
    # mapping) cannot be silently violated. ``logger.debug`` keeps the
    # path observable in development without alarming operators on a
    # path that is unreachable in the current code shape.
    missing_indexes = [i for i, r in enumerate(results) if r is None]
    if missing_indexes:
        logger.debug(
            "download_files left %d unfilled slot(s) for paths %s; "
            "filling with permission_denied to preserve batch shape",
            len(missing_indexes),
            [_fingerprint_path(paths[i]) for i in missing_indexes[:5]],
        )
        for idx in missing_indexes:
            results[idx] = FileDownloadResponse(
                path=paths[idx], content=None, error="permission_denied"
            )
    # ``cast`` would suffice but a comprehension keeps the type narrow
    # without a typing import — every slot is non-None per the fill-in
    # above, so ``r is not None`` is total.
    final = [r for r in results if r is not None]
    if pending_teardown is not None:
        # Slots are filled (so a debugger on the re-raise can still
        # inspect partial progress via the locals), but the teardown
        # signal itself is propagated so callers stop submitting more
        # work. ``from None`` suppresses implicit chaining to any
        # exception the ``with ThreadPoolExecutor`` exit path observed:
        # the teardown traceback is what callers (Ctrl-C, an outer
        # asyncio cancel scope) need to act on, and a bystander
        # ``__context__`` would only add noise to the surfaced trace.
        # The original traceback on ``pending_teardown`` is preserved.
        raise pending_teardown from None
    return final


def _make_safe_download(
    download_one: Callable[[str], FileDownloadResponse],
) -> Callable[[str], FileDownloadResponse]:
    """Wrap ``download_one`` so unexpected exceptions become responses.

    ``download_one`` already classifies ``ClientError``/``OversizeError``;
    this guard catches anything else (malformed dicts, programming
    errors in a stub) so a single bad object cannot tear down the
    parallel batch.
    """

    def safe(path: str) -> FileDownloadResponse:
        try:
            return download_one(path)
        except Exception as exc:  # noqa: BLE001
            # ``BaseException`` (KeyboardInterrupt, SystemExit,
            # CancelledError) is left to propagate; we only flatten
            # ``Exception`` and below. The catch is intentionally broad
            # because ``download_one`` already classifies ``ClientError``
            # and ``OversizeError`` itself, so anything reaching here is
            # malformed S3 response shape, a misbehaving stub, a
            # transport-layer failure, or any unexpected ``RuntimeError``
            # from a callback — none of which should tear down the rest
            # of the parallel batch. Narrowing to a specific tuple was
            # considered (KeyError/ValueError/TypeError/AttributeError/
            # OSError) but the contract that callers rely on — "every
            # path resolves to a per-path response, never an exception
            # across the protocol boundary" — requires the broad form.
            # The real cause is captured by ``logger.exception`` below
            # at ERROR level so triage still surfaces it; the public
            # response is collapsed to ``permission_denied`` because
            # deepagents' ``FileOperationError`` Literal has no broader
            # bucket.
            # Trim the path before logging so a pathologically long
            # caller-supplied path (e.g. several KiB) cannot bloat the
            # log line — we only need enough context to triage.
            # ``_safe_path_repr`` mirrors the helper used by
            # ``download_files`` so a non-``str`` entry slipped past the
            # protocol boundary (e.g. ``Path`` / ``bytes``) cannot raise
            # ``TypeError`` inside this exception handler and mask the
            # original failure.
            logger.exception(
                "S3 download for '%s' raised unexpectedly (%s); "
                "surfacing as permission_denied to keep the batch running",
                _safe_path_repr(path),
                type(exc).__name__,
            )
            return FileDownloadResponse(
                path=path, content=None, error="permission_denied"
            )

    return safe


def download_one(
    path: str,
    *,
    path_to_file_key: Callable[[str], str],
    read_capped: Callable[[str], CappedReadResult],
    max_file_size_mb: int,
) -> FileDownloadResponse:
    """Download a single file with the configured size cap."""
    try:
        key = path_to_file_key(path)
    except ValueError:
        return FileDownloadResponse(path=path, content=None, error="invalid_path")
    try:
        fetched = read_capped(key)
    except ClientError as exc:
        if is_not_found(exc):
            return FileDownloadResponse(path=path, content=None, error="file_not_found")
        log_upload_download_error("download", path, exc)
        return FileDownloadResponse(path=path, content=None, error="permission_denied")
    except OversizeError as exc:
        logger.error(
            "S3 download refused (oversize) for '%s': object exceeds "
            "max_file_size_mb=%d (content_length=%s)",
            _safe_path_repr(path),
            max_file_size_mb,
            exc.content_length,
        )
        return FileDownloadResponse(
            path=path,
            content=None,
            error=OVERSIZE_ERROR_TAG,  # type: ignore[arg-type]
        )
    return FileDownloadResponse(path=path, content=fetched.raw_bytes, error=None)
