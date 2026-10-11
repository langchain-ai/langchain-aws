"""Internal helpers for :mod:`langchain_backend_aws.s3.backend`.

Splitting these out keeps ``backend.py`` focused on the
:class:`BackendProtocol` surface and lets the lower-level concerns
(error classification, glob compilation, capped object reads) be
unit-tested independently.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Final, NamedTuple

import regex as regex_mod
from botocore.exceptions import BotoCoreError, ClientError

# Set of characters whose count we treat as a "complexity" measure for
# user-supplied glob and grep patterns. Counting them gives an
# input-shape cap orthogonal to the source-length cap so a short pattern
# cannot stack ``(?:(?:...)+)+`` past the metachar bound.
GREP_METACHARS = frozenset("()[]{}*+?|\\")

# Glob is a much smaller language than regex: only ``*`` and ``?`` are
# wildcards; everything else is escaped through ``re.escape`` in
# :func:`_compile_glob_regex_uncached`. Bounding the count of these
# wildcards bounds the number of stacked ``[^/]*`` runs that the
# translated regex can expand into, which is the catastrophic-backtracking
# surface for glob.
GLOB_METACHARS = frozenset("*?")


def count_metachars(pattern: str, metachars: frozenset[str]) -> int:
    """Count occurrences of ``metachars`` in ``pattern``."""
    return sum(1 for ch in pattern if ch in metachars)


logger = logging.getLogger(__name__)

# Codes returned by AWS S3 / S3-compatible stores when the caller lacks
# permission to perform the operation.
_ACCESS_DENIED_CODES = frozenset(
    {"AccessDenied", "AllAccessDisabled", "403", "SignatureDoesNotMatch"}
)

# Codes returned by S3 when a conditional ``PutObject`` (with
# ``IfNoneMatch`` / ``IfMatch``) is rejected because the precondition
# evaluated to false. AWS returns ``PreconditionFailed`` (HTTP 412); some
# S3-compatible stores additionally surface the raw status as ``"412"``.
_PRECONDITION_FAILED_CODES = frozenset({"PreconditionFailed", "412"})

# Codes representing transient/5xx conditions where retry is the right
# response (vs. permission errors where retrying is futile). Used to
# select log severity for upload/download errors that the public
# ``FileOperationError`` Literal cannot represent.
_TRANSIENT_CODES = frozenset(
    {
        "SlowDown",
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "ServiceUnavailable",
        "InternalError",
        "500",
        "503",
    }
)

_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey"})

# Maximum length for caller-supplied strings (object keys, glob
# patterns, paths) interpolated into log messages. Truncating bounds
# the size of any single log line so a pathological input cannot blow
# up downstream log shipping. Centralised here so ``_grep`` and
# ``_glob`` cannot drift out of step on the cap.
LOG_TRIM: Final[int] = 200


def normalize_listing_size(raw_size: Any) -> int:
    """Normalize an S3 ``Size`` field to a non-negative int.

    Defends listing consumers against a hostile or buggy S3-compatible
    store that returns negative or non-numeric ``Size`` values: a
    negative value would otherwise bypass the cheap ``obj_size >
    max_size`` skip check in the ``grep`` / ``glob`` / ``ls`` paginator
    paths. Non-numeric values raise ``ValueError`` so callers fail
    closed via the surrounding broad-exception handler rather than
    surfacing as a silent "no match".
    """
    size = int(raw_size if raw_size is not None else 0)
    return max(size, 0)


def is_precondition_failed(exc: ClientError) -> bool:
    """Return ``True`` if a ClientError signals a conditional-write rejection."""
    code = exc.response.get("Error", {}).get("Code", "")
    return code in _PRECONDITION_FAILED_CODES


def is_not_found(exc: ClientError) -> bool:
    """Return ``True`` if a ClientError signals a missing object."""
    code = exc.response.get("Error", {}).get("Code", "")
    return code in _NOT_FOUND_CODES


def sanitize_error_code(exc: ClientError) -> str:
    """Extract a short error code from a ClientError without leaking AWS details."""
    return exc.response.get("Error", {}).get("Code", "Unknown")


def log_upload_download_error(operation: str, path: str, exc: ClientError) -> None:
    """Log an upload/download failure with severity matched to its cause.

    The :class:`BackendProtocol` constrains ``FileOperationError`` to a
    fixed ``Literal`` so we cannot distinguish transient or unknown
    failures in the response itself. We do distinguish them in the log
    so operators triaging an incident still see the real shape:

    - WARNING for permission denials — usually caller misconfiguration.
    - ERROR for transient/5xx conditions (``SlowDown``, ``InternalError``)
      so they bubble into alerting even though the user-facing tag is
      still ``permission_denied``.
    - ERROR for unknown codes — better to over-alert than to bury an
      unfamiliar failure mode at WARNING level.
    """
    code = sanitize_error_code(exc)
    if code in _ACCESS_DENIED_CODES:
        logger.warning("S3 %s denied for '%s': %s", operation, path, code)
    elif code in _TRANSIENT_CODES:
        logger.error(
            "S3 %s failed transiently for '%s' with code %s; surfaced as "
            "permission_denied because BackendProtocol has no transient tag",
            operation,
            path,
            code,
        )
    else:
        logger.error(
            "S3 %s failed for '%s' with code %s; surfaced as permission_denied",
            operation,
            path,
            code,
        )


def _compile_glob_regex_uncached(pattern: str) -> regex_mod.Pattern[str]:
    """Translate a glob pattern to a compiled anchored regex (no cache).

    The pure-function core wrapped by :func:`make_glob_compiler` — kept
    separate so per-instance caches reuse the translation logic without
    pulling in cache state.

    Supported syntax:

    - ``**/`` matches zero or more path segments (recursive). Only the
      ``**/`` form is treated specially. ``**`` *not* followed by ``/``
      collapses to ``[^/]*`` — identical to a single ``*`` — so it never
      crosses ``/`` boundaries. (Earlier revisions translated bare
      ``**`` to ``.*``; that was changed because shell glob semantics
      treat ``a**b`` as a single-segment match and the cross-segment
      form was a footgun for prefix-violation accounting.) Prefer
      ``**/`` for recursion.
    - ``*`` matches any number of characters except ``/``.
    - ``?`` matches a single character except ``/``.
    - All other characters are matched literally.

    The pattern is anchored at both ends so ``foo/*.py`` only matches
    paths exactly two segments deep.

    Compiled with the third-party :mod:`regex` engine so callers can
    pass ``timeout=`` to :py:meth:`regex.Pattern.match`. The stdlib
    :mod:`re` engine has no interruption hook, so a stacked-quantifier
    pattern like ``[^/]*[^/]*[^/]*...`` (translated from ``****…``)
    would otherwise pin a worker on catastrophic backtracking. ``regex``
    accepts the standard ``re`` syntax we emit, so this is a drop-in.
    """
    i = 0
    out: list[str] = ["^"]
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    # ``**`` not followed by ``/`` is treated as a single
                    # ``*`` so cross-segment matching only happens when
                    # the user explicitly writes ``**/``. See docstring.
                    out.append("[^/]*")
            else:
                out.append("[^/]*")
                i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    out.append("$")
    return regex_mod.compile("".join(out))


# Default LRU sizing for per-instance caches built by
# :func:`make_glob_compiler`. 256 covers a realistic working-set of
# distinct globs while keeping the footprint small —
# ``S3BackendConfig.glob_max_pattern_length`` already bounds per-entry
# size, so an attacker cannot inflate the worst-case memory cost.
_GLOB_CACHE_MAXSIZE = 256


def make_glob_compiler(
    maxsize: int = _GLOB_CACHE_MAXSIZE,
) -> Callable[[str], regex_mod.Pattern[str]]:
    """Build a per-instance LRU-cached glob translator.

    Returns a callable that wraps :func:`_compile_glob_regex_uncached`
    in its own :func:`functools.lru_cache`. Each :class:`S3Backend`
    instance owns one of these so a tenant cycling through many
    distinct one-off patterns cannot evict another tenant's hot
    pattern from a process-global cache.

    The translation itself is a pure function of the input pattern, so
    the only reason to scope the cache per-instance is tenant
    isolation. There is no module-level fallback: any caller that
    needs a translator must obtain one through this factory so the
    cache surface always carries a tenant boundary.
    """
    return lru_cache(maxsize=maxsize)(_compile_glob_regex_uncached)


class OversizeError(Exception):
    """Raised by :func:`read_capped_object` when an object exceeds the cap.

    Derived directly from :class:`Exception` (rather than ``OSError`` or
    a botocore class) on purpose: callers classify it side-by-side with
    :class:`botocore.exceptions.ClientError` and must never catch one
    while reaching for the other. Keeping the hierarchies disjoint makes
    the ``except ClientError`` / ``except OversizeError`` split in
    ``_io.py``/``_read.py``/``_write.py`` unambiguous.

    ``content_length`` carries the value reported by S3's
    ``ContentLength`` header when the cap was hit pre-read; it is
    ``None`` when the cap was hit by the body itself growing past the
    header (a lying or stale ``ContentLength``).
    """

    def __init__(self, content_length: int | None) -> None:
        # Include ``content_length`` in the exception message so a stray
        # ``str(exc)`` / traceback surface (e.g. an unexpected re-raise
        # in callers that bypass :func:`format_oversize_message`) still
        # reports the observed size rather than an opaque generic. The
        # user-facing wording lives in :func:`format_oversize_message`;
        # this is the internal/diagnostic representation.
        if content_length is None:
            super().__init__("object exceeds size cap (read past header limit)")
        else:
            super().__init__(f"object exceeds size cap ({content_length} bytes)")
        self.content_length = content_length


class CappedReadResult(NamedTuple):
    """Result of a successful capped object read."""

    raw_bytes: bytes
    etag: str
    last_modified: datetime


def read_capped_object(
    client: Any,
    bucket: str,
    key: str,
    max_bytes: int,
) -> CappedReadResult:
    """Fetch an S3 object body, refusing reads that exceed ``max_bytes``.

    The cap is enforced twice — once against the ``ContentLength``
    reported by ``GetObject`` and once against the bytes actually pulled
    from the body — so neither a lying header nor an object that grew
    between list and get can bypass it.

    Memory profile: a successful read materialises up to ``max_bytes``
    into a single :class:`bytes` buffer via one ``body.read(max_bytes
    + 1)`` call. The cap itself is the only bound — operators who
    raise ``max_file_size_mb`` are responsible for sizing it against
    available memory, since the read is intentionally not chunked
    (chunking would defeat the second-pass length check that catches
    a lying ``ContentLength`` header). Callers that fan this out
    (``download_files``) should size ``download_concurrency`` and
    ``max_file_size_mb`` together — the backend's resident-memory
    high-water mark is roughly ``download_concurrency * max_file_size``.
    See the README "Memory profile" section.

    Args:
        client: boto3 S3 client.
        bucket: Bucket name.
        key: Object key.
        max_bytes: Maximum number of bytes the caller is willing to load.

    Returns:
        :class:`CappedReadResult` containing the body bytes, the
        ``ETag`` returned by S3 (empty string if absent), and the
        ``LastModified`` timestamp.

    Raises:
        botocore.exceptions.ClientError: Bubbled unchanged so the caller
            can apply its own ``not-found``/``denied`` classification.
        OversizeError: When the read would exceed ``max_bytes``. The
            streaming body is closed before raising so the connection
            can return to the pool.
    """
    response = client.get_object(Bucket=bucket, Key=key)
    # Treat a missing ``Body`` as oversize/fail-closed rather than a
    # ``KeyError`` ricocheting through callers. Some S3-compatible
    # stores (older MinIO builds, partial mocks) can return a malformed
    # response shape; surfacing it as :class:`OversizeError` keeps the
    # caller's classifier symmetrical with a genuinely-too-large object
    # and avoids a noisy "unexpected error" in upstream logs.
    try:
        body = response["Body"]
    except KeyError as exc:
        raise OversizeError(None) from exc
    try:
        # Clamp to ``>= 0`` so a malicious S3-compatible server returning
        # ``ContentLength: -1`` cannot bypass the ``> max_bytes`` guard
        # and slip into the post-read length check with a negative value
        # leaking into the user-facing oversize message. A non-numeric
        # ``ContentLength`` (also possible from a hostile S3-compatible
        # server) would otherwise raise ``TypeError``/``ValueError``
        # uncategorized — fail closed as oversize so the caller sees
        # the same surface as a genuinely-too-large object.
        try:
            content_length = max(0, int(response.get("ContentLength", 0)))
        except (TypeError, ValueError) as exc:
            raise OversizeError(None) from exc
        if content_length > max_bytes:
            raise OversizeError(content_length)
        raw_bytes = body.read(max_bytes + 1)
        # Defense-in-depth: a misbehaving S3-compatible store (or a
        # custom streaming wrapper) may return ``str`` or another type
        # from ``read``. Reject anything that is not ``bytes`` rather
        # than letting downstream regex / decode raise an opaque
        # ``TypeError`` deep in the call stack.
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            msg = (
                f"S3 response body returned {type(raw_bytes).__name__}; "
                f"expected bytes-like object"
            )
            raise TypeError(msg)
        raw_bytes = bytes(raw_bytes)
        # Best-effort EOF confirmation when the read returned exactly
        # ``max_bytes`` bytes: ``StreamingBody.read(n)`` is best-effort
        # and may return fewer bytes than requested when the underlying
        # socket short-reads, so the post-read ``> max_bytes`` length
        # check below could pass for a body that is *exactly* at the cap
        # while more bytes are still pending. Confirming EOF with one
        # extra ``read(1)`` closes that gap. Any exception during this
        # probe is swallowed — we already have a usable body and the
        # ``ContentLength`` header check is the primary defence; this is
        # purely defence-in-depth against a lying header paired with a
        # short-read.
        if len(raw_bytes) == max_bytes:
            # Mirror the narrow tuple used by the ``close()`` cleanup
            # path below: a broken socket / half-closed TLS surfaces as
            # ``OSError``, while ``botocore`` wraps streaming faults as
            # ``BotoCoreError`` (e.g. ``ResponseStreamingError``).
            # Anything outside that set — ``BaseException`` such as
            # ``KeyboardInterrupt``/``SystemExit``, or a genuinely
            # unexpected ``Exception`` from a stub — propagates rather
            # than being silently absorbed into a "no tail" decision.
            try:
                tail = body.read(1)
            except (OSError, BotoCoreError):
                tail = b""
            if tail:
                raise OversizeError(None)
    finally:
        # Close the body for every exit path (oversize, type error on
        # ContentLength, transport failure mid-read) so the underlying
        # connection is returned to the pool. ``close()`` itself can
        # raise (broken socket, half-closed TLS); swallow those into a
        # warning so we never mask the original exception that drove us
        # into ``finally`` — the original cause is what callers need to
        # triage, and a leaked connection just rotates through the pool.
        close = getattr(body, "close", None)
        if close is not None:
            try:
                close()
            except (OSError, AttributeError, BotoCoreError) as close_exc:
                # Cleanup-path failures (broken socket / half-closed
                # TLS as ``OSError``; missing ``close`` slot on a stub
                # as ``AttributeError``; ``botocore`` surfacing
                # ``ResponseStreamingError`` — a ``BotoCoreError``
                # subclass — from a half-read body) are demoted to a
                # warning so the original exception raised from the
                # ``try`` block — the one callers classify
                # (``ClientError``/``OversizeError``) — is never masked
                # by a secondary failure on cleanup. The leaked
                # connection just rotates through the pool;
                # ``BaseException`` (``KeyboardInterrupt``,
                # ``SystemExit``) is still allowed to propagate, and a
                # truly unexpected ``Exception`` from a stub will
                # propagate too rather than be silenced here.
                logger.warning(
                    "S3 read body close failed for '%s/%s': %s",
                    bucket[:LOG_TRIM],
                    key[:LOG_TRIM],
                    close_exc,
                )
    if len(raw_bytes) > max_bytes:
        raise OversizeError(None)
    etag = response.get("ETag") or ""
    # Same fail-closed rationale as the missing-``Body`` branch above:
    # a malformed S3-compatible response without ``LastModified`` should
    # not propagate as ``KeyError`` — callers classify ``OversizeError``
    # alongside ``ClientError`` and rely on the disjoint hierarchy.
    try:
        last_modified: datetime = response["LastModified"]
    except KeyError as exc:
        raise OversizeError(None) from exc
    return CappedReadResult(raw_bytes, etag, last_modified)


def build_glob_regexes(
    glob: str | None,
    compile_glob: Callable[[str], regex_mod.Pattern[str]],
) -> tuple[regex_mod.Pattern[str] | None, regex_mod.Pattern[str] | None]:
    """Compile an optional glob filter for grep/glob.

    Returns ``(path_regex, basename_regex)``. The basename regex is
    only set when ``glob`` has no ``/`` so shell-like patterns such as
    ``*.py`` still match files at any depth.

    Leading ``/`` is stripped here so the ``"/"`` membership check that
    selects basename mode operates on the same shape that ``compile_glob``
    receives. Callers that also enforce length / metachar caps on the
    pattern (e.g. :meth:`S3Backend.glob_search`) must strip the leading
    slash *before* counting so the cap matches the compiled form;
    :meth:`S3Backend.grep` only forwards an opaque filter and lets this
    helper own the normalization.

    Args:
        glob: The optional glob filter string.
        compile_glob: Required translator callable. Callers must pass a
            per-context compiler (typically from :func:`make_glob_compiler`)
            so a process-global cache cannot bridge tenants.
    """
    if not glob:
        return None, None
    pattern = glob.lstrip("/")
    path_regex = compile_glob(pattern)
    basename_regex = path_regex if "/" not in pattern else None
    return path_regex, basename_regex


def glob_matches(
    relative: str,
    path_regex: regex_mod.Pattern[str] | None,
    basename_regex: regex_mod.Pattern[str] | None,
    *,
    timeout: float,
) -> bool:
    """Apply a glob filter to a relative path with basename fallback.

    Each :py:meth:`regex.Pattern.match` call carries a per-attempt
    ``timeout`` so a crafted pattern that triggered catastrophic
    backtracking on a single object cannot pin a worker indefinitely.
    The :class:`TimeoutError` raised by :mod:`regex` propagates to the
    caller, which fails closed.
    """
    if path_regex is None:
        return True
    if path_regex.match(relative, timeout=timeout):
        return True
    if basename_regex is None:
        return False
    basename = relative.rsplit("/", 1)[-1]
    return bool(basename_regex.match(basename, timeout=timeout))


# Backend-specific error tag returned via ``FileUploadResponse.error``
# and ``FileDownloadResponse.error`` when a body exceeds the configured
# size cap. The :data:`deepagents.backends.protocol.FileOperationError`
# Literal has no ``oversize`` member, but the response docstrings
# explicitly sanction backend-specific error strings for failures that
# cannot be normalized into the Literal:
#
#     > A ``FileOperationError`` literal for known conditions, or a
#     > backend-specific error string when the failure cannot be
#     > normalized.
#
# Surfacing ``"oversize"`` directly lets callers (and the LLM looking
# at the error string in the agent prompt) see the real reason instead
# of the misleading ``"permission_denied"`` flattening. Typed ``Any``
# because the runtime value is a string the Literal does not contain;
# using :func:`typing.cast` to ``FileOperationError`` would be a lie.
# Callsites assign this to ``FileUploadResponse.error`` /
# ``FileDownloadResponse.error``, both documented as accepting an
# open-ended string.
# ``Final[str]`` keeps the constant immutable and statically typed; the
# narrow ``# type: ignore[assignment]`` lives at the assignment sites
# that store this into ``FileUploadResponse.error`` /
# ``FileDownloadResponse.error`` (both Literal-typed but documented as
# accepting open-ended strings).
OVERSIZE_ERROR_TAG: Final[str] = "oversize"


def strip_line_terminator(raw_line: str) -> str:
    """Strip a single trailing ``\\r?\\n`` to match :py:meth:`str.splitlines`.

    Shared by :mod:`._read` and :mod:`._grep`, both of which iterate
    a body via :class:`io.StringIO` (which preserves terminators on
    each yielded line) but want match/output text without them.
    """
    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith("\n"):
        return raw_line[:-1]
    return raw_line


def format_oversize_message(file_path: str, max_mb: int, exc: OversizeError) -> str:
    """Format a human-readable oversize error string.

    Centralises wording so ``read``/``edit``/``download_files`` produce
    identical messages for the same condition.
    """
    if exc.content_length is not None:
        return (
            f"File '{file_path}' is {exc.content_length} bytes which exceeds "
            f"max_file_size_mb={max_mb}."
        )
    return (
        f"File '{file_path}' exceeds max_file_size_mb={max_mb} "
        "(read past header limit)."
    )
