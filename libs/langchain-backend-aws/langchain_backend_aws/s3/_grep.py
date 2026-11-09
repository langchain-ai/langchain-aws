"""``grep`` enumeration and per-object scanning for :class:`S3Backend`.

Extracted from ``backend.py`` to keep the protocol surface compact;
behavior is unchanged from the inlined implementation.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Iterator, NamedTuple

import regex as regex_mod
from botocore.exceptions import ClientError
from deepagents.backends.protocol import GrepMatch

from langchain_backend_aws.s3._internal import (
    GLOB_METACHARS,
    GREP_METACHARS,
    LOG_TRIM,
    OVERSIZE_ERROR_TAG,
    OversizeError,
    build_glob_regexes,
    count_metachars,
    glob_matches,
    normalize_listing_size,
    read_capped_object,
    sanitize_error_code,
    strip_line_terminator,
)

logger = logging.getLogger(__name__)


def validate_grep_inputs(
    pattern: str,
    glob: str | None,
    *,
    grep_max_pattern_length: int,
    grep_max_pattern_metachars: int,
    glob_max_pattern_length: int,
    glob_max_pattern_metachars: int,
) -> str | None:
    """Validate grep ``pattern``/``glob`` length and shape caps.

    Returns an error string for the caller to surface via
    ``GrepResult.error``, or ``None`` when both inputs satisfy the
    configured caps. Kept separate so :meth:`S3Backend.grep` stays under
    the 50-line guideline and the ReDoS shape-cap reasoning lives in
    one place.
    """
    if len(pattern) > grep_max_pattern_length:
        return (
            f"Regex pattern length {len(pattern)} exceeds "
            f"grep_max_pattern_length={grep_max_pattern_length}."
        )
    # Length alone does not bound nested-quantifier compile cost: a
    # short pattern can still stack ``(?:(?:(?:...)+)+)+``. Counting
    # the metacharacters that drive backtracking gives an
    # input-shape cap orthogonal to the source-length cap.
    metachar_count = count_metachars(pattern, GREP_METACHARS)
    if metachar_count > grep_max_pattern_metachars:
        return (
            f"Regex pattern metacharacter count {metachar_count} "
            f"exceeds grep_max_pattern_metachars={grep_max_pattern_metachars}."
        )
    if glob is None:
        return None
    if len(glob) > glob_max_pattern_length:
        return (
            f"Glob filter length {len(glob)} exceeds "
            f"glob_max_pattern_length={glob_max_pattern_length}."
        )
    # Same shape-cap as the top-level ``glob()`` call: count the
    # wildcards that drive stacked-quantifier backtracking in
    # the translated regex. Without this a per-object glob filter
    # could be DoS'd with a ``****…**.py`` even when
    # ``grep_regex_timeout`` keeps the body search bounded.
    glob_metachars = count_metachars(glob, GLOB_METACHARS)
    if glob_metachars > glob_max_pattern_metachars:
        return (
            f"Glob filter wildcard count {glob_metachars} "
            f"exceeds glob_max_pattern_metachars={glob_max_pattern_metachars}."
        )
    return None


def classify_grep_result(
    paginate_result: GrepPaginateResult,
    search_path: str,
    *,
    grep_max_objects: int,
) -> tuple[list[GrepMatch] | None, str | None]:
    """Map a :class:`GrepPaginateResult` to ``(matches, error)``.

    Three outcomes share this path:

    - A per-object fetch fail-closed message bubbled up through
      ``paginate_result.fail_closed_error``.
    - The paginator hit ``grep_max_objects`` (``truncated=True``);
      mirrors the glob/ls contract — partial matches as success
      would let callers conclude a pattern is absent when the scan
      cap merely cut short.
    - Success: matches are sorted by ``(path, line)`` for a
      deterministic order callers can diff against.

    Returns ``(None, error)`` for fail-closed paths and
    ``(matches, None)`` on success.
    """
    if paginate_result.fail_closed_error is not None:
        return None, (
            f"Error during grep at '{search_path}': {paginate_result.fail_closed_error}"
        )
    if paginate_result.truncated:
        return None, (
            f"grep scan exceeded grep_max_objects={grep_max_objects} under "
            f"'{search_path}'. Narrow the path or raise the limit."
        )
    matches = sorted(
        paginate_result.matches,
        key=lambda m: (m.get("path", ""), m.get("line", 0)),
    )
    return matches, None


def format_grep_timeout_error(
    pattern: str,
    search_path: str,
    exc: TimeoutError,
    *,
    grep_regex_timeout: float,
    glob_regex_timeout: float,
) -> str:
    """Format the fail-closed message for a regex timeout in grep.

    ``regex.search``/``regex.match`` raised ``TimeoutError`` because a
    single match attempt exceeded the configured timeout
    (``grep_regex_timeout`` for the body search, ``glob_regex_timeout``
    for the per-object glob filter). A partial match list would let
    callers conclude the pattern is absent from objects we never
    finished scanning, so :meth:`grep` always fails closed on this path.
    """
    logger.error(
        "grep: regex match exceeded timeout for pattern "
        "(length=%d) at '%s' (grep_regex_timeout=%.3fs, "
        "glob_regex_timeout=%.3fs)",
        len(pattern),
        search_path,
        grep_regex_timeout,
        glob_regex_timeout,
    )
    return (
        f"Error during grep at '{search_path}': regex match "
        f"exceeded configured timeout "
        f"(grep_regex_timeout={grep_regex_timeout}s, "
        f"glob_regex_timeout={glob_regex_timeout}s; "
        f"catastrophic backtracking suspected): {exc}"
    )


class GrepPrepared(NamedTuple):
    """Successful output of :func:`prepare_grep`.

    ``compiled`` is the user-supplied regex compiled with the third-party
    :mod:`regex` engine (so callers can pass ``timeout=`` later);
    ``base_key`` is the S3 key prefix the paginator should scan, always
    ending with ``/`` for non-empty values; ``glob_regexes`` carries the
    optional ``(path_regex, basename_regex)`` filter.
    """

    compiled: regex_mod.Pattern[str]
    base_key: str
    glob_regexes: tuple[regex_mod.Pattern[str] | None, regex_mod.Pattern[str] | None]


def prepare_grep(
    pattern: str,
    glob: str | None,
    *,
    path_to_key: Callable[[str], str],
    compile_glob: Callable[[str], regex_mod.Pattern[str]],
    search_path: str,
) -> tuple[GrepPrepared | None, str | None]:
    """Compile ``pattern``/``glob`` and resolve ``search_path`` to a base key.

    Returns ``(prepared, None)`` on success or ``(None, error)`` when
    either compilation or path resolution failed. Centralising the three
    pre-paginate steps (regex compile, path-to-key resolution, glob
    compile) here keeps :meth:`S3Backend.grep` focused on validation
    dispatch and result classification — the same way
    :func:`validate_grep_inputs` and :func:`classify_grep_result`
    handle the bookend phases.
    """
    # ``regex`` accepts the standard ``re`` syntax used by callers; we
    # compile here rather than in the visit loop so per-search ``timeout=``
    # bounds remain the only mechanism that interrupts catastrophic
    # backtracking, not pattern compilation cost.
    try:
        compiled = regex_mod.compile(pattern)
    except regex_mod.error as exc:
        return None, f"Invalid regex pattern: {exc}"

    try:
        base_key = path_to_key(search_path)
    except ValueError as exc:
        return None, str(exc)
    if base_key and not base_key.endswith("/"):
        base_key += "/"

    glob_regexes = build_glob_regexes(glob, compile_glob=compile_glob)
    return GrepPrepared(compiled, base_key, glob_regexes), None


class GrepPaginateResult(NamedTuple):
    """Outcome of a paginated grep walk.

    ``truncated`` is set when ``max_objects`` was reached;
    ``fail_closed_error`` carries a sanitized message when a per-object
    fetch failed in a way that must not be downgraded to "no match".
    ``matches`` is a freshly-allocated list — callers extend their own
    accumulator from it rather than passing one in for the helper to
    mutate.
    """

    matches: list[GrepMatch]
    truncated: bool
    fail_closed_error: str | None


class _VisitOutcome(NamedTuple):
    """Per-object visit outcome used by :func:`_visit_object`."""

    matches: list[GrepMatch]
    fail_closed_error: str | None


class _FetchOutcome(NamedTuple):
    """Per-object fetch outcome used by :func:`_fetch_object`."""

    matches: list[GrepMatch]
    error: str | None


class _Truncated(Exception):
    """Internal signal raised by :func:`_iter_listed_objects` at the cap.

    Distinct exception class so :func:`grep_paginate` can distinguish a
    truncation from a fail-closed error without inspecting a flag — the
    classifier-by-exception form keeps the visit loop flat.
    """


def _iter_listed_objects(
    paginator: Any,
    bucket: str,
    base_key: str,
    *,
    max_objects: int,
) -> Iterator[dict[str, Any]]:
    """Yield ``Contents`` entries from a paginator, capped at ``max_objects``.

    Raises :class:`_Truncated` once the cap is reached so callers can
    distinguish "scan finished" from "scan cut short" without mixing a
    flag into the iteration loop.
    """
    scanned = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=base_key):
        for obj in page.get("Contents", []):
            if scanned >= max_objects:
                raise _Truncated
            scanned += 1
            yield obj


def grep_paginate(
    client: Any,
    bucket: str,
    prefix: str,
    base_key: str,
    regex: regex_mod.Pattern[str],
    glob_regexes: tuple[regex_mod.Pattern[str] | None, regex_mod.Pattern[str] | None],
    *,
    key_to_path: Callable[[str], str],
    max_objects: int,
    max_size: int,
    max_line_length: int,
    timeout: float,
    glob_timeout: float,
) -> GrepPaginateResult:
    """Walk listed objects under ``base_key`` and feed them to grep.

    Returns a :class:`GrepPaginateResult` carrying the (newly allocated)
    match list, a truncation flag, and an optional fail-closed error
    string. The helper does not mutate caller-supplied state.
    """
    matches: list[GrepMatch] = []
    paginator = client.get_paginator("list_objects_v2")
    listed = _iter_listed_objects(paginator, bucket, base_key, max_objects=max_objects)
    try:
        for obj in listed:
            outcome = _safe_visit(
                client,
                bucket,
                prefix,
                obj,
                base_key,
                regex,
                glob_regexes,
                key_to_path=key_to_path,
                max_size=max_size,
                max_line_length=max_line_length,
                timeout=timeout,
                glob_timeout=glob_timeout,
            )
            if outcome.fail_closed_error is not None:
                # Fail-closed before extending: today ``_visit_object``
                # returns ``matches=[]`` whenever it sets the error, but
                # a future change that allowed partial matches alongside
                # a fail-closed flag would otherwise silently leak them
                # into the result. Drop everything observed in this
                # outcome and surface only the error.
                return GrepPaginateResult(matches, False, outcome.fail_closed_error)
            matches.extend(outcome.matches)
    except _Truncated:
        return GrepPaginateResult(matches, True, None)
    return GrepPaginateResult(matches, False, None)


def _safe_visit(
    client: Any,
    bucket: str,
    prefix: str,
    obj: dict[str, Any],
    base_key: str,
    regex: regex_mod.Pattern[str],
    glob_regexes: tuple[regex_mod.Pattern[str] | None, regex_mod.Pattern[str] | None],
    *,
    key_to_path: Callable[[str], str],
    max_size: int,
    max_line_length: int,
    timeout: float,
    glob_timeout: float,
) -> _VisitOutcome:
    """Wrap :func:`_visit_object` so unexpected failures fail closed.

    ``ClientError`` and ``TimeoutError`` are re-raised because
    :meth:`grep` classifies them explicitly (sanitized code / ReDoS
    timeout). Anything else (malformed S3 response, stub bug, missing
    ``Key`` / ``Body`` / ``ETag``) is converted to a fail-closed
    outcome with a sanitized message, since silently propagating it
    across the protocol boundary would tear down the whole batch and
    surface to callers as "no match". ``BaseException``
    (KeyboardInterrupt, SystemExit) is intentionally left to propagate.
    """
    try:
        return _visit_object(
            client,
            bucket,
            prefix,
            obj,
            base_key,
            regex,
            glob_regexes,
            key_to_path=key_to_path,
            max_size=max_size,
            max_line_length=max_line_length,
            timeout=timeout,
            glob_timeout=glob_timeout,
        )
    except (ClientError, TimeoutError):
        raise
    except (KeyError, ValueError, TypeError, AttributeError, UnicodeDecodeError) as exc:
        # Narrow to the shapes a malformed S3-compatible response or a
        # mis-typed stub realistically produces:
        # - ``KeyError`` from ``obj["Key"]`` on a malformed listing,
        # - ``ValueError`` from ``int(obj["Size"])`` or path-key conversion,
        # - ``TypeError`` from a non-bytes body slipping past the
        #   defense-in-depth check (see :func:`read_capped_object`),
        # - ``AttributeError`` from a stub returning the wrong shape,
        # - ``UnicodeDecodeError`` is already handled in :func:`_fetch_object`,
        #   listed for completeness.
        # A genuine programming bug in our own code (anything outside
        # this set) is left to propagate so callers see the real failure
        # rather than a silenced "fail closed" mask.
        obj_key = obj.get("Key", "<unknown>") if isinstance(obj, dict) else "<unknown>"
        logger.exception(
            "grep: unexpected error while scanning '%s'; failing closed",
            obj_key[:LOG_TRIM],
        )
        return _VisitOutcome(
            [],
            f"unexpected error scanning '{obj_key}': {type(exc).__name__}",
        )


def _validate_key_containment(
    obj_key: str,
    base_key: str,
    prefix: str,
    key_to_path: Callable[[str], str],
) -> tuple[str, str] | _VisitOutcome:
    """Verify that ``obj_key`` is contained in the configured prefix.

    Validation runs BEFORE any glob/size filtering so a paginator that
    leaks keys outside the configured prefix cannot be silently masked
    by a non-matching glob. Returns ``(virtual_path, relative)`` on
    success, or a fail-closed :class:`_VisitOutcome` carrying the
    sanitized error message when either:

    - ``key_to_path`` rejects the key (storage prefix violation), or
    - ``obj_key`` does not start with the requested ``base_key``
      (paginator returned a key outside the requested path).
    """
    # Defence-in-depth: ``base_key`` must be either empty (root scope) or
    # end with ``/`` so the ``startswith`` check below cannot accept a
    # sibling key (``"foo/bar"`` matching ``"foo/b"``). ``prepare_grep``
    # enforces this shape; this guard surfaces a future regression even
    # under ``python -O`` (which strips ``assert`` statements).
    if base_key != "" and not base_key.endswith("/"):
        msg = f"base_key must be empty or end with '/', got {base_key!r}"
        raise AssertionError(msg)
    try:
        virtual_path = key_to_path(obj_key)
    except ValueError:
        # Trim before logging: a paginator-returned key in a hostile or
        # malformed S3-compatible store could be multi-KiB and bloat the
        # log line. Triage only needs the leading bytes.
        logger.error(
            "grep: storage prefix violation — S3 returned '%s' outside "
            "configured prefix '%s'; failing closed",
            obj_key[:LOG_TRIM],
            prefix[:LOG_TRIM],
        )
        return _VisitOutcome(
            [],
            f"storage prefix violation (returned key '{obj_key}' "
            f"outside configured prefix)",
        )
    if not obj_key.startswith(base_key):
        logger.error(
            "grep: paginator returned '%s' outside requested base '%s'; failing closed",
            obj_key[:LOG_TRIM],
            base_key[:LOG_TRIM],
        )
        return _VisitOutcome(
            [],
            f"storage prefix violation (returned key '{obj_key}' "
            f"outside requested path)",
        )
    relative = obj_key[len(base_key) :]
    return virtual_path, relative


def _visit_object(
    client: Any,
    bucket: str,
    prefix: str,
    obj: dict[str, Any],
    base_key: str,
    regex: regex_mod.Pattern[str],
    glob_regexes: tuple[regex_mod.Pattern[str] | None, regex_mod.Pattern[str] | None],
    *,
    key_to_path: Callable[[str], str],
    max_size: int,
    max_line_length: int,
    timeout: float,
    glob_timeout: float,
) -> _VisitOutcome:
    """Filter and search one listed object.

    Returns the per-object match list (possibly empty) and an optional
    fail-closed error string. When ``fail_closed_error`` is set the
    caller stops iterating and surfaces the message — same contract as
    the previous in-place version, just expressed as a return value.
    """
    obj_key: str = obj["Key"]
    # Normalize ``Size`` up-front so the rest of the function reads it
    # as a non-negative ``int``. A non-numeric ``Size`` from a malformed
    # S3-compatible store raises here and is caught by the broad
    # ``except Exception`` in :func:`grep_paginate`, which fail-closes
    # with a sanitized message rather than letting the error surface as
    # silent "no match"; a negative value is clamped to zero so the
    # cheap ``obj_size > max_size`` skip check below cannot be bypassed.
    obj_size = normalize_listing_size(obj.get("Size"))
    contained = _validate_key_containment(obj_key, base_key, prefix, key_to_path)
    if isinstance(contained, _VisitOutcome):
        return contained
    virtual_path, relative = contained
    if not relative:
        return _VisitOutcome([], None)
    glob_path_regex, glob_basename_regex = glob_regexes
    if not glob_matches(
        relative, glob_path_regex, glob_basename_regex, timeout=glob_timeout
    ):
        return _VisitOutcome([], None)
    if obj_size > max_size:
        return _VisitOutcome([], None)
    fetched = _fetch_object(
        client,
        bucket,
        obj_key,
        virtual_path,
        regex,
        max_size=max_size,
        max_line_length=max_line_length,
        timeout=timeout,
    )
    if fetched.error is None:
        return _VisitOutcome(fetched.matches, None)
    # Treat a list/read race (object deleted between ``ListObjectsV2``
    # and ``GetObject``) as a benign skip.
    if fetched.error in {"NoSuchKey", "404"}:
        return _VisitOutcome([], None)
    # Asymmetric oversize handling: the cheap ``Size`` filter at line
    # ``obj_size > max_size`` above silently skips objects whose listed
    # size already exceeds the cap (mirrors normal grep behavior on a
    # binary blob). Once :func:`read_capped_object` has actually fetched
    # bytes and discovered the object grew between list and get (or had
    # a missing/lying ``Size``), we MUST fail closed: a partial scan
    # result would let callers conclude the pattern is absent from
    # files we never finished reading. Surface the
    # :data:`OVERSIZE_ERROR_TAG` sentinel verbatim so the message
    # explicitly names the cause rather than the misleading
    # ``permission_denied`` flattening.
    if fetched.error == OVERSIZE_ERROR_TAG:
        return _VisitOutcome(
            [], f"failed to read '{virtual_path}': {OVERSIZE_ERROR_TAG}"
        )
    # Anything else (AccessDenied, SlowDown, transient 5xx) must fail
    # closed for the same reason: a negative grep result must not be
    # silently produced from a partial scan.
    return _VisitOutcome([], f"failed to read '{virtual_path}': {fetched.error}")


def _fetch_object(
    client: Any,
    bucket: str,
    key: str,
    virtual_path: str,
    regex: regex_mod.Pattern[str],
    *,
    max_size: int,
    max_line_length: int,
    timeout: float,
) -> _FetchOutcome:
    """Fetch a single object and return its regex matches.

    Uses :func:`read_capped_object` so the cap is enforced both against
    the GET ``ContentLength`` and against the actual bytes pulled from
    the body — a missing/lying ``Size`` in the prior listing or an
    object that grew between list and get cannot bypass it.

    Lines longer than ``max_line_length`` are skipped to keep regex
    backtracking cost bounded (ReDoS guard).

    Returns:
        :class:`_FetchOutcome`. ``matches`` carries the per-object
        results; ``error`` is ``None`` on success (including non-UTF-8
        bodies, which are skipped) or a short error code string when
        the caller must fail closed.
    """
    try:
        fetched = read_capped_object(client, bucket, key, max_size)
    except ClientError as exc:
        logger.exception("grep: failed to fetch '%s'", virtual_path)
        return _FetchOutcome([], sanitize_error_code(exc))
    except OversizeError:
        return _FetchOutcome([], OVERSIZE_ERROR_TAG)

    try:
        content = fetched.raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _FetchOutcome([], None)

    matches: list[GrepMatch] = []
    # ``TimeoutError`` from ``regex.search`` propagates up to ``grep()``
    # so the whole call fails closed; we do not catch it here because
    # finishing the remaining lines would still produce a partial result
    # while one line consumed the entire CPU budget.
    #
    # Iterate through ``StringIO`` rather than ``content.splitlines()``:
    # at the 5 MiB ``grep_max_file_size`` cap the list form would
    # double the resident set per object. ``StringIO`` yields one line
    # at a time and trailing terminators are stripped explicitly so
    # match text matches the prior splitlines() behavior.
    for line_num, raw_line in enumerate(io.StringIO(content), start=1):
        line = strip_line_terminator(raw_line)
        if len(line) > max_line_length:
            continue
        if regex.search(line, timeout=timeout):
            matches.append({"path": virtual_path, "line": line_num, "text": line})
    return _FetchOutcome(matches, None)
