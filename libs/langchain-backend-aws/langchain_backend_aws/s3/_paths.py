"""Path-to-key validation helpers for :class:`S3Backend`.

Extracted from ``backend.py`` so the ``BackendProtocol`` implementation
stays focused on protocol methods and the traversal/normalization
rules can be tested in isolation. Each helper covers one specific
defense and raises ``ValueError`` with a caller-facing message; the
orchestrator :func:`path_to_key` chains them in the order required to
catch encoded-then-normalized variants.
"""

from __future__ import annotations

import unicodedata

from deepagents.backends.utils import validate_path

_TRAVERSAL_SEGMENTS: frozenset[str] = frozenset({"", "..", "."})

# Overlong UTF-8 percent-encoded forms of ``.`` (U+002E) and ``/``
# (U+002F). RFC 3629 forbids these encodings, but historical decoders
# (and some hand-rolled normalizers in the upstream ``validate_path``
# stack) have decoded them to the canonical ASCII byte. Treat them the
# same as the single-layer ``%2e``/``%2f`` forms so a path like
# ``foo%c0%ae%c0%ae/bar`` is rejected before reaching the segment scan.
_OVERLONG_PERCENT_ENCODINGS: frozenset[str] = frozenset(
    {
        # Two-byte overlong encodings (C0 80–BF range).
        "%c0%ae",
        "%c0%af",
        # Three-byte overlong encodings (E0 80 80–BF range).
        "%e0%80%ae",
        "%e0%80%af",
        # Four-byte overlong encodings (F0 80 80 80–BF range).
        "%f0%80%80%ae",
        "%f0%80%80%af",
    }
)


def _assert_prefix_shape(prefix: str) -> None:
    """Assert the bound ``prefix`` is empty, well-shaped, and traversal-free.

    The boundary check in :func:`key_to_path` is a byte-prefix
    ``startswith(prefix)``: with ``prefix="tenantA"`` (no trailing
    slash) it would also accept ``"tenantAB/..."`` and silently leak
    cross-tenant. :attr:`S3BackendConfig.normalized_prefix` already
    appends a trailing ``/`` to a non-empty prefix, so the live path
    is safe; this assertion locks the invariant for any future caller
    that wires these helpers up directly.

    Defense-in-depth: also rescans for ``..``/empty/``.`` segments.
    ``S3BackendConfig.__post_init__`` already rejects these at
    construction time, but a future caller that builds a prefix
    string from a different source would otherwise bypass that check
    when calling :func:`path_to_key` directly.
    """
    if prefix and not prefix.endswith("/"):
        msg = (
            f"S3 prefix {prefix!r} must be empty or end with '/' so "
            "byte-prefix containment maps to segment-aware containment."
        )
        raise ValueError(msg)
    if prefix:
        # Trim the single trailing ``/`` so a well-shaped prefix
        # ``tenantA/`` becomes ``tenantA`` for segment splitting; an
        # internal ``//`` (empty segment) still surfaces.
        for segment in prefix[:-1].split("/"):
            if segment in _TRAVERSAL_SEGMENTS:
                msg = (
                    f"S3 prefix {prefix!r} contains an empty or "
                    "traversal segment ('', '.', '..'); refusing to "
                    "use it as the prefix isolation boundary."
                )
                raise ValueError(msg)


def _reject_percent_encoded_traversal(path: str, nfkc_lower: str) -> None:
    """Reject single-layer percent-encoded traversal sequences.

    Scans for a single layer of encoding only (``%2e%2e`` / ``%2f``,
    plus the overlong UTF-8 encodings of ``.`` and ``/`` listed in
    :data:`_OVERLONG_PERCENT_ENCODINGS`); doubly-encoded variants such
    as ``%252e%252e`` are intentionally NOT decoded here. We rely on
    ``validate_path`` upstream to perform at most one round of
    percent-decoding before handing us the path, so a literal ``%25``
    reaching this point is treated as user data, not as an encoded
    ``%``. If a future caller introduces a second decode pass before
    reaching :func:`path_to_key` this assumption must be re-evaluated.

    Overlong UTF-8 forms (``%c0%ae`` etc.) are forbidden by RFC 3629
    but historically decoded to the canonical ASCII byte by lenient
    normalizers; rejecting them here closes the gap so a hardened
    upstream cannot be silently weakened by a future regression.

    Args:
        path: Original path (used only for the error message).
        nfkc_lower: NFKC-normalized, lower-cased candidate to scan.

    Raises:
        ValueError: If the candidate contains ``%2e%2e``, ``%2f``, or
            any overlong UTF-8 encoding of ``.``/``/``.
    """
    if "%2e%2e" in nfkc_lower or "%2f" in nfkc_lower:
        msg = (
            f"Path {path!r} contains percent-encoded traversal "
            "sequences; refusing to resolve it to an S3 key."
        )
        raise ValueError(msg)
    for token in _OVERLONG_PERCENT_ENCODINGS:
        if token in nfkc_lower:
            msg = (
                f"Path {path!r} contains an overlong UTF-8 "
                "percent-encoded byte; refusing to resolve it to an "
                "S3 key."
            )
            raise ValueError(msg)


def _reject_bare_percent_encoded_dot(path: str, nfkc_lower: str) -> None:
    """Reject segments that are exactly ``%2e`` (encoded ``.``).

    A segment encoded as a bare ``%2e`` would decode to ``.`` and slip
    past the literal segment scan in :func:`_reject_traversal_segments`,
    which only sees ``%2e`` as a regular character. Reject any segment
    that *is* a single ``%2e`` (case-insensitive) so a path like
    ``foo/%2e/bar`` cannot mask a self-segment intent.

    Raises:
        ValueError: If any path segment equals ``%2e``.
    """
    for segment in nfkc_lower.split("/"):
        if segment == "%2e":
            msg = (
                f"Path {path!r} contains a percent-encoded dot segment; "
                "refusing to resolve it to an S3 key."
            )
            raise ValueError(msg)


def _reject_traversal_segments(path: str, stripped: str, nfkc: str) -> None:
    """Reject empty / ``.`` / ``..`` segments after normalization.

    Scans both the original stripped form and the NFKC-normalized form
    so visually-confusable code points (e.g. ``U+FF0E FULLWIDTH FULL
    STOP`` ``．．`` collapsing to ``..``) are flattened into the
    canonical ASCII form the segment check rejects.

    Raises:
        ValueError: If any candidate contains a traversal segment.
    """
    for candidate in (stripped, nfkc):
        if not candidate:
            continue
        if any(seg in _TRAVERSAL_SEGMENTS for seg in candidate.split("/")):
            msg = (
                f"Path {path!r} contains a traversal or empty "
                "segment after normalization; refusing to resolve it "
                "to an S3 key."
            )
            raise ValueError(msg)


def path_to_key(path: str, prefix: str) -> str:
    """Convert a virtual path to an S3 object key.

    Validates the path to prevent traversal attacks. May return an
    empty string when both ``prefix`` and the normalized path are
    empty (root of an unscoped bucket); :func:`path_to_file_key` is
    the per-file variant that rejects this case.

    Defense-in-depth: ``validate_path`` is upstream-supplied and has
    historically had edge cases (Unicode normalization, percent-encoded
    ``..``) that could let a path slip past the traversal check. Each
    helper below covers one specific attack so a regression upstream
    cannot escape the configured prefix.

    Args:
        path: Virtual path supplied by the caller.
        prefix: Configured backend prefix (already shape-validated).

    Returns:
        ``prefix`` followed by the normalized, leading-slash-stripped
        path. Empty when both sides are empty.

    Raises:
        ValueError: If the path contains traversal sequences, or if
            ``prefix`` is non-empty without a trailing ``/`` (which
            would break segment-aware containment in :func:`key_to_path`).
    """
    _assert_prefix_shape(prefix)
    normalized = validate_path(path)
    stripped = normalized.lstrip("/")
    nfkc = unicodedata.normalize("NFKC", stripped)
    nfkc_lower = nfkc.lower()
    _reject_percent_encoded_traversal(path, nfkc_lower)
    _reject_bare_percent_encoded_dot(path, nfkc_lower)
    _reject_traversal_segments(path, stripped, nfkc)
    return f"{prefix}{stripped}" if stripped else prefix


def path_to_file_key(path: str, prefix: str) -> str:
    """Convert a virtual path to an S3 key for a single-file operation.

    Same as :func:`path_to_key` but additionally rejects paths that
    resolve to an empty key. With ``prefix=""`` and ``path="/"`` the
    underlying key is ``""`` which would surface as a low-level
    ``ParamValidationError`` from boto3 (or worse, an
    ``InvalidArgument`` from S3); reject it close to the public API
    so callers see a clear "path is not a file" error.

    Raises:
        ValueError: If the path contains traversal sequences or
            resolves to an empty object key.
    """
    key = path_to_key(path, prefix)
    if not key:
        msg = (
            f"Path {path!r} does not refer to a file (resolved to an empty object key)."
        )
        raise ValueError(msg)
    return key


def key_to_path(key: str, prefix: str) -> str:
    """Convert an S3 object key to a virtual path.

    Assumes ``key`` was returned from a paginator scoped to ``prefix``
    (i.e. always begins with the prefix). When the prefix is empty
    the check is trivially satisfied.

    Raises:
        ValueError: If ``key`` does not start with ``prefix``, or if
            ``prefix`` is non-empty without a trailing ``/``.
    """
    _assert_prefix_shape(prefix)
    if not key.startswith(prefix):
        msg = f"S3 key {key!r} is outside the configured prefix"
        raise ValueError(msg)
    return f"/{key[len(prefix) :]}"
