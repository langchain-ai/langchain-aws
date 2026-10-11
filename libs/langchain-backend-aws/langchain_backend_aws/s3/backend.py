"""Amazon S3 backend implementing BackendProtocol for Deep Agents."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from botocore.exceptions import ClientError
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from langchain_backend_aws.s3._config import S3BackendConfig, build_client
from langchain_backend_aws.s3._glob import glob_search
from langchain_backend_aws.s3._grep import (
    GrepPrepared,
    classify_grep_result,
    format_grep_timeout_error,
    grep_paginate,
    prepare_grep,
    validate_grep_inputs,
)
from langchain_backend_aws.s3._internal import (
    GLOB_METACHARS,
    CappedReadResult,
    count_metachars,
    make_glob_compiler,
    read_capped_object,
    sanitize_error_code,
)
from langchain_backend_aws.s3._io import (
    download_files as _download_files,
)
from langchain_backend_aws.s3._io import (
    download_one as _download_one,
)
from langchain_backend_aws.s3._io import (
    upload_files as _upload_files,
)
from langchain_backend_aws.s3._ls import ls_listing
from langchain_backend_aws.s3._paths import (
    key_to_path as _key_to_path,
)
from langchain_backend_aws.s3._paths import (
    path_to_file_key as _path_to_file_key,
)
from langchain_backend_aws.s3._paths import (
    path_to_key as _path_to_key,
)
from langchain_backend_aws.s3._read import read_file as _read_file
from langchain_backend_aws.s3._write import edit_file as _edit_file
from langchain_backend_aws.s3._write import write_file as _write_file

logger = logging.getLogger(__name__)


class S3Backend(BackendProtocol):
    """Amazon S3 backend conforming to BackendProtocol for Deep Agents.

    Uses the standard S3 API (``get_object``, ``put_object``,
    ``list_objects_v2``) to persist agent files. File contents are stored
    as raw bytes so that objects remain directly readable by other tools
    (S3 Console, Athena, downstream services).

    Both ``created_at`` and ``modified_at`` timestamps in file metadata are
    sourced from the S3 object's ``LastModified`` field, since S3 does not
    track creation time separately.

    Example:
        .. code-block:: python

            from langchain_backend_aws.s3 import S3Backend, S3BackendConfig
            from deepagents import create_deep_agent

            config = S3BackendConfig(
                bucket="my-agent-files",
                prefix="sessions/abc123/",
                region_name="us-west-2",
            )
            agent = create_deep_agent(
                model="anthropic.claude-sonnet-4-20250514",
                backend=S3Backend(config),
                system_prompt="You are a helpful assistant.",
            )
    """

    def __init__(
        self,
        config: S3BackendConfig,
        *,
        client: Any | None = None,
    ) -> None:
        """Initialize the backend.

        The constructor accepts a fully-built :class:`S3BackendConfig`.
        For ad-hoc one-liners that do not want to import
        :class:`S3BackendConfig`, use :meth:`from_kwargs` instead — it
        forwards typed keyword arguments to :class:`S3BackendConfig` and
        keeps the validation surface explicit.

        Args:
            config: S3BackendConfig instance.
            client: Pre-built boto3 S3 client. If provided, the boto3
                client is not constructed from ``config``.

        Raises:
            ValueError: Propagated from :class:`S3BackendConfig` for
                semantic violations (bad prefix shape, SSRF allow-list
                rejection, malformed endpoint/proxy URL, etc.).
            TypeError: Propagated from :class:`S3BackendConfig` for
                shape violations in ``extra_boto_config`` (e.g. a
                ``proxies`` mapping whose values are not strings).
        """
        # ``prefix`` shape (no ``..``/empty/``.`` segments,
        # ``require_prefix`` honored) is enforced in
        # :meth:`S3BackendConfig.__post_init__`, so by the time we get
        # here the value is already safe to use. The slash-shape
        # normalisation lives on the config as ``normalized_prefix`` so
        # any future caller wiring ``_paths`` up directly inherits the
        # same invariant.
        self._config = config
        self._bucket = config.bucket
        self._prefix = config.normalized_prefix
        if not self._prefix:
            logger.warning(
                "S3Backend initialized with empty prefix; the backend has "
                "access to the entire bucket %r. Set ``prefix`` for tenant "
                "isolation in production deployments, or pass "
                "``require_prefix=True`` to fail closed.",
                self._bucket,
            )
        self._client = client if client is not None else build_client(config)
        # Per-instance glob compiler so a tenant cycling through many
        # one-off patterns cannot evict another tenant's hot pattern from
        # a process-global cache. The translation itself is pure, so the
        # only reason to scope it per-instance is cache isolation.
        self._compile_glob = make_glob_compiler()

    @classmethod
    def from_kwargs(
        cls,
        *,
        client: Any | None = None,
        **kwargs: Any,
    ) -> S3Backend:
        """Construct an :class:`S3Backend` from ad-hoc keyword arguments.

        Equivalent to ``S3Backend(S3BackendConfig(**kwargs), client=client)``
        but lets callers avoid importing :class:`S3BackendConfig` for a
        one-liner. Use this form (or pass an explicit
        :class:`S3BackendConfig` to ``__init__``) — the constructor
        does not accept config keyword arguments directly.

        Raises:
            ValueError: Forwarded from :class:`S3BackendConfig` for
                semantic violations (SSRF allow-list, bad prefix, etc.).
            TypeError: Forwarded from :class:`S3BackendConfig` for shape
                violations in ``extra_boto_config``.
        """
        return cls(S3BackendConfig(**kwargs), client=client)

    def clear_glob_cache(self) -> None:
        """Drop every entry from this instance's glob translator cache.

        Each :class:`S3Backend` owns a per-instance ``lru_cache``
        wrapping :func:`make_glob_compiler` so tenants cannot share
        compiled patterns. Long-lived backends serving a high cardinality
        of distinct globs accumulate one regex object per cache entry
        (up to the ``maxsize`` ceiling); this hook lets operators reclaim
        that memory between sessions without rebuilding the backend.
        Idempotent and safe to call from any thread because
        :py:func:`functools.lru_cache` synchronises ``cache_clear``.
        """
        cache_clear = getattr(self._compile_glob, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _max_bytes(self) -> int:
        """Cap in bytes for in-memory reads (read/edit/download_files)."""
        return self._config.max_file_size_mb * 1024 * 1024

    def _path_to_key(self, path: str) -> str:
        """Resolve ``path`` to an S3 object key under the configured prefix.

        Thin wrapper around :func:`_paths.path_to_key` that binds the
        backend's prefix; the validation rules live in
        :mod:`langchain_backend_aws.s3._paths`.
        """
        return _path_to_key(path, self._prefix)

    def _path_to_file_key(self, path: str) -> str:
        """Resolve ``path`` to a non-empty single-file S3 key.

        Thin wrapper around :func:`_paths.path_to_file_key`.
        """
        return _path_to_file_key(path, self._prefix)

    def _key_to_path(self, key: str) -> str:
        """Convert an S3 object key (under the prefix) back to a path.

        Thin wrapper around :func:`_paths.key_to_path`.
        """
        return _key_to_path(key, self._prefix)

    @staticmethod
    def _format_timestamp(dt: datetime) -> str:
        """Format a datetime to ISO 8601 string."""
        return dt.isoformat()

    def _read_capped(self, key: str) -> CappedReadResult:
        """Read a single object capped at :attr:`_max_bytes`.

        Wraps :func:`read_capped_object` with the backend's bucket and
        size cap so callers stay short.
        """
        return read_capped_object(self._client, self._bucket, key, self._max_bytes)

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List files and directories at the given path.

        Enumeration is capped at ``ls_max_objects`` to prevent unbounded
        scans on large buckets. When the cap is exceeded the call fails
        closed with an error rather than returning partial entries — a
        truncated listing could let callers conclude a file does not
        exist when it merely was not enumerated.

        Fails closed on prefix violation: if ``ListObjectsV2`` returns a
        key that does not start with the configured prefix, the entire
        listing is discarded and an error is returned. This indicates a
        broken isolation contract (misbehaving S3-compatible store,
        proxy, or stub) and silently dropping the offending entry would
        let the caller treat a partial result as authoritative.

        Args:
            path: Virtual path of the directory to list (``/`` for the
                backend root).

        Returns:
            :class:`LsResult` with the matching ``FileInfo`` entries
            sorted by path, or an error message describing why the
            listing could not be produced.
        """
        try:
            key_prefix = self._path_to_key(path)
        except ValueError as exc:
            return LsResult(error=str(exc))
        if key_prefix and not key_prefix.endswith("/"):
            key_prefix += "/"

        return ls_listing(
            client=self._client,
            bucket=self._bucket,
            prefix=self._prefix,
            path=path,
            key_prefix=key_prefix,
            key_to_path=self._key_to_path,
            format_timestamp=self._format_timestamp,
            max_objects=self._config.ls_max_objects,
        )

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content from S3 with line-based pagination.

        Notes:
            The file is split with :py:meth:`str.splitlines`, so
            ``\\n``/``\\r\\n`` line terminators and any trailing newline
            are normalized away. Callers needing byte-exact reads
            should use :meth:`download_files`.

            ``limit <= 0`` yields an empty selection (not an error).

            Offset semantics treat "read from the start" as always
            valid, while "skip past N lines" requires those lines to
            exist: ``offset=0`` against an empty file returns ``""``
            (success); ``offset == len(file)`` returns an empty
            selection (the requested skip is exactly satisfied);
            ``offset > len(file)`` returns an error.

        Args:
            file_path: Virtual path of the file to read.
            offset: Zero-based line offset to start at.
            limit: Maximum number of lines to return.

        Returns:
            :class:`ReadResult` carrying the requested slice (or the
            full base64 body for binary files when
            ``binary_read_mode="base64"``), or an error message.
        """
        return _read_file(
            file_path,
            offset,
            limit,
            path_to_file_key=self._path_to_file_key,
            read_capped=self._read_capped,
            format_timestamp=self._format_timestamp,
            max_file_size_mb=self._config.max_file_size_mb,
            binary_read_mode=self._config.binary_read_mode,
        )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a new file to S3. Errors if the key already exists.

        Uses S3's conditional ``PutObject`` with ``IfNoneMatch="*"`` so
        that "create if absent" is atomic against concurrent writers.
        Two writers racing on the same key cannot both succeed: the
        loser receives ``PreconditionFailed`` and is reported as
        already-exists, preventing silent overwrite.

        Note:
            Conditional ``PutObject`` is supported by AWS S3 (since Nov
            2024) and is required of any S3-compatible store this
            backend targets. MinIO/LocalStack are validation-only
            environments and may surface the same precondition as a
            different code on older builds. Older S3-compatible stores
            that silently ignore ``IfNoneMatch`` will accept the write
            unconditionally — atomicity in that case degrades to
            "last writer wins". Verify your target supports
            RFC 7232 ``If-None-Match`` before relying on the
            create-if-absent contract.

        Args:
            file_path: Virtual path of the file to create.
            content: Full UTF-8 text content for the new object.

        Returns:
            :class:`WriteResult` indicating success, an
            ``already_exists`` error when a concurrent writer won the
            ``IfNoneMatch`` race, or another classified error.
        """
        return _write_file(
            file_path,
            content,
            client=self._client,
            bucket=self._bucket,
            path_to_file_key=self._path_to_file_key,
        )

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by performing string replacement.

        Uses optimistic concurrency: the ETag observed at read time is
        passed back as ``IfMatch`` on the conditional ``PutObject`` so a
        concurrent writer that mutated the object between get and put
        cannot have its change silently overwritten. On precondition
        failure the caller is told to re-read and retry.

        Note:
            ``replace_all`` is intentionally positional-with-default to
            match the :class:`BackendProtocol` signature exactly. Linting
            (``FBT001``/``FBT002``) flags positional booleans, but
            keyword-only would break protocol conformance for callers
            that pass it positionally; the noqa is therefore load-bearing.

            **Upstream contract watch:** if :class:`BackendProtocol.edit`
            ever moves ``replace_all`` to keyword-only, the positional
            form here will silently diverge — the protocol type-check
            still passes because Python does not flag a positional
            parameter as incompatible with a keyword-only one in a
            ``Protocol`` subclass. Mirror the upstream change here when
            it lands, and add an integration test pinning the protocol
            signature so the divergence cannot recur unnoticed.

            Older S3-compatible stores that silently ignore ``IfMatch``
            will accept the write unconditionally — optimistic
            concurrency in that case degrades to "last writer wins".
            Verify your target supports RFC 7232 ``If-Match`` before
            relying on the conflict-detection contract.

        Args:
            file_path: Virtual path of the file to edit.
            old_string: Substring to locate in the current body.
            new_string: Replacement text. Pass an empty string to delete
                the substring.
            replace_all: When ``True``, replace every occurrence; when
                ``False`` (default), require a single occurrence and
                error otherwise.

        Returns:
            :class:`EditResult` describing the change, an
            ``already_exists``/``modified`` error when the ETag check
            fails, or another classified error.
        """
        return _edit_file(
            file_path,
            old_string,
            new_string,
            replace_all,
            client=self._client,
            bucket=self._bucket,
            path_to_file_key=self._path_to_file_key,
            read_capped=self._read_capped,
            max_file_size_mb=self._config.max_file_size_mb,
        )

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a regex pattern in text files under ``path``.

        Lists objects under the search prefix, applies an optional
        ``glob`` filter before fetching content, and skips binary or
        oversized files. The ``glob`` filter shares semantics with
        :meth:`glob`: ``**``/``*``/``?`` are honored and a pattern with
        no ``/`` also matches against the basename.

        Enumeration is capped at ``grep_max_objects``. When the cap is
        exceeded the call fails closed (returns an error) — same
        contract as :meth:`glob` and :meth:`ls`, so a missing match is
        never confused with "scan cap reached".

        ReDoS protection: ``pattern`` is rejected if it exceeds
        ``grep_max_pattern_length``, and individual lines longer than
        ``grep_max_line_length`` are skipped before reaching
        ``re.search`` to bound worst-case backtracking cost.

        Args:
            pattern: Regex pattern to search for.
            path: Optional virtual path scoping the search. Defaults to root.
            glob: Optional glob filter applied before fetch. Uses the
                same syntax as :meth:`glob`.

        Returns:
            GrepResult with matching lines or an error message.
        """
        validation_error = validate_grep_inputs(
            pattern,
            glob,
            grep_max_pattern_length=self._config.grep_max_pattern_length,
            grep_max_pattern_metachars=self._config.grep_max_pattern_metachars,
            glob_max_pattern_length=self._config.glob_max_pattern_length,
            glob_max_pattern_metachars=self._config.glob_max_pattern_metachars,
        )
        if validation_error is not None:
            return GrepResult(error=validation_error)

        search_path = path or "/"
        prepared, prepare_error = prepare_grep(
            pattern,
            glob,
            path_to_key=self._path_to_key,
            compile_glob=self._compile_glob,
            search_path=search_path,
        )
        if prepared is None:
            return GrepResult(error=prepare_error)

        return self._run_grep_pipeline(prepared, pattern, search_path)

    def _run_grep_pipeline(
        self,
        prepared: GrepPrepared,
        pattern: str,
        search_path: str,
    ) -> GrepResult:
        """Run the paginate + classify stages of :meth:`grep`.

        :meth:`grep` keeps the validation/preparation responsibilities
        (input shape caps, regex/glob compilation, base-key resolution);
        this helper owns the runtime pipeline (S3 enumeration, regex
        timeout / ClientError classification, sort + truncation
        bookkeeping). Splitting them keeps :meth:`grep` close to the
        50-line guideline and lets the runtime stages be exercised
        independently of input-validation tests.
        """
        try:
            paginate_result = grep_paginate(
                self._client,
                self._bucket,
                self._prefix,
                prepared.base_key,
                prepared.compiled,
                prepared.glob_regexes,
                key_to_path=self._key_to_path,
                max_objects=self._config.grep_max_objects,
                max_size=self._config.grep_max_file_size,
                max_line_length=self._config.grep_max_line_length,
                timeout=self._config.grep_regex_timeout,
                glob_timeout=self._config.glob_regex_timeout,
            )
        except ClientError as exc:
            logger.exception("Error during grep at '%s'", search_path)
            code = sanitize_error_code(exc)
            return GrepResult(error=f"Error during grep at '{search_path}': {code}")
        except TimeoutError as exc:
            return GrepResult(
                error=format_grep_timeout_error(
                    pattern,
                    search_path,
                    exc,
                    grep_regex_timeout=self._config.grep_regex_timeout,
                    glob_regex_timeout=self._config.glob_regex_timeout,
                )
            )

        matches, error = classify_grep_result(
            paginate_result,
            search_path,
            grep_max_objects=self._config.grep_max_objects,
        )
        if error is not None:
            return GrepResult(error=error)
        return GrepResult(matches=matches or [])

    # ------------------------------------------------------------------
    # glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern under ``path``.

        The pattern supports ``**`` (recursive segments), ``*`` (any
        characters except ``/``), and ``?`` (a single character except
        ``/``) and is matched against the path relative to ``path``.

        The basename fallback is **only** active when the pattern
        contains no ``/``. In that case ``*.py`` also matches any
        ``.py`` file at any depth (shell-like ergonomics). When the
        pattern contains a ``/`` (e.g. ``src/*.py``) the match is
        strict against the relative path, so depth is fixed by the
        number of segments in the pattern. This means ``*.py`` and
        ``src/*.py`` behave differently on purpose: the former is
        recursive, the latter is not.

        Enumeration is capped at ``glob_max_objects`` to prevent
        unbounded scans on large buckets.

        Args:
            pattern: Glob pattern. Leading ``/`` is stripped.
            path: Virtual path under which to evaluate the pattern.
                Defaults to backend root.

        Returns:
            :class:`GlobResult` with the matching ``FileInfo`` entries
            sorted by path, or an error message.
        """
        pattern = pattern.lstrip("/")
        if len(pattern) > self._config.glob_max_pattern_length:
            return GlobResult(
                error=(
                    f"Glob pattern length {len(pattern)} exceeds "
                    f"glob_max_pattern_length="
                    f"{self._config.glob_max_pattern_length}."
                )
            )
        # Counting glob wildcards bounds the number of stacked
        # ``[^/]*`` runs in the translated regex (same shape cap
        # ``grep`` applies to its source pattern). Length alone allows
        # a stuffed ``****…****`` under the length cap to compile into
        # a regex that backtracks catastrophically on match.
        glob_metachars = count_metachars(pattern, GLOB_METACHARS)
        if glob_metachars > self._config.glob_max_pattern_metachars:
            return GlobResult(
                error=(
                    f"Glob pattern wildcard count {glob_metachars} "
                    f"exceeds glob_max_pattern_metachars="
                    f"{self._config.glob_max_pattern_metachars}."
                )
            )

        try:
            base_key = self._path_to_key(path)
        except ValueError as exc:
            return GlobResult(error=str(exc))
        if base_key and not base_key.endswith("/"):
            base_key += "/"

        return glob_search(
            client=self._client,
            bucket=self._bucket,
            prefix=self._prefix,
            base_key=base_key,
            pattern=pattern,
            path=path,
            key_to_path=self._key_to_path,
            format_timestamp=self._format_timestamp,
            max_objects=self._config.glob_max_objects,
            timeout=self._config.glob_regex_timeout,
            compile_glob=self._compile_glob,
        )

    # ------------------------------------------------------------------
    # upload_files / download_files
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to S3.

        Each upload is bounded by ``max_file_size_mb`` for parity with
        :meth:`read`/:meth:`edit`/:meth:`download_files` — without this
        check a caller could push an arbitrarily large body straight
        through ``put_object``. Oversized uploads surface as
        ``error="oversize"`` — a backend-specific string sanctioned by
        the :class:`FileUploadResponse` docstring, which says the
        ``error`` field accepts "a backend-specific error string when
        the failure cannot be normalized" into the
        :data:`FileOperationError` Literal. The real cause is also
        logged at ERROR for triage.

        Args:
            files: Sequence of ``(virtual_path, body_bytes)`` pairs.

        Returns:
            One :class:`FileUploadResponse` per input file in input
            order. Failures (oversize, transport error, denied) are
            reported per slot rather than raising.
        """
        return _upload_files(
            self._client,
            self._bucket,
            files,
            path_to_file_key=self._path_to_file_key,
            max_bytes=self._max_bytes,
            max_file_size_mb=self._config.max_file_size_mb,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from S3.

        Each object is read with the same ``max_file_size_mb`` cap that
        :meth:`read` and :meth:`edit` apply, so a single oversized
        object cannot exhaust worker memory. Oversized reads surface as
        ``error="oversize"`` — a backend-specific string sanctioned by
        the :class:`FileDownloadResponse` docstring, which says the
        ``error`` field accepts "a backend-specific error string when
        the failure cannot be normalized" into the
        :data:`FileOperationError` Literal. The actual cause is also
        logged at ERROR for triage.

        Args:
            paths: Virtual paths of the files to fetch.

        Returns:
            One :class:`FileDownloadResponse` per input path in input
            order. Failures (oversize, not_found, denied) are reported
            per slot rather than raising.
        """
        return _download_files(
            paths,
            download_one=self._download_one_path,
            download_concurrency=self._config.download_concurrency,
            max_pool_connections=self._config.max_pool_connections,
        )

    def _download_one_path(self, path: str) -> FileDownloadResponse:
        """Download a single file via the shared helper."""
        return _download_one(
            path,
            path_to_file_key=self._path_to_file_key,
            read_capped=self._read_capped,
            max_file_size_mb=self._config.max_file_size_mb,
        )


__all__ = ["S3Backend", "S3BackendConfig"]
