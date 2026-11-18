"""Configuration dataclass and boto3 client factory for :class:`S3Backend`.

Kept separate from ``backend.py`` so the configuration surface — which
is the public API most callers touch — can be read without scrolling
past the ``BackendProtocol`` implementation.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import boto3
from botocore.config import Config as BotoConfig

from langchain_backend_aws.s3._defaults import (
    ALLOWED_BOTO_KEYS as _ALLOWED_BOTO_KEYS,
)
from langchain_backend_aws.s3._defaults import (
    ALLOWED_PROXIES_CONFIG_KEYS as _ALLOWED_PROXIES_CONFIG_KEYS,
)
from langchain_backend_aws.s3._defaults import (
    DEFAULT_DOWNLOAD_CONCURRENCY,
    DEFAULT_GLOB_MAX_OBJECTS,
    DEFAULT_GLOB_MAX_PATTERN_LENGTH,
    DEFAULT_GLOB_MAX_PATTERN_METACHARS,
    DEFAULT_GLOB_REGEX_TIMEOUT,
    DEFAULT_GREP_MAX_FILE_SIZE,
    DEFAULT_GREP_MAX_LINE_LENGTH,
    DEFAULT_GREP_MAX_OBJECTS,
    DEFAULT_GREP_MAX_PATTERN_LENGTH,
    DEFAULT_GREP_MAX_PATTERN_METACHARS,
    DEFAULT_GREP_REGEX_TIMEOUT,
    DEFAULT_LS_MAX_OBJECTS,
    DEFAULT_MAX_FILE_SIZE_MB,
)
from langchain_backend_aws.s3._defaults import (
    EXPLICIT_BOTO_KEYS as _EXPLICIT_BOTO_KEYS,
)
from langchain_backend_aws.s3._ssrf import validate_url_against_ssrf

BinaryReadMode = Literal["base64", "error"]

logger = logging.getLogger(__name__)

# Per-cap rationales (ReDoS, blast-radius, SSRF) and the
# ``EXPLICIT_BOTO_KEYS`` / ``ALLOWED_BOTO_KEYS`` allow-lists live in
# :mod:`._defaults` so the dataclass surface in this module stays
# focused on configuration. SSRF allow-list constants and helpers live
# in :mod:`._ssrf`.

# Derived from the ``BinaryReadMode`` Literal via :func:`typing.get_args`
# so the runtime accepted set stays in lockstep with the type alias —
# adding a new mode requires editing only the Literal above. ``get_args``
# returns a tuple in declaration order; we freeze it for membership
# testing and cheap repr ordering.
_BINARY_READ_MODE_VALUES: frozenset[str] = frozenset(get_args(BinaryReadMode))


def _fingerprint_client_cert(client_cert: Any) -> str:
    """Return a redacted, deterministic fingerprint for ``client_cert``.

    The audit log records that ``client_cert`` is set without leaking
    the underlying filesystem paths (especially the private-key path).
    A SHA-256 hex digest of each path component, truncated to 12 chars,
    is enough for an operator to correlate identical configurations
    across logs while hiding the literal path.
    """

    def _hash(value: Any) -> str:
        encoded = str(value).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()[:12]

    if isinstance(client_cert, tuple):
        return "(" + ", ".join(f"sha256:{_hash(item)}" for item in client_cert) + ")"
    return f"sha256:{_hash(client_cert)}"


@dataclass(frozen=True)
class S3BackendConfig:
    """Configuration for :class:`S3Backend`.

    Security note:
        ``extra_boto_config`` (including nested ``proxies`` /
        ``proxies_config``) is treated as **operator-supplied** trusted
        input. Validation here enforces shape and SSRF allow-listing on
        URL-bearing fields, but does NOT deep-validate non-URL metadata
        such as ``client_cert`` tuples or CA bundle paths. Never source
        ``extra_boto_config`` (or ``endpoint_url``) from agent-controlled
        data; operators who need a tighter contract should build their
        own :class:`botocore.config.Config` and pass a pre-built
        ``client=`` to :class:`S3Backend`, removing this surface
        entirely.

    Attributes:
        bucket: S3 bucket name.
        prefix: Optional key prefix scoping all operations under a sub-path.
        region_name: AWS region for the S3 client.
        aws_access_key_id: Explicit AWS access key ID.
        aws_secret_access_key: Explicit AWS secret access key.
        aws_session_token: Optional AWS session token.
        endpoint_url: Custom endpoint URL (useful for LocalStack/MinIO).
            Must NEVER be sourced from untrusted user input — accepting
            arbitrary URLs would expose internal network services to SSRF.
            Validated at construction: only ``http``/``https`` schemes are
            accepted, and hostnames resolving to loopback, link-local
            (IMDS ``169.254.0.0/16``), or RFC1918 private ranges are
            rejected unless ``allow_private_endpoints=True`` opts in.

            **DNS-resolution gap (residual risk):** the SSRF allow-list
            inspects only IP literals and a static set of private/wildcard
            DNS suffixes (see :mod:`._ssrf`). A hostname that resolves to
            a private/loopback range *at runtime* via plain DNS (no
            literal in the name, no known suffix match) is NOT blocked
            here — config-time DNS resolution would be racy and easy to
            bypass. Operators who source ``endpoint_url`` from a
            less-trusted layer (configmaps, sidecar templates, etc.) are
            responsible for locking that layer down; this field assumes a
            trusted operator-supplied value.
        allow_private_endpoints: When ``True``, ``endpoint_url`` is allowed
            to point at loopback/RFC1918/link-local hosts. Defaults to
            ``False`` so a misconfiguration cannot accidentally talk to
            IMDS or a sidecar. Set this only for legitimate local-dev
            cases (LocalStack on ``localhost``, MinIO in a Compose
            network) where the operator has audited the destination.
        max_retries: boto3 client max retry attempts. The retry *mode*
            is fixed at ``"adaptive"`` (botocore's token-bucket retry
            policy) and cannot be overridden via ``extra_boto_config``;
            ``extra_boto_config['retries']`` is silently dropped to
            avoid passing the same kwarg twice. Operators who need a
            different mode must build their own
            :class:`botocore.config.Config` and pass a pre-built
            ``client=`` to :class:`S3Backend`.
        connect_timeout: boto3 client connect timeout in seconds.
        read_timeout: boto3 client read timeout in seconds.
        max_pool_connections: boto3 client connection pool size.
        grep_max_objects: Maximum number of objects enumerated from
            ``ListObjectsV2`` in a single grep call. The cap counts every
            listed object — including ones later skipped by the ``glob``
            filter or by ``grep_max_file_size`` — so it bounds list
            traffic, not match attempts.
        grep_max_file_size: Maximum object size in bytes considered by grep.
        glob_max_objects: Maximum number of objects enumerated in a single
            glob call. Caps blast radius on large buckets.
        ls_max_objects: Maximum number of entries (``Contents`` and
            ``CommonPrefixes`` combined) enumerated in a single ls call.
            Both file objects and directory-style common prefixes count
            toward the same cap, so on a directory that mixes the two
            the file slice and directory slice share one budget. Caps
            blast radius on large buckets.
        max_file_size_mb: Maximum file size in MiB that ``read``,
            ``edit``, ``download_files`` and ``upload_files`` will load
            into memory or accept on the wire. Larger objects fail with
            an error rather than being partially read or risking OOM.
            Set in MiB (mirrors deepagents' ``FilesystemBackend``).
        grep_max_pattern_length: Maximum length of the user-supplied
            regex pattern accepted by grep. Patterns longer than this
            are rejected with an error before reaching ``re.compile``,
            limiting the surface for catastrophic backtracking (ReDoS).
        grep_max_line_length: Skip lines longer than this many bytes
            during grep. Long lines are the dominant ReDoS amplifier;
            refusing to feed them to ``re.search`` keeps worst-case CPU
            bounded even when the pattern itself is benign.
        glob_max_pattern_length: Maximum length of the glob pattern
            accepted by ``glob`` and by the optional ``glob`` filter on
            ``grep``. Patterns longer than this are rejected with an
            error before reaching the glob translator, limiting the
            surface for stacked-quantifier backtracking on match.
        grep_regex_timeout: Wall-clock cap (in seconds) on each
            ``regex.search`` call performed by ``grep``. Bounds the
            worst case when a crafted pattern (catastrophic
            backtracking) would otherwise pin a worker. Exceeding the
            cap surfaces as a grep error rather than freezing the
            process.
        require_prefix: When ``True``, ``S3Backend.__init__`` raises
            ``ValueError`` if ``prefix`` is empty. Use in production to
            fail closed rather than silently grant the backend access to
            the entire bucket. Defaults to ``False`` for backwards
            compatibility (a warning is still emitted).
        binary_read_mode: How ``read`` surfaces non-UTF-8 bodies. With
            ``"base64"`` (default), the full body is returned encoded as
            base64 with ``encoding="base64"`` on ``file_data`` — note
            that ``offset``/``limit`` do not apply. With ``"error"``,
            ``read`` returns an error directing the caller to
            ``download_files`` instead, which avoids loading several
            MiB of binary data into the agent's context window.
        grep_max_pattern_metachars: Maximum count of regex metacharacters
            (parens, brackets, quantifiers, alternation, escape) in the
            user-supplied grep pattern. Bounds compile cost and
            nested-quantifier ReDoS even when ``grep_max_pattern_length``
            permits the source string.
        glob_max_pattern_metachars: Maximum count of glob wildcards
            (``*`` and ``?``) accepted in glob patterns and in the
            ``glob`` filter on ``grep``. Bounds the number of stacked
            ``[^/]*`` runs in the translated regex — the catastrophic
            backtracking surface for glob — independently of the
            source-length cap.
        glob_regex_timeout: Wall-clock cap (in seconds) on each glob
            regex match. Mirrors ``grep_regex_timeout`` and is enforced
            both in ``glob`` and in ``grep``'s glob filter so a crafted
            wildcard run cannot pin a worker on a single object.
        download_concurrency: Maximum number of concurrent
            ``download_files`` fetches. Capped at runtime by
            ``max_pool_connections`` to avoid exceeding the boto3
            connection pool. Set to ``1`` to force sequential fetches.
        extra_boto_config: Extra kwargs forwarded to
            :class:`botocore.config.Config`. Keys that overlap with
            explicit config attributes (``retries``, ``connect_timeout``,
            ``read_timeout``, ``max_pool_connections``) are silently
            dropped so the explicit attributes always win — passing
            both via different fields would otherwise raise
            ``TypeError`` from a duplicate keyword argument. Use this
            for orthogonal options like ``signature_version`` or
            ``s3={"addressing_style": "virtual"}``.

            ``proxies`` URLs go through the same SSRF allow-list as
            ``endpoint_url`` (scheme restricted to http/https; loopback
            / link-local / RFC1918 hosts rejected unless
            ``allow_private_endpoints=True``). A malicious proxy URL
            would otherwise re-introduce the same SSRF surface via the
            transport. ``proxies_config`` carries non-URL metadata
            (CA bundle paths, client cert) and is forwarded as-is.
            Either key — and ``extra_boto_config`` as a whole — must
            never be sourced from agent-controlled data.
    """

    bucket: str
    prefix: str = ""
    region_name: str | None = None
    # Credential fields are excluded from ``repr`` so the value cannot
    # leak via logs, tracebacks, or test assertion output. The dataclass
    # default ``repr`` would otherwise print every field.
    aws_access_key_id: str | None = field(default=None, repr=False)
    aws_secret_access_key: str | None = field(default=None, repr=False)
    aws_session_token: str | None = field(default=None, repr=False)
    endpoint_url: str | None = None
    max_retries: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    max_pool_connections: int = 50
    grep_max_objects: int = DEFAULT_GREP_MAX_OBJECTS
    grep_max_file_size: int = DEFAULT_GREP_MAX_FILE_SIZE
    glob_max_objects: int = DEFAULT_GLOB_MAX_OBJECTS
    ls_max_objects: int = DEFAULT_LS_MAX_OBJECTS
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    grep_max_pattern_length: int = DEFAULT_GREP_MAX_PATTERN_LENGTH
    grep_max_line_length: int = DEFAULT_GREP_MAX_LINE_LENGTH
    glob_max_pattern_length: int = DEFAULT_GLOB_MAX_PATTERN_LENGTH
    grep_regex_timeout: float = DEFAULT_GREP_REGEX_TIMEOUT
    grep_max_pattern_metachars: int = DEFAULT_GREP_MAX_PATTERN_METACHARS
    glob_max_pattern_metachars: int = DEFAULT_GLOB_MAX_PATTERN_METACHARS
    glob_regex_timeout: float = DEFAULT_GLOB_REGEX_TIMEOUT
    require_prefix: bool = False
    binary_read_mode: BinaryReadMode = "base64"
    download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY
    allow_private_endpoints: bool = False
    extra_boto_config: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_prefix(self) -> str:
        """Return ``prefix`` normalized to the byte-prefix shape used as
        the bucket-isolation boundary.

        Either empty (unscoped) or ``<segments>/`` with a single trailing
        slash; ``key.startswith(normalized_prefix)`` therefore implies
        segment-aware containment and never partially matches a sibling
        key like ``"tenantAB/..."``. Centralising the construction here
        keeps the invariant in one place — the backend, ``_paths`` and
        ``_internal`` all consume this property instead of re-deriving
        the slash shape from the raw ``prefix`` field, so a future
        caller cannot accidentally feed an unsanitised prefix into
        :func:`_paths.path_to_key`.

        ``__post_init__`` already rejected ``..``/``.``/empty segments
        before this property is read, so the value here is safe.
        """
        raw = self.prefix.strip("/")
        return f"{raw}/" if raw else ""

    def __post_init__(self) -> None:
        """Validate fields that have a non-trivial accepted shape.

        Each validation rule is split into a private ``_validate_*``
        helper so the dispatcher here stays readable and individual
        rules can be unit-tested in isolation.

        ``extra_boto_config`` is deep-copied **before** validation so
        the dict the validators inspect is the same dict the backend
        will use. A caller holding a reference to the original mapping
        therefore cannot bypass the SSRF allow-list either by
        post-construction insertion or by aliasing a nested dict
        (``proxies``, ``proxies_config``) and mutating it between two
        validators. ``deepcopy`` breaks the alias as well as the outer
        reference.

        Defense-in-depth: scheme is restricted to http/https, and the
        hostname is rejected if it resolves to loopback, link-local
        (IMDS), or RFC1918 private ranges unless ``allow_private_endpoints``
        is set. Operators running against LocalStack/MinIO must opt in
        explicitly; production deployments leave the flag off so a stray
        ``endpoint_url=http://169.254.169.254/...`` cannot reach IMDS.

        Raises:
            ValueError: A field value fails a semantic check
                (bad binary_read_mode, malformed bucket/region/prefix,
                SSRF allow-list rejection, non-copyable
                ``extra_boto_config`` value, etc.).
            TypeError: A field value has the wrong shape
                (``proxies``/``proxies_config`` not a dict, proxy URL
                value not a string, etc.) — surfaced separately from
                ``ValueError`` so callers can distinguish a
                misuse-of-API from a misconfigured-but-typed value.
        """
        # Order is load-bearing:
        #   1. cheap shape checks (binary mode, bucket/region, prefix)
        #      run first so a mistake surfaces before we touch the
        #      ``extra_boto_config`` dict at all,
        #   2. ``deepcopy`` then snapshots ``extra_boto_config`` so
        #      every subsequent validator (and ``build_client``) reads
        #      the same dict the backend will use — a caller cannot
        #      bypass the SSRF allow-list by aliasing ``proxies`` and
        #      mutating it between two validators,
        #   3. boto-config validators (allow-listed keys → endpoint URL
        #      → proxies → proxies_config) run last so failures from
        #      the network surface (SSRF, malformed proxy URL) report
        #      against the snapshotted dict.
        self._validate_binary_read_mode()
        self._validate_bucket_region()
        self._validate_prefix()
        # Bypass ``frozen=True`` once during construction to snapshot
        # ``extra_boto_config``. Post-construction the dataclass is fully
        # immutable so callers cannot bypass SSRF validation by mutating
        # the original mapping or aliased nested dicts after the fact.
        #
        # ``deepcopy`` can raise ``TypeError`` (or any other exception
        # from a third-party ``__deepcopy__``) when the supplied dict
        # contains non-copyable values such as file handles, sockets,
        # or objects holding ``threading.Lock``. Normalize the failure
        # into a ``ValueError`` with a configuration-shaped message so
        # the caller sees a consistent diagnostic alongside the other
        # ``_validate_*`` checks rather than an opaque copy error.
        # The ``Exception`` catch deliberately stops below
        # ``BaseException`` so ``SystemExit``/``KeyboardInterrupt``
        # raised by a malicious or buggy ``__deepcopy__`` keep
        # propagating; only normal copy failures are remapped.
        try:
            snapshot = copy.deepcopy(self.extra_boto_config)
        except Exception as exc:  # noqa: BLE001
            msg = (
                "extra_boto_config contains a non-copyable value "
                f"({type(exc).__name__}: {exc}); supply only plain "
                "dict/list/str/int/bool/None values so the snapshot "
                "cannot be mutated post-construction."
            )
            raise ValueError(msg) from exc
        object.__setattr__(self, "extra_boto_config", snapshot)
        self._validate_extra_boto_keys()
        self._validate_endpoint_url()
        self._validate_proxies()
        self._validate_proxies_config()
        self._validate_s3_options()
        self._audit_client_cert()

    def _validate_binary_read_mode(self) -> None:
        """Reject ``binary_read_mode`` values outside the literal set."""
        if self.binary_read_mode not in _BINARY_READ_MODE_VALUES:
            msg = (
                f"binary_read_mode={self.binary_read_mode!r} is not "
                f"one of {sorted(_BINARY_READ_MODE_VALUES)}."
            )
            raise ValueError(msg)

    def _validate_bucket_region(self) -> None:
        """Reject blank ``bucket`` / ``region_name`` close to the call site.

        Letting these flow into boto3 surfaces them as a generic
        ``ParamValidationError`` or as a successful client build that
        fails on the first call; the explicit check keeps misuse
        contained to the dataclass.
        """
        if not isinstance(self.bucket, str) or not self.bucket.strip():
            msg = (
                "S3BackendConfig.bucket must be a non-empty string; got "
                f"{self.bucket!r}."
            )
            raise ValueError(msg)
        if self.region_name is not None and (
            not isinstance(self.region_name, str) or not self.region_name.strip()
        ):
            msg = (
                "S3BackendConfig.region_name must be a non-empty string when "
                f"set; got {self.region_name!r}. Pass ``None`` to fall back to "
                "boto3's default resolution chain."
            )
            raise ValueError(msg)

    def _validate_prefix(self) -> None:
        """Reject ``..``/empty/`.` segments and honor ``require_prefix``.

        ``_path_to_key`` only checks ``startswith(self._prefix)``, so a
        prefix like ``foo/../`` would normalize against the bucket root
        and silently widen the isolation boundary if it slipped past.
        """
        raw_prefix = self.prefix.strip("/")
        if raw_prefix:
            for segment in raw_prefix.split("/"):
                if segment in {"", ".", ".."}:
                    msg = (
                        f"S3BackendConfig.prefix {self.prefix!r} contains "
                        "an empty or traversal segment ('', '.', '..'); "
                        "the backend rejects these to keep prefix isolation "
                        "intact."
                    )
                    raise ValueError(msg)
        if not raw_prefix and self.require_prefix:
            msg = (
                "S3BackendConfig.require_prefix=True but prefix is empty. "
                "Set ``prefix`` to scope the backend under a sub-path; "
                "an empty prefix would grant access to the entire bucket."
            )
            raise ValueError(msg)

    def _validate_extra_boto_keys(self) -> None:
        """Reject ``extra_boto_config`` keys outside the allow-list."""
        unknown = set(self.extra_boto_config) - _ALLOWED_BOTO_KEYS - _EXPLICIT_BOTO_KEYS
        if unknown:
            msg = (
                f"extra_boto_config contains unsupported keys "
                f"{sorted(unknown)}. Allowed keys are "
                f"{sorted(_ALLOWED_BOTO_KEYS)}; explicit fields cover "
                f"{sorted(_EXPLICIT_BOTO_KEYS)}. Build your own "
                "botocore.config.Config and pass a pre-built client= if "
                "you need an option outside this set."
            )
            raise ValueError(msg)

    def _validate_endpoint_url(self) -> None:
        """Apply the SSRF allow-list to ``endpoint_url`` when set."""
        if self.endpoint_url is not None:
            validate_url_against_ssrf(
                self.endpoint_url,
                field_name="endpoint_url",
                allow_private=self.allow_private_endpoints,
            )

    def _validate_proxies(self) -> None:
        """Validate ``extra_boto_config['proxies']`` shape and SSRF.

        botocore expects a dict mapping scheme string to URL string.
        Each URL goes through the same SSRF allow-list as ``endpoint_url``
        so a malicious proxy URL cannot re-introduce the SSRF surface
        via the transport.
        """
        proxies = self.extra_boto_config.get("proxies")
        if proxies is None:
            return
        if not isinstance(proxies, dict):
            msg = (
                f"extra_boto_config['proxies'] must be a dict mapping "
                f"scheme to URL, got {type(proxies).__name__}."
            )
            raise TypeError(msg)
        for scheme_key, url in proxies.items():
            if not isinstance(scheme_key, str):
                msg = (
                    "extra_boto_config['proxies'] keys must be str, got "
                    f"{type(scheme_key).__name__}."
                )
                raise TypeError(msg)
            if not isinstance(url, str):
                msg = (
                    f"extra_boto_config['proxies'][{scheme_key!r}] must "
                    f"be a string URL, got {type(url).__name__}."
                )
                raise TypeError(msg)
            validate_url_against_ssrf(
                url,
                field_name=f"extra_boto_config['proxies'][{scheme_key!r}]",
                allow_private=self.allow_private_endpoints,
            )

    def _validate_proxies_config(self) -> None:
        """Type-check ``extra_boto_config['proxies_config']`` shape.

        Carries non-URL metadata (CA bundle paths, ``client_cert``
        tuples) and is forwarded to botocore as-is; rejecting
        malformed values here avoids confusing TLS-init failures deep
        inside botocore.

        **Contents are NOT deep-validated** beyond the
        ``str | bool | tuple | None`` shape: a ``client_cert`` tuple
        like ``(cert_path, key_path)`` could point at any local file,
        and a malicious dict could exfiltrate sensitive paths through
        TLS handshake errors. This is acceptable only because
        ``extra_boto_config`` is documented as operator-supplied and
        must never be sourced from agent-controlled data. Operators
        who want a tighter contract should build their own
        :class:`botocore.config.Config` and pass a pre-built
        ``client=`` to :class:`S3Backend`, removing the
        ``extra_boto_config`` surface entirely.
        """
        proxies_config = self.extra_boto_config.get("proxies_config")
        if proxies_config is None:
            return
        if not isinstance(proxies_config, dict):
            msg = (
                f"extra_boto_config['proxies_config'] must be a dict, "
                f"got {type(proxies_config).__name__}."
            )
            raise TypeError(msg)
        unknown = set(proxies_config) - _ALLOWED_PROXIES_CONFIG_KEYS
        if unknown:
            # Mirror the outer ``extra_boto_config`` allow-list discipline:
            # an unknown inner key would otherwise reach botocore and
            # surface as a confusing TLS-init failure on the first request.
            msg = (
                f"extra_boto_config['proxies_config'] contains unsupported "
                f"keys {sorted(unknown)}. Allowed keys are "
                f"{sorted(_ALLOWED_PROXIES_CONFIG_KEYS)}. Build your own "
                "botocore.config.Config and pass a pre-built client= if "
                "you need an option outside this set."
            )
            raise ValueError(msg)
        for cfg_key, cfg_value in proxies_config.items():
            if not isinstance(cfg_key, str):
                msg = (
                    "extra_boto_config['proxies_config'] keys must be "
                    f"str, got {type(cfg_key).__name__}."
                )
                raise TypeError(msg)
            if cfg_value is not None and not isinstance(cfg_value, (str, bool, tuple)):
                msg = (
                    f"extra_boto_config['proxies_config'][{cfg_key!r}] "
                    f"must be str, bool, tuple, or None; got "
                    f"{type(cfg_value).__name__}."
                )
                raise TypeError(msg)
            # ``proxy_client_cert`` is the only tuple-shaped key botocore
            # documents (``(cert_path, key_path)``). Validate its exact
            # shape, and reject tuples for any other key outright so a
            # future regression that accepts an unknown tuple does not
            # slip through and surface as a confusing botocore error.
            if isinstance(cfg_value, tuple):
                if cfg_key != "proxy_client_cert":
                    msg = (
                        "extra_boto_config['proxies_config']"
                        f"[{cfg_key!r}] does not accept a tuple value; "
                        "tuples are only valid for 'proxy_client_cert'."
                    )
                    raise TypeError(msg)
                if len(cfg_value) != 2 or not all(
                    isinstance(item, str) for item in cfg_value
                ):
                    msg = (
                        "extra_boto_config['proxies_config']"
                        "['proxy_client_cert'] tuple must be (cert_path, "
                        f"key_path) — exactly two strings; got {cfg_value!r}."
                    )
                    raise TypeError(msg)

    def _validate_s3_options(self) -> None:
        """Type-check ``extra_boto_config['s3']`` shape.

        botocore expects a dict of S3-specific transport options
        (``addressing_style``, ``use_accelerate_endpoint``, etc.).
        Mirroring the shape check applied to ``proxies`` /
        ``proxies_config`` keeps the validation surface symmetric so a
        non-dict value here surfaces as a clear ``TypeError`` at
        construction rather than as a confusing botocore error during
        the first request. Inner-key shapes are not deep-validated; the
        ``extra_boto_config`` contract continues to assume an
        operator-supplied trusted dict.
        """
        s3_options = self.extra_boto_config.get("s3")
        if s3_options is None:
            return
        if not isinstance(s3_options, dict):
            msg = (
                f"extra_boto_config['s3'] must be a dict of S3-specific "
                f"transport options, got {type(s3_options).__name__}."
            )
            raise TypeError(msg)

    def _audit_client_cert(self) -> None:
        """Emit an audit warning when ``proxy_client_cert`` is configured.

        ``proxies_config['proxy_client_cert']`` accepts an arbitrary
        local file path tuple — by design, since this surface is
        documented as operator-supplied trusted input. Logging that the
        value is set keeps the choice auditable post-deployment, but
        the raw paths are SHA-256 fingerprinted before logging so
        neither the certificate path nor (more sensitively) the
        private-key path is leaked to log aggregators where it could
        expose filesystem layout or hint at sensitive locations.
        """
        proxies_config = self.extra_boto_config.get("proxies_config")
        if not isinstance(proxies_config, dict):
            return
        client_cert = proxies_config.get("proxy_client_cert")
        if client_cert is None:
            return
        logger.warning(
            "S3BackendConfig: extra_boto_config['proxies_config']"
            "['proxy_client_cert'] is set (fingerprint=%s); operator-"
            "supplied path is forwarded to botocore as-is. Audit the "
            "source of this value if it was not configured intentionally.",
            _fingerprint_client_cert(client_cert),
        )


def build_client(config: S3BackendConfig) -> Any:
    """Construct a boto3 S3 client from configuration.

    The function combines the explicit timeout/retry/pool fields on
    ``config`` with the allow-listed ``extra_boto_config`` extras into
    a single :class:`botocore.config.Config`, then forwards the
    region/credentials/endpoint kwargs to :func:`boto3.client`. Keys in
    ``extra_boto_config`` that overlap with the explicit fields are
    silently dropped (logged at DEBUG) so the explicit attributes
    always win and we never raise ``TypeError`` from a duplicate
    keyword argument.

    Args:
        config: Validated :class:`S3BackendConfig`. Construction-time
            invariants (SSRF allow-list, prefix shape, allow-listed
            ``extra_boto_config`` keys) are assumed to have already
            been enforced by ``__post_init__``.

            **Precondition:** the ``config`` MUST have flowed through
            :meth:`S3BackendConfig.__post_init__`. Constructing the
            dataclass via ``S3BackendConfig(...)`` (or
            ``dataclasses.replace``) guarantees this. Bypassing
            construction — e.g. ``object.__new__(S3BackendConfig)``
            with hand-rolled ``__setattr__`` to circumvent
            ``frozen=True`` — would also bypass the SSRF allow-list
            and proxy validation, producing a client that can reach
            arbitrary network destinations. ``build_client`` does NOT
            re-run validation; do not call it on a hand-constructed
            instance.

    Returns:
        A boto3 S3 client. The type is reported as :class:`Any`
        because boto3's runtime client classes are generated and have
        no useful static type — operators who want sharper typing
        should wrap the return value themselves.
    """
    dropped = set(config.extra_boto_config) & _EXPLICIT_BOTO_KEYS
    if dropped:
        # Surface the override at WARNING (not DEBUG) so an operator
        # running at the default INFO level still sees that an
        # explicitly supplied ``extra_boto_config`` key (e.g. ``retries``)
        # was ignored in favor of the dedicated S3BackendConfig field —
        # silently dropping it at DEBUG hid configuration bugs in
        # practice.
        logger.warning(
            "extra_boto_config keys %s ignored — set the explicit "
            "S3BackendConfig field instead",
            sorted(dropped),
        )
    # Deep-copy the forwarded subset so botocore cannot observe a
    # later mutation of nested dicts (``proxies``, ``proxies_config``,
    # ``s3``) the caller still holds a reference to. ``__post_init__``
    # already deep-copied the outer dict on assignment, but a dict
    # comprehension only re-binds the top-level keys; the nested values
    # would still be aliased back to the validator's view. Copying
    # again here keeps the immutability contract intact across both
    # the validation surface and the boto3 client surface.
    extra_boto = copy.deepcopy(
        {
            k: v
            for k, v in config.extra_boto_config.items()
            if k not in _EXPLICIT_BOTO_KEYS
        }
    )
    boto_config = BotoConfig(
        retries={"max_attempts": config.max_retries, "mode": "adaptive"},
        connect_timeout=config.connect_timeout,
        read_timeout=config.read_timeout,
        max_pool_connections=config.max_pool_connections,
        **extra_boto,
    )
    kwargs: dict[str, Any] = {"config": boto_config}
    if config.region_name:
        kwargs["region_name"] = config.region_name
    if config.aws_access_key_id:
        kwargs["aws_access_key_id"] = config.aws_access_key_id
    if config.aws_secret_access_key:
        kwargs["aws_secret_access_key"] = config.aws_secret_access_key
    if config.aws_session_token:
        kwargs["aws_session_token"] = config.aws_session_token
    if config.endpoint_url:
        kwargs["endpoint_url"] = config.endpoint_url
    return boto3.client("s3", **kwargs)
