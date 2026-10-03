"""Default values and allow-lists for :class:`S3BackendConfig`.

Extracted from ``_config.py`` so the dataclass surface that callers
read stays compact. The rationale for each cap is preserved here as
the load-bearing ReDoS / SSRF / blast-radius reasoning — moving them
keeps ``_config.py`` close to its 800-line cap without losing the
context. Importers should treat every name here as part of the
package's internal API; only :class:`S3BackendConfig` field defaults
and :func:`build_client` reach the public surface.
"""

from __future__ import annotations

# Maximum number of objects enumerated by grep before stopping. The cap
# is applied to the raw ``ListObjectsV2`` stream — any object pulled
# from the listing counts against the cap, including ones later skipped
# by the glob filter or the size cap. This makes the cap a simple
# enumeration bound rather than a "matched object" bound.
DEFAULT_GREP_MAX_OBJECTS = 10_000

# Maximum object size considered for grep, in bytes. Larger objects are
# skipped to avoid memory exhaustion.
DEFAULT_GREP_MAX_FILE_SIZE = 5 * 1024 * 1024

# Maximum number of objects enumerated by glob before stopping. Prevents
# unbounded enumeration on large buckets (parallel to ``grep_max_objects``).
DEFAULT_GLOB_MAX_OBJECTS = 10_000

# Maximum number of objects enumerated by ls before stopping. Prevents
# unbounded enumeration on large buckets when a single prefix holds many
# direct children.
DEFAULT_LS_MAX_OBJECTS = 10_000

# Maximum file size in MiB read into memory by ``read``, ``edit`` and
# ``download_files``. Mirrors deepagents' ``FilesystemBackend.max_file_size_mb``
# default and bounds memory pressure when an agent references an
# unexpectedly large object.
DEFAULT_MAX_FILE_SIZE_MB = 10

# Upper bound on the length of the regex source string accepted by grep.
# Caller-supplied patterns reach ``re.compile`` directly, so without a
# length cap a long crafted pattern can drive catastrophic backtracking
# (ReDoS). 1000 chars comfortably covers realistic search expressions
# while denying the obviously-pathological ones.
DEFAULT_GREP_MAX_PATTERN_LENGTH = 1_000

# Skip lines longer than this many bytes during grep. Long lines are the
# main amplifier of ReDoS — backtracking cost grows super-linearly with
# input length — so refusing to feed them to ``re.search`` keeps
# worst-case CPU bounded even when the pattern itself slipped through
# the length cap. Long-line files (minified JS, single-line JSON dumps)
# are skipped silently because the grep contract has no
# "skipped, line too long" tag.
DEFAULT_GREP_MAX_LINE_LENGTH = 100_000

# Upper bound on the length of glob patterns accepted by ``glob`` and by
# the optional ``glob`` filter on ``grep``. Glob patterns are translated
# to anchored regex (via :func:`make_glob_compiler`); without a length
# cap a long crafted pattern with stacked ``*`` runs can compile into a
# regex with many overlapping quantifiers and amplify backtracking on
# match. 1000 chars covers realistic shell-style globs while denying
# pathological inputs.
DEFAULT_GLOB_MAX_PATTERN_LENGTH = 1_000

# Maximum count of regex metacharacters in a grep pattern. Bounds compile
# cost and nested-quantifier ReDoS surface even when the pattern slips
# under ``grep_max_pattern_length``. Counts the union of
# ``( ) [ ] { } * + ? | \``: realistic patterns rarely cross 50, so 200
# leaves headroom while denying the obviously stacked forms.
DEFAULT_GREP_MAX_PATTERN_METACHARS = 200

# Default thread-pool size for parallel ``download_files`` fetches. Bounded
# by ``max_pool_connections`` at runtime so we never exceed the boto3
# connection pool.
DEFAULT_DOWNLOAD_CONCURRENCY = 8

# Hard wall-clock cap (in seconds) on each ``regex.search`` call inside
# grep. Pattern-length and line-length caps reduce the chance of
# catastrophic backtracking, but neither bounds runtime when both inputs
# slip through. The ``regex`` package's ``timeout`` argument raises
# ``TimeoutError`` once the budget is exceeded, so a worker cannot be
# pinned indefinitely by a crafted pattern such as ``(a+)+$``.
DEFAULT_GREP_REGEX_TIMEOUT = 1.0

# Maximum count of glob wildcards (``*`` and ``?``) accepted in glob
# patterns and in the ``glob`` filter on ``grep``. Bounds the number of
# stacked ``[^/]*`` runs in the translated regex — the catastrophic
# backtracking surface for glob — independently of the source-length
# cap. Realistic globs rarely cross 10 wildcards; 50 leaves headroom.
DEFAULT_GLOB_MAX_PATTERN_METACHARS = 50

# Hard wall-clock cap (in seconds) on each glob regex match. Mirrors
# ``grep_regex_timeout`` but applied to the glob translation step.
# Without this a stacked-wildcard pattern like ``****...****`` (under
# the length cap) compiled to ``[^/]*[^/]*...`` would pin a worker
# during the ``Pattern.match`` call.
DEFAULT_GLOB_REGEX_TIMEOUT = 1.0

# Keys in ``extra_boto_config`` that would otherwise duplicate the
# explicit attributes managed on ``S3BackendConfig``. Filtered out
# before forwarding so explicit fields always win and we never raise
# ``TypeError`` from passing the same kwarg twice.
EXPLICIT_BOTO_KEYS = frozenset(
    {"retries", "connect_timeout", "read_timeout", "max_pool_connections"}
)

# Allowlist of ``extra_boto_config`` keys forwarded to
# :class:`botocore.config.Config`. Constraining the surface keeps the
# integration boundary narrow — botocore's config grows over time and
# we do not want a typo (or untrusted dict) to silently change
# transport behavior. Operators who need a key outside this set should
# build their own ``BotoConfig`` and pass a pre-built ``client=`` to
# :class:`S3Backend`.
#
# ``user_agent_extra`` is permitted; the full ``user_agent`` override is
# intentionally excluded because the parent ``langchain-aws`` package
# patches the boto3 client with a framework user-agent header for AWS
# attribution. Accepting a full replacement here would let a stray
# config silently strip that identifier. Operators who genuinely need a
# different base user-agent should build their own ``BotoConfig`` and
# pass a pre-built ``client=`` to :class:`S3Backend`.
ALLOWED_BOTO_KEYS = frozenset(
    {
        "signature_version",
        "s3",
        "proxies",
        "proxies_config",
        "user_agent_extra",
        "client_cert",
        "tcp_keepalive",
        "parameter_validation",
        "inject_host_prefix",
        "endpoint_discovery_enabled",
        "request_min_compression_size_bytes",
        "disable_request_compression",
    }
)

# Allowlist of inner ``extra_boto_config['proxies_config']`` keys forwarded
# to botocore. Mirrors the outer ``ALLOWED_BOTO_KEYS`` discipline so a
# typo or an injected unknown key surfaces as a clear ``ValueError``
# rather than as a silently-ignored mapping that surprises operators
# at first request. Keys here match botocore's documented surface for
# ``proxies_config``: ``ca_bundle``, ``client_cert``, and
# ``proxy_use_forwarding_for_https``. Operators who need an option
# outside this set should build their own ``BotoConfig`` and pass a
# pre-built ``client=`` to :class:`S3Backend`.
ALLOWED_PROXIES_CONFIG_KEYS = frozenset(
    {
        "proxy_ca_bundle",
        "proxy_client_cert",
        "proxy_use_forwarding_for_https",
    }
)
