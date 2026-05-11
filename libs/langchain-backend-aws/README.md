# langchain-backend-aws

Amazon S3 backend for [Deep Agents](https://github.com/langchain-ai/deepagents).

Stores agent files as S3 objects (raw bytes) so they remain directly readable
from the S3 Console and downstream tools.

## Design guidelines (Deep Agents virtual filesystem)

This package implements a "virtual filesystem" backend that projects S3 into
the Deep Agents tools namespace. Per the upstream
[Deep Agents backends guide][backends-guide], custom virtual-FS backends
should follow these rules — `S3Backend` is built against them and any
fork/extension should preserve them:

- **Paths are absolute** (`/x/y.txt`). Decide how to map them to your
  storage keys/rows. `S3Backend` maps `/x/y.txt` to
  `<prefix>x/y.txt` after rejecting traversal (`..`, repeated `/`,
  Windows drive letters).
- **Implement `ls` and `glob` efficiently.** Prefer server-side
  filtering where available, otherwise local filter. `S3Backend` uses
  `ListObjectsV2` with `Prefix` (and `Delimiter='/'` for `ls`) and
  applies the glob regex client-side.
- **External persistence returns no `files_update`.** For S3 / Postgres /
  any out-of-process store, return `files_update=None` (Python) or omit
  `filesUpdate` (JS) in `write`/`edit` results — only in-memory state
  backends need to return a files-update dict. `S3Backend` does this.
- **Use `ls` and `glob` as the method names.** Do not rename them; the
  Deep Agents tool wiring expects exactly those identifiers.
- **Return structured result types with an `error` field; do not raise.**
  Missing files, invalid patterns, prefix violations, oversize bodies,
  and S3 transient/denial responses all surface as
  `*Result(error=...)` strings, never as exceptions across the
  protocol boundary. Callers can branch on `result.error` instead of
  wrapping every call in `try`/`except`.

[backends-guide]: https://docs.langchain.com/oss/python/deepagents/backends#use-a-virtual-filesystem

## Installation

```bash
pip install langchain-backend-aws
```

## Available backends

| Backend | Import path | Status |
| --- | --- | --- |
| S3 | `langchain_backend_aws.s3` | Available |

Top-level re-exports also work for convenience
(`from langchain_backend_aws import S3Backend`).

## Usage (S3)

```python
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
```

For ad-hoc construction without importing `S3BackendConfig`, use
`S3Backend.from_kwargs(...)`:

```python
backend = S3Backend.from_kwargs(bucket="my-agent-files", region_name="us-west-2")
```

The constructor only accepts a fully-built `S3BackendConfig`. Use
`S3Backend.from_kwargs(...)` for ad-hoc construction; the bare
`S3Backend(bucket=...)` form is not supported, so the configuration
validation surface always lives at `S3BackendConfig` (or `from_kwargs`)
and stays visible in the import list.

## Configuration

| Parameter | Description |
| --- | --- |
| `bucket` | S3 bucket name (required) |
| `prefix` | Key prefix for all operations. **Required in production** for tenant isolation; an empty prefix grants the backend full-bucket access (every `ls` / `glob` / `grep` enumerates the entire bucket) and emits a warning at construction. Always pair production deployments with `require_prefix=True`. |
| `require_prefix` | When `True`, raises `ValueError` at construction if `prefix` is empty. Defaults to `False` only for backwards compatibility with single-tenant local-dev setups; **set to `True` in every production deployment** so a config typo fails closed rather than silently exposing the whole bucket. |
| `region_name` | AWS region |
| `aws_access_key_id` | AWS access key ID |
| `aws_secret_access_key` | AWS secret access key |
| `aws_session_token` | AWS session token |
| `endpoint_url` | Custom S3 endpoint URL (LocalStack/MinIO). **Must never be sourced from untrusted user input** — accepting an arbitrary URL would let a caller redirect S3 traffic to internal network services (SSRF). Treat this as a deployment-time configuration value only. |
| `max_retries` | boto3 max retry attempts (default 3). The retry *mode* is fixed at `"adaptive"`; `extra_boto_config['retries']` is silently dropped so the explicit field always wins. Build your own `botocore.config.Config` if you need a different mode. |
| `connect_timeout` | boto3 connect timeout in seconds (default 5.0) |
| `read_timeout` | boto3 read timeout in seconds (default 30.0) |
| `max_pool_connections` | boto3 connection pool size (default 50) |
| `grep_max_objects` | Max objects enumerated per grep call from `ListObjectsV2` (default 10000). The cap counts every listed object — including ones later filtered out by `glob` or skipped for exceeding `grep_max_file_size` — so it bounds list traffic, not match attempts. Fails closed with an error if exceeded — same contract as `glob_max_objects` / `ls_max_objects`. |
| `grep_max_file_size` | Max file size considered by grep (default 5 MiB) |
| `glob_max_objects` | Max objects enumerated per glob call (default 10000). When the scan exceeds this limit, `glob` returns an error instead of partial matches to avoid callers concluding a file does not exist when it merely was not scanned. Narrow `path` or raise this limit. |
| `ls_max_objects` | Max objects enumerated per ls call (default 10000). Same fail-closed semantics as `glob_max_objects`. |
| `max_file_size_mb` | Max file size in MiB that `read`, `edit`, `upload_files`, and `download_files` will load into memory or accept on the wire (default 10). Larger objects fail with an error rather than risking OOM. Mirrors deepagents' `FilesystemBackend.max_file_size_mb`. |
| `grep_max_pattern_length` | Max length of the user-supplied regex pattern accepted by `grep` (default 1000). Patterns longer than this are rejected before reaching `re.compile`, limiting the surface for catastrophic backtracking (ReDoS). |
| `grep_max_line_length` | Skip lines longer than this many bytes during `grep` (default 100000). Long lines are the dominant ReDoS amplifier; refusing to feed them to `re.search` keeps worst-case CPU bounded. |
| `glob_max_pattern_length` | Max length of glob patterns accepted by `glob` and by the optional `glob` filter on `grep` (default 1000). Patterns longer than this are rejected before reaching the glob translator, limiting the surface for stacked-quantifier backtracking on match. |
| `grep_regex_timeout` | Wall-clock cap (seconds) on each `regex.search` call inside `grep` (default 1.0). Pattern/line-length caps reduce ReDoS risk but cannot bound runtime alone; this hard cap raises `TimeoutError` once the budget is hit so a crafted pattern (e.g. `(a+)+$`) cannot pin a worker. Surfaces as a fail-closed grep error, never as an exception across the protocol boundary. |
| `grep_max_pattern_metachars` | Maximum count of regex metacharacters (`()[]{}*+?\|\`) in the user-supplied grep pattern (default 200). Caps compile cost and nested-quantifier ReDoS surface even when the source string is shorter than `grep_max_pattern_length`. |
| `binary_read_mode` | How `read` handles non-UTF-8 bodies. `"base64"` (default) returns the full body base64-encoded; `"error"` returns an error directing the caller to `download_files`, avoiding several MiB of base64 in the agent context window. |
| `download_concurrency` | Maximum number of concurrent `download_files` fetches (default 8). Capped at runtime by `max_pool_connections`, so a `max_pool_connections=1` deployment forces sequential fetches even when `download_concurrency` is higher. Set to `1` explicitly to opt out of the thread pool entirely. |
| `extra_boto_config` | Extra kwargs forwarded to `botocore.config.Config` (allowlist: `signature_version`, `s3`, `proxies`, `proxies_config`, `user_agent`, `user_agent_extra`, `client_cert`, `tcp_keepalive`, `parameter_validation`, `inject_host_prefix`, `endpoint_discovery_enabled`, `request_min_compression_size_bytes`, `disable_request_compression`). `proxies` URLs go through the same SSRF allow-list as `endpoint_url` (http/https only; loopback / link-local / RFC1918 hosts rejected unless `allow_private_endpoints=True`). |
| `allow_private_endpoints` | When `True`, both `endpoint_url` and `extra_boto_config['proxies']` may point at loopback / RFC1918 / link-local hosts (LocalStack, MinIO). Defaults to `False` so a misconfigured endpoint cannot reach IMDS or a sidecar. |

## Supported operations

| Operation | Notes |
| --- | --- |
| `ls` | Uses `Delimiter='/'` for non-recursive listing |
| `read` | Line-based pagination; binary files returned as base64 (with `encoding="base64"` set on `file_data` — `offset`/`limit` do not apply to the base64 fallback, which always returns the full body). Bounded by `max_file_size_mb`. |
| `write` | Atomic create via `PutObject` with `IfNoneMatch="*"`; returns an error if the key already exists, even under concurrent writers. |
| `edit` | Optimistic concurrency via `PutObject` with `IfMatch=<ETag>`; returns a conflict error if the file changed between read and write. Bounded by `max_file_size_mb`. |
| `glob` | Recursive enumeration with custom `**`/`*`/`?` matcher. Patterns without `/` (e.g. `*.py`) also match against the basename, so they find files at any depth; patterns with `/` (e.g. `src/*.py`) are strict against the relative path. Only the `**/` form expands to "zero or more path segments"; `**` not followed by `/` is treated as a single `*` (does not cross `/`) — use `**/` explicitly for recursive matching. |
| `grep` | Regex search across listed objects, with optional `glob` filter sharing `glob`'s syntax (including basename fallback). Fails closed when `grep_max_objects` is exceeded; treats per-object `NoSuchKey` (deleted between list and get) as a benign skip. |
| `upload_files` / `download_files` | Per-file partial-success semantics |

### Known limitations

- **`read` normalizes line terminators; `edit` preserves them.** `read`
  splits the body with `str.splitlines` and joins selected lines with
  `\n`, so `\r\n` / `\r` terminators and any trailing newline are
  collapsed to `\n` in the returned string. `edit` decodes the bytes,
  performs the string replacement, and re-encodes the result without
  going through `splitlines`, so original terminators outside the
  replaced span are preserved byte-for-byte. Callers that need
  byte-exact reads should use `download_files`.

- **`oversize` is reported as `permission_denied` for upload/download.**
  deepagents' `FileOperationError` Literal does not include an
  `"oversize"` member, so when `upload_files` or `download_files` rejects
  a file for exceeding `max_file_size_mb` the response is set to
  `error="permission_denied"`. The actual cause (file path and byte
  count) is logged at ERROR. Tracked as a TODO for an upstream change in
  deepagents.

  **Triage tip:** `permission_denied` from `upload_files` /
  `download_files` is overloaded — it may signal genuine `AccessDenied`,
  a transient 5xx from S3, *or* an oversize rejection. The ERROR-level
  log line (look for `S3 upload refused` / `S3 download refused` or the
  per-code `S3 upload/download failed ...`) carries the real cause.
  Filter logs by `langchain_backend_aws.s3` before assuming an IAM
  problem.

### Memory profile

`read`, `edit`, `download_files`, and `grep` materialise object bodies
into memory up to `max_file_size_mb` (default 10 MiB) per object. The
practical resident-memory ceiling is roughly:

```
peak ≈ download_concurrency × max_file_size_mb       # download_files
peak ≈ max_file_size_mb                              # read / edit
peak ≈ 1 × grep_max_file_size                        # grep (sequential per object)
```

Defaults give `8 × 10 MiB = 80 MiB` for `download_files`. Tune
`download_concurrency`, `max_file_size_mb`, and `grep_max_file_size`
together if you need a tighter envelope.

### Glob/grep regex cache

`glob` patterns are translated to anchored regex via a process-global
`functools.lru_cache(maxsize=256)`. The translation is a pure function
of the pattern string, so the cache is safely shared across `S3Backend`
instances and tenants — the same pattern always compiles to the same
regex.

The trade-off is eviction-based: a tenant cycling through many distinct
one-off patterns can evict another tenant's hot pattern, costing the
victim a re-compile on the next match. This affects throughput, not
correctness. The `glob_max_pattern_length` and
`glob_max_pattern_metachars` caps bound per-entry size, so an attacker
cannot inflate cache footprint to evict more than ~256 entries.

### Endpoint URL and SSRF

`endpoint_url` is constructor-validated. The same checks are also
applied to every URL value in `extra_boto_config['proxies']` so a
proxy cannot re-introduce the SSRF surface that `endpoint_url`
already blocks.

- only `http`/`https` schemes are accepted (so `file://`,
  `gopher://`, …  are rejected at construction);
- IP-literal hosts in loopback / link-local (IMDS `169.254.0.0/16`) /
  RFC1918 / multicast / reserved ranges are rejected;
- known cloud-metadata DNS aliases (`metadata.google.internal`,
  `metadata.ec2.internal`, `metadata.goog`, `metadata`) are rejected
  by literal-name match;
- wildcard-DNS services that encode an IPv4 literal in the subdomain
  (`*.nip.io`, `*.sslip.io`, `*.xip.io`, `*.traefik.me`,
  `*.local-ip.sh`) are rejected by suffix match. These are the
  obvious bypasses for the literal-IP check — `127-0-0-1.nip.io`
  resolves to `127.0.0.1` only inside the resolver, so without the
  suffix guard a config like `endpoint_url=http://127-0-0-1.nip.io/`
  would slip through. The suffix list is static: new wildcard-DNS
  services appear over time, so operators should periodically review
  `PRIVATE_HOSTNAME_SUFFIXES` in `_ssrf.py` against the public landscape
  and submit additions as a regular PR. Runtime extension via config is
  intentionally not supported — letting agent-supplied data widen the
  list would defeat the guard.

The configuration layer **does not resolve arbitrary DNS names** — a
config-time DNS lookup would be racy, and a custom DNS server could
return a public address at validation time and a private one later.
If you point `endpoint_url` at a private DNS name (LocalStack on
`localstack.svc.cluster.local`, MinIO on a Compose hostname) you must
opt in with `allow_private_endpoints=True` and accept responsibility
for the destination. Never source `endpoint_url` from user input.

DNS rebinding (a host that resolves to a public address at validation
time and a private one when boto3 later connects) is **out of scope**
for the constructor check — the validator never resolves DNS, so a
public name pointing at a private IP is only catchable by inspecting
the IP at connect time, which boto3 does not expose. If your threat
model includes DNS rebinding, pin `endpoint_url` to a known IP literal
(which the validator inspects directly) or front the egress path with
a network-layer allowlist.

### S3 compatibility

`write` and `edit` rely on conditional `PutObject` (`IfNoneMatch` / `IfMatch`),
GA on AWS S3 since November 2024. MinIO and LocalStack are supported only as
validation environments; they may handle preconditions differently on older
builds, so production deployments should target real S3.

`edit` additionally requires that `GetObject` return a non-empty `ETag`
header so the read-modify-write cycle can pass it back as `IfMatch`. AWS S3
always returns one. S3-compatible stores that omit `ETag` (some legacy
MinIO builds, custom proxies) will see every `edit` call fail closed with
"S3 did not return an ETag" — the backend refuses to fall back to an
unconditional PUT because that would silently overwrite a concurrent
writer's change. If you must run against such a store, plan for `write`
+ delete instead of `edit`.

| Capability | Required of S3-compatible target |
| --- | --- |
| `write` | Conditional `PutObject` with `IfNoneMatch="*"` (returns `PreconditionFailed` / `412` on conflict) |
| `edit` | Same as `write`, plus a non-empty `ETag` on `GetObject` for the `IfMatch` precondition |
| `glob` / `grep` / `ls` | `ListObjectsV2` honoring the requested `Prefix`. The backend fails closed if a paginator page returns a key outside the configured prefix, so misbehaving stores or proxies surface as an error rather than a partial result. |
| `read` / `download_files` | `GetObject` body that respects the requested object's `ContentLength` (the backend caps both header and body bytes against `max_file_size_mb`). |

## Testing

### Unit tests

Unit tests use a mocked boto3 client and run offline in milliseconds:

```bash
make tests
```

### Integration tests

The integration suite exercises the backend against a live S3-compatible
endpoint. Both real AWS S3 and any S3-compatible store (MinIO, LocalStack)
are supported.

The suite auto-skips when credentials are absent, so the default
`make integration_tests` is safe to run in any environment.

#### Option A: MinIO (recommended for local development)

1. Start MinIO with Docker:

   ```bash
   docker run -d --name minio \
       -p 9000:9000 -p 9001:9001 \
       -e MINIO_ROOT_USER=minioadmin \
       -e MINIO_ROOT_PASSWORD=minioadmin \
       minio/minio:latest server /data --console-address ":9001"
   ```

2. Export the test environment variables:

   ```bash
   export S3_BACKEND_ENDPOINT_URL=http://localhost:9000
   export S3_BACKEND_ACCESS_KEY=minioadmin
   export S3_BACKEND_SECRET_KEY=minioadmin
   export S3_BACKEND_BUCKET=langchain-backend-aws-tests
   ```

3. Run the suite:

   ```bash
   make integration_tests
   ```

The bucket is auto-created on first run. Each test uses a unique key
prefix and cleans up after itself, so the bucket can be reused across
runs.

#### Option B: Real AWS S3

```bash
export S3_BACKEND_ACCESS_KEY=AKIA...
export S3_BACKEND_SECRET_KEY=...
export S3_BACKEND_BUCKET=your-test-bucket
export S3_BACKEND_REGION=us-west-2
# S3_BACKEND_ENDPOINT_URL must be unset

make integration_tests
```

The bucket must exist and the credentials must have
`s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`
permissions on it.

#### Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `S3_BACKEND_ACCESS_KEY` | yes | — | AWS / MinIO access key |
| `S3_BACKEND_SECRET_KEY` | yes | — | AWS / MinIO secret key |
| `S3_BACKEND_BUCKET` | no | `langchain-backend-aws-tests` | Bucket name |
| `S3_BACKEND_ENDPOINT_URL` | no | — | Custom endpoint (set for MinIO/LocalStack) |
| `S3_BACKEND_REGION` | no | `us-east-1` | AWS region |
