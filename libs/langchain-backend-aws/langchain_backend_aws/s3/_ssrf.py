"""SSRF guard helpers for :class:`S3BackendConfig`.

Extracted from ``_config.py`` so the dataclass surface — which is the
public API most callers touch — stays compact, and so the SSRF
allow-list logic can be unit-tested independently.

Both ``endpoint_url`` and ``extra_boto_config['proxies']`` flow through
:func:`validate_url_against_ssrf` so a misconfiguration cannot point
either transport at internal services (loopback, link-local IMDS,
RFC1918) without an explicit opt-in.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

# Allowed URL schemes for ``S3BackendConfig.endpoint_url`` and the
# proxies forwarded via ``extra_boto_config``. Only the transport-layer
# schemes for HTTPS/HTTP are accepted so a misuse like
# ``file:///etc/passwd`` or a custom-scheme reflective gadget is
# rejected at construction.
ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https"})

# Hostnames whose literal form indicates a private/loopback target. The
# scheme check alone does not block these (``http://localhost`` is a
# valid http URL), so a misconfiguration that pointed ``endpoint_url``
# at IMDS or a sidecar would slip through. Hostnames matched here are
# rejected unless ``allow_private=True`` is set explicitly.
#
# Cloud metadata service DNS names are included so a hostname like
# ``metadata.google.internal`` (which would otherwise resolve to
# ``169.254.169.254`` only at runtime) is rejected at construction.
# :func:`is_private_host` cannot resolve DNS names safely (no network
# at config time, and DNS resolution would be racy), so the
# literal-name check is the only line of defense for these.
PRIVATE_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        # GCE / Google Cloud metadata service.
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        # EC2 IMDS DNS alias (rarely used but documented).
        "metadata.ec2.internal",
    }
)

# Wildcard-DNS services that encode an arbitrary IPv4 literal into the
# subdomain (``127-0-0-1.nip.io`` → ``127.0.0.1``). The literal-IP guard
# in :func:`is_private_host` cannot see through these names because the
# resolution happens inside the resolver, not at validation time. We
# reject the entire suffix so an attacker cannot bypass the IP check by
# wrapping a private literal in a public DNS hostname. Any host whose
# normalized name ends with one of these suffixes is treated as private
# unless ``allow_private=True`` is set.
#
# Maintenance: this is a static allow-deny list — new wildcard-DNS
# services appear over time (e.g. ``localtest.me``,
# ``readymade.dev`` style services), so the suffixes here need a
# periodic review against the public landscape. Operators who already
# know they need an additional suffix today can wrap
# :func:`validate_url_against_ssrf` in their own pre-check rather than
# waiting on a release; we keep the list closed because expanding it
# at runtime via config would let agent-supplied data weaken the SSRF
# guard.
PRIVATE_HOSTNAME_SUFFIXES = frozenset(
    {
        ".nip.io",
        ".sslip.io",
        ".xip.io",
        ".traefik.me",
        ".local-ip.sh",
    }
)


def validate_url_against_ssrf(
    url: str,
    *,
    field_name: str,
    allow_private: bool,
) -> None:
    """Reject URLs whose scheme or host would expose internal services.

    Shared between ``endpoint_url`` and ``extra_boto_config['proxies']``
    so both inputs apply the same allow-list. Without parity, a proxy
    pointed at IMDS would re-introduce the SSRF surface that
    ``endpoint_url`` already blocks.

    Args:
        url: The candidate URL string.
        field_name: Human-readable name shown in error messages.
        allow_private: When ``True``, loopback/link-local/RFC1918 hosts
            are permitted (LocalStack/MinIO opt-in).

    Raises:
        ValueError: If the scheme is not http/https, the URL has no
            hostname, or the hostname resolves to a private/loopback/
            link-local literal.
    """
    # Reject CRLF (and other ASCII control characters) up front.
    # ``repr()`` already escapes these in the error messages below, so
    # log injection is not exploitable today; rejecting at construction
    # is defense-in-depth so a future code path that interpolates the
    # URL via ``str()`` cannot be silently weakened. CR/LF in a URL
    # string is also the splitting primitive for HTTP request smuggling
    # against a misbehaving proxy, so refusing them keeps the transport
    # surface clean regardless of the logging story.
    if any(ch in url for ch in ("\r", "\n", "\x00")):
        msg = (
            f"{field_name} {url!r} contains a control character "
            "(CR/LF/NUL); refusing to use it as a URL."
        )
        raise ValueError(msg)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_ENDPOINT_SCHEMES:
        msg = (
            f"{field_name} scheme {scheme!r} is not allowed. "
            "Only http/https URLs are accepted; this field must never "
            "be sourced from untrusted input (SSRF risk)."
        )
        raise ValueError(msg)
    raw_host = (parsed.hostname or "").lower()
    # IDNA-normalise so a homoglyph host (e.g. ``nip․io`` —
    # U+2024 ONE DOT LEADER instead of ``.``) cannot bypass the suffix
    # match against ``PRIVATE_HOSTNAME_SUFFIXES``: the resolver would
    # ultimately follow the lookalike to its real ASCII form, but the
    # naive ``.lower()`` form preserves the lookalike byte-for-byte and
    # would slip through ``host.endswith('.nip.io')``.
    #
    # IDNA is only applied to hosts that actually contain non-ASCII
    # characters: ``encode('idna')`` raises ``UnicodeError`` on IPv4/IPv6
    # literals, trailing-dot FQDNs, 63+ char labels, and empty-label
    # forms (``foo..bar``). Falling back to raw_host on those benign
    # ASCII shapes would weaken the suffix match for valid inputs, so
    # we route ASCII hosts straight to the literal/IP checks below and
    # only invoke IDNA when normalization is actually needed.
    if raw_host and not raw_host.isascii():
        try:
            host = raw_host.encode("idna").decode("ascii").lower()
        except (UnicodeError, UnicodeDecodeError):
            # Non-encodable Unicode host (control chars, empty labels):
            # fall back to raw_host so the IP-literal / ``PRIVATE_HOSTNAMES``
            # checks below still run. The suffix check will not match a
            # homoglyph form here, but rejecting any non-encodable host
            # upfront would break valid edge cases; ``is_private_host``
            # remains the last line of defense.
            host = raw_host
    else:
        host = raw_host
    if not host:
        # Reject early instead of relying on boto3 to surface a confusing
        # downstream error. A URL with no host (``http://`` or
        # ``https:///path``) cannot route to anything legitimate; failing
        # closed here makes the misconfiguration obvious at construction.
        msg = (
            f"{field_name} {url!r} is missing a hostname. "
            "Only fully-qualified http/https URLs are accepted."
        )
        raise ValueError(msg)
    if allow_private:
        return
    if (
        host in PRIVATE_HOSTNAMES
        or _matches_private_suffix(host)
        or is_private_host(host)
    ):
        msg = (
            f"{field_name} host {host!r} resolves to a private, "
            "loopback, or link-local address. Pass "
            "allow_private_endpoints=True to opt in (LocalStack/MinIO) "
            "or use a public endpoint; the default fails closed to "
            "block accidental SSRF (e.g. IMDS at 169.254.169.254)."
        )
        raise ValueError(msg)


def _matches_private_suffix(host: str) -> bool:
    """Return ``True`` if ``host`` ends with a known wildcard-DNS suffix.

    These services (``nip.io``, ``sslip.io``, ``xip.io``, …) resolve any
    IPv4 literal embedded in the subdomain (``192-168-1-1.nip.io`` →
    ``192.168.1.1``), so a literal-IP allow-list alone cannot block
    them at validation time. Rejecting the entire suffix family means
    even a hostname whose IP we cannot inspect at construction is
    refused unless the operator opts in via ``allow_private=True``.
    """
    return any(host.endswith(suffix) for suffix in PRIVATE_HOSTNAME_SUFFIXES)


def is_private_host(host: str) -> bool:
    """Return ``True`` if ``host`` is an IP literal in a private range.

    Hostnames that are not IP literals (DNS names) are not resolved
    here; we cannot make a network call at config-construction time, and
    DNS resolution in :mod:`__post_init__` would also be racy. Known
    cloud-metadata DNS aliases (``metadata.google.internal``,
    ``metadata.ec2.internal`` etc.) are listed in :data:`PRIVATE_HOSTNAMES`
    so the canonical SSRF targets are rejected even without resolution;
    operators pointing ``endpoint_url`` at any other DNS name that
    resolves to a private range must set ``allow_private_endpoints=True``
    explicitly.

    IPv6 normalization caveats handled here:

    - ``urlparse(...).hostname`` already strips bracket syntax, so the
      ``[::1]`` form arrives as ``::1``. We still ``strip("[]")`` for
      callers that pass a raw literal.
    - Zone identifiers (``fe80::1%eth0``) make :func:`ipaddress.ip_address`
      raise ``ValueError``. Splitting on ``%`` keeps the literal part so
      link-local addresses with a zone ID are still classified as
      private.
    - IPv4-mapped IPv6 (``::ffff:127.0.0.1``) is not flagged as
      ``is_loopback`` on older CPython releases. Re-checking against the
      embedded :class:`IPv4Address` closes that gap deterministically.
    """
    candidate = host.strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
