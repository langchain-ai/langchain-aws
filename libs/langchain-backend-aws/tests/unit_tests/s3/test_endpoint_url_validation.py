"""Unit tests for ``S3BackendConfig`` endpoint_url scheme validation.

The field's docstring already warns it must never come from untrusted
input (SSRF). The constructor enforces an http/https allowlist as
defense-in-depth so a misuse like ``file://`` or a non-HTTP URI is
rejected close to the construction site.
"""

from __future__ import annotations

import pytest

from langchain_backend_aws.s3._config import S3BackendConfig


class TestEndpointUrlValidation:
    def test_https_endpoint_accepted(self) -> None:
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.com")
        assert config.endpoint_url == "https://s3.example.com"

    def test_http_endpoint_accepted(self) -> None:
        # Public hostname over http is accepted without opt-in. MinIO/
        # LocalStack on localhost requires ``allow_private_endpoints``;
        # see ``test_localhost_requires_opt_in`` below.
        config = S3BackendConfig(bucket="b", endpoint_url="http://s3.example.com:9000")
        assert config.endpoint_url == "http://s3.example.com:9000"

    def test_localhost_rejected_by_default(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://localhost:9000")

    def test_localhost_accepted_with_opt_in(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://localhost:9000",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://localhost:9000"

    def test_imds_address_rejected(self) -> None:
        # 169.254.169.254 is the IMDS endpoint on EC2; mistakenly
        # pointing the client at it is the canonical SSRF case.
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b", endpoint_url="http://169.254.169.254/latest/meta-data"
            )

    def test_loopback_ip_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://127.0.0.1:9000")

    def test_rfc1918_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://10.0.0.1:9000")

    def test_ipv6_loopback_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[::1]:9000")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="file:///etc/passwd")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="ftp://example.com")

    def test_empty_scheme_rejected(self) -> None:
        # ``urlparse('localhost:9000').scheme`` is ``''`` — no implicit
        # http; reject so callers cannot omit the scheme by accident.
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="//example.com/")

    def test_endpoint_url_none_is_fine(self) -> None:
        config = S3BackendConfig(bucket="b")
        assert config.endpoint_url is None

    # ------------------------------------------------------------------
    # Metadata-service DNS aliases. ``_is_private_host`` cannot resolve
    # DNS names safely at construction time, so the literal-name
    # allowlist is the only line of defense against pointing the client
    # at IMDS via its hostname (instead of the raw ``169.254.169.254``).
    # These tests pin that allowlist as a regression guard — adding a
    # new metadata alias to a cloud provider should also land here.
    # ------------------------------------------------------------------

    def test_gce_metadata_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata.google.internal/")

    def test_gce_metadata_short_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata/")

    def test_gce_metadata_goog_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata.goog/")

    def test_ec2_metadata_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata.ec2.internal/")

    def test_metadata_dns_accepted_with_opt_in(self) -> None:
        # ``allow_private_endpoints=True`` is the documented escape
        # hatch; if an operator opts in we trust the destination.
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://metadata.google.internal/",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://metadata.google.internal/"

    # ------------------------------------------------------------------
    # IPv6 corner cases. ``urlparse(...).hostname`` strips bracket
    # syntax, but zone identifiers (``%eth0``) and IPv4-mapped IPv6
    # (``::ffff:127.0.0.1``) used to slip past ``ipaddress.ip_address``
    # / ``is_loopback``. Pin the defenses so a regression surfaces.
    # ------------------------------------------------------------------

    def test_ipv6_zone_id_loopback_rejected(self) -> None:
        # ``%25`` is the percent-encoded zone-id separator. The host
        # arrives as ``::1%eth0`` after ``urlparse``; without zone
        # stripping ``ipaddress.ip_address`` would raise ``ValueError``
        # and ``_is_private_host`` would return ``False``, letting the
        # loopback target slip through.
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[::1%25eth0]:9000")

    def test_ipv6_zone_id_link_local_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[fe80::1%25eth0]:9000")

    def test_ipv4_mapped_ipv6_loopback_rejected(self) -> None:
        # ``::ffff:127.0.0.1`` is IPv4-mapped IPv6 pointing at the v4
        # loopback. Some CPython releases do not flag this as
        # ``is_loopback`` directly; we re-check via ``ipv4_mapped`` to
        # be safe.
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[::ffff:127.0.0.1]:9000")

    def test_ipv4_mapped_ipv6_imds_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                endpoint_url="http://[::ffff:169.254.169.254]/latest/meta-data",
            )

    @pytest.mark.parametrize(
        "url",
        [
            "https://s3.example.com\r\nHost: evil.example.com",
            "https://s3.example.com\nX-Injected: true",
            "https://s3.example.com\rfoo",
            "https://s3.example.com/\x00",
        ],
    )
    def test_control_characters_rejected(self, url: str) -> None:
        # Defense-in-depth: CR/LF/NUL in a URL string is the splitting
        # primitive for HTTP request smuggling and log injection. They
        # must be rejected before ``urlparse`` to keep the transport
        # surface clean even if a future caller stringifies the URL via
        # ``str()`` instead of ``repr()``.
        with pytest.raises(ValueError, match="control character"):
            S3BackendConfig(bucket="b", endpoint_url=url)

    def test_unknown_dns_name_passes(self) -> None:
        # We deliberately do not resolve arbitrary DNS names; a public
        # hostname slips through. Operators pointing at a private DNS
        # name must opt in via ``allow_private_endpoints=True``. This
        # behavior is documented in the README "Endpoint URL and SSRF"
        # section — pin it here so a future change to add DNS
        # resolution surfaces in CI.
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.com")
        assert config.endpoint_url == "https://s3.example.com"
