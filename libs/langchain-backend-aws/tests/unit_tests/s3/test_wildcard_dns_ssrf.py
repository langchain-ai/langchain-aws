"""SSRF guard for wildcard-DNS services that encode an IPv4 in the name.

``nip.io`` / ``sslip.io`` / ``xip.io`` resolve any IPv4 literal embedded
in the subdomain (``127-0-0-1.nip.io`` → ``127.0.0.1``). The literal-IP
allow-list cannot see through these because the resolution happens
inside the resolver, so the constructor rejects the entire suffix
family unless ``allow_private_endpoints=True`` opts in.
"""

from __future__ import annotations

import pytest

from langchain_backend_aws.s3._config import S3BackendConfig


class TestWildcardDnsSsrf:
    @pytest.mark.parametrize(
        "host",
        [
            "127-0-0-1.nip.io",
            "169-254-169-254.nip.io",
            "192-168-1-1.sslip.io",
            "10-0-0-1.xip.io",
            "anything.traefik.me",
            "evil.local-ip.sh",
        ],
    )
    def test_wildcard_dns_endpoint_rejected(self, host: str) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url=f"http://{host}/")

    @pytest.mark.parametrize(
        "host",
        [
            "127-0-0-1.nip.io",
            "192-168-1-1.sslip.io",
        ],
    )
    def test_wildcard_dns_proxy_rejected(self, host: str) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {"http": f"http://{host}/"}},
            )

    def test_wildcard_dns_allowed_with_opt_in(self) -> None:
        # ``allow_private_endpoints=True`` is the explicit escape hatch
        # for local-dev setups that rely on wildcard DNS to reach a
        # bound-on-loopback service.
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://127-0-0-1.nip.io/",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://127-0-0-1.nip.io/"

    def test_unrelated_public_domain_still_accepted(self) -> None:
        # The suffix list must not over-match — a public hostname that
        # merely contains ``.io`` or ``.me`` somewhere is unaffected.
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.io/")
        assert config.endpoint_url == "https://s3.example.io/"
