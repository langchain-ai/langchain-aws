"""Unit tests for ``extra_boto_config['proxies']`` SSRF validation.

``proxies`` URLs are forwarded to botocore's transport layer, so a
proxy pointed at IMDS or a sidecar would re-introduce the SSRF surface
that ``endpoint_url`` already blocks. The constructor applies the same
allow-list to every value in ``proxies`` so the two SSRF entry points
stay symmetric.
"""

from __future__ import annotations

import pytest

from langchain_backend_aws.s3._config import S3BackendConfig


class TestProxiesSsrf:
    def test_public_proxy_accepted(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies": {
                    "http": "http://proxy.example.com:3128",
                    "https": "https://proxy.example.com:3129",
                }
            },
        )
        assert config.extra_boto_config["proxies"]["http"].startswith("http://")

    def test_imds_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://169.254.169.254"},
                },
            )

    def test_localhost_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"https": "https://localhost:3128"},
                },
            )

    def test_rfc1918_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://10.0.0.1:3128"},
                },
            )

    def test_metadata_alias_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://metadata.google.internal"},
                },
            )

    def test_non_http_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "file:///etc/passwd"},
                },
            )

    def test_private_proxy_allowed_with_opt_in(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            allow_private_endpoints=True,
            extra_boto_config={
                "proxies": {"http": "http://localhost:3128"},
            },
        )
        assert config.extra_boto_config["proxies"]["http"] == "http://localhost:3128"

    def test_proxies_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": "http://proxy.example.com"},
            )

    def test_proxy_value_must_be_string(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {"http": 12345}},
            )

    def test_proxy_key_must_be_string(self) -> None:
        # botocore expects scheme-string keys; a non-str key would
        # otherwise propagate to the transport layer and surface as an
        # opaque error.
        with pytest.raises(TypeError, match="keys must be str"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {1: "http://proxy.example.com"}},
            )

    def test_proxies_config_metadata_passthrough(self) -> None:
        # ``proxies_config`` carries non-URL metadata (CA bundle paths,
        # client cert, forwarding flag) and is intentionally not
        # validated against the SSRF allow-list.
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies_config": {
                    "proxy_ca_bundle": "/etc/ssl/ca.pem",
                    "proxy_use_forwarding_for_https": True,
                },
            },
        )
        assert "proxies_config" in config.extra_boto_config
