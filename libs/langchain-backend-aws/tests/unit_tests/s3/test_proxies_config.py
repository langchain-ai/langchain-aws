"""Validation rules for ``extra_boto_config['proxies_config']``.

The botocore ``proxies_config`` surface is small but heterogeneously
typed (str / bool / tuple), so the dataclass enforces an allow-list and
per-key shape so misconfiguration fails at construction time. The
``proxy_client_cert`` path is additionally audit-logged so operators can
trace which certificate file was loaded without exposing the raw path.
"""

from __future__ import annotations

import logging

import pytest

from langchain_backend_aws import S3BackendConfig


class TestProxiesConfigKeyAllowList:
    """Unknown ``proxies_config`` keys are rejected with the allow-list."""

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported keys"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {"unknown_option": "value"},
                },
            )

    def test_known_keys_accepted(self) -> None:
        # All three documented botocore keys must pass through cleanly.
        cfg = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies_config": {
                    "proxy_ca_bundle": "/etc/ssl/ca.pem",
                    "proxy_use_forwarding_for_https": True,
                    "proxy_client_cert": ("/tmp/cert.pem", "/tmp/key.pem"),
                },
            },
        )
        assert "proxies_config" in cfg.extra_boto_config


class TestProxiesConfigTupleShape:
    """Only ``proxy_client_cert`` may carry a tuple value in ``proxies_config``."""

    def test_non_client_cert_tuple_rejected(self) -> None:
        # ``proxy_ca_bundle`` is in the allow-list but only accepts
        # str/None — passing a tuple must fail closed.
        with pytest.raises(TypeError, match="does not accept a tuple"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {"proxy_ca_bundle": ("a", "b")},
                },
            )


class TestClientCertAuditLog:
    """Configured ``client_cert`` paths must be logged at WARNING."""

    def test_client_cert_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        config_logger = "langchain_backend_aws.s3._config"
        with caplog.at_level(logging.WARNING, logger=config_logger):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {
                        "proxy_client_cert": (
                            "/etc/ssl/cert.pem",
                            "/etc/ssl/key.pem",
                        ),
                    },
                },
            )
        records = [rec for rec in caplog.records if "proxy_client_cert" in rec.message]
        assert records, "expected an audit warning for configured proxy_client_cert"
        # Raw filesystem paths must not appear in the audit log; only
        # SHA-256 fingerprints (12-char hex) are emitted so log
        # aggregators cannot recover the operator's private-key path.
        for rec in records:
            assert "/etc/ssl/cert.pem" not in rec.getMessage()
            assert "/etc/ssl/key.pem" not in rec.getMessage()
            assert "sha256:" in rec.getMessage()

    def test_no_client_cert_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        config_logger = "langchain_backend_aws.s3._config"
        with caplog.at_level(logging.WARNING, logger=config_logger):
            S3BackendConfig(bucket="b")
        assert not any("proxy_client_cert" in rec.message for rec in caplog.records)
