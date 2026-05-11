"""Regression tests for the eleventh-round review follow-ups.

Covers gaps the code reviewer flagged:

- :func:`read_capped_object` must reject a non-bytes body shape from a
  misbehaving stub (``_internal.py``'s defense-in-depth ``TypeError``).
- :meth:`S3Backend.clear_glob_cache` empties the per-instance LRU.
- ``extra_boto_config['s3']`` must be a dict (shape-parity with
  ``proxies`` / ``proxies_config``).
- A configured ``client_cert`` is logged at WARNING for audit.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws.s3._internal import read_capped_object


def _response_with_str_body(body: str) -> dict[str, Any]:
    """Build a get_object response whose ``Body.read()`` returns a str."""
    stream = MagicMock()
    stream.read.return_value = body
    return {
        "Body": stream,
        "ContentLength": len(body),
        "ETag": '"x"',
        "LastModified": datetime(2025, 3, 7, tzinfo=UTC),
    }


class TestNonBytesBodyDefense:
    """``read_capped_object`` must surface a ``TypeError`` for non-bytes."""

    def test_str_body_raises_type_error(self) -> None:
        client = MagicMock()
        client.get_object.return_value = _response_with_str_body("not-bytes")

        with pytest.raises(TypeError, match="bytes-like"):
            read_capped_object(client, "b", "k", max_bytes=1024)


class TestClearGlobCache:
    """Public ``clear_glob_cache`` drops every entry from the LRU."""

    def test_clear_glob_cache_empties_cache(self) -> None:
        backend = S3Backend.from_kwargs(bucket="b", client=MagicMock())
        backend._compile_glob("*.py")
        backend._compile_glob("*.md")
        info_before = backend._compile_glob.cache_info()  # type: ignore[attr-defined]
        assert info_before.currsize >= 2

        backend.clear_glob_cache()

        info_after = backend._compile_glob.cache_info()  # type: ignore[attr-defined]
        assert info_after.currsize == 0

    def test_clear_glob_cache_idempotent(self) -> None:
        backend = S3Backend.from_kwargs(bucket="b", client=MagicMock())
        backend.clear_glob_cache()
        backend.clear_glob_cache()  # second call must not raise


class TestExtraBotoConfigS3Shape:
    """``extra_boto_config['s3']`` must be a dict."""

    def test_non_dict_s3_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"extra_boto_config\['s3'\]"):
            S3BackendConfig(bucket="b", extra_boto_config={"s3": "virtual"})

    def test_dict_s3_accepted(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"s3": {"addressing_style": "virtual"}},
        )
        assert config.extra_boto_config["s3"] == {"addressing_style": "virtual"}


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
        records = [
            rec for rec in caplog.records if "proxy_client_cert" in rec.message
        ]
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
        assert not any(
            "proxy_client_cert" in rec.message for rec in caplog.records
        )


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


class TestExplicitKeyDroppedWarning:
    """Explicit boto keys passed via ``extra_boto_config`` warn at WARNING."""

    def test_dropped_key_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_backend_aws.s3._config import build_client

        config_logger = "langchain_backend_aws.s3._config"
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"retries": {"max_attempts": 99}},
        )
        with caplog.at_level(logging.WARNING, logger=config_logger):
            build_client(config)
        assert any(
            "retries" in rec.getMessage() and rec.levelname == "WARNING"
            for rec in caplog.records
        )
