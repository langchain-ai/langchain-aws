"""Regression tests for the fifteenth-round review follow-ups.

Covers the five HIGH-severity findings:

- ``extra_boto_config`` deepcopy failures surface as ``ValueError`` with a
  configuration-shaped message rather than an opaque ``TypeError``.
- ``proxies_config`` keys outside the documented botocore surface are
  rejected up-front (mirrors the outer ``ALLOWED_BOTO_KEYS`` discipline).
- ``download_files`` logs a WARNING when a second teardown signal arrives
  while one is already pending (Python's implicit ``__context__`` chaining
  cannot link sibling-future exceptions).
- ``read`` warns when ``offset``/``limit`` are non-default but the body is
  non-UTF-8 (the base64 path returns the full capped body).
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_backend_aws import S3BackendConfig
from langchain_backend_aws.s3._io import download_files


class TestDeepcopyFailureNormalization:
    """A non-copyable ``extra_boto_config`` value surfaces as ``ValueError``."""

    def test_lock_value_surfaces_value_error(self) -> None:
        # ``threading.Lock`` rejects ``deepcopy`` with a ``TypeError``.
        # The dataclass must normalize that into ``ValueError`` so callers
        # see a consistent diagnostic shape across all ``_validate_*``
        # checks.
        with pytest.raises(ValueError, match="non-copyable"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"_lock": threading.Lock()},
            )


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


class TestDownloadFilesTeardownLogging:
    """Subsequent teardown signals are visible in the operator log."""

    def test_second_teardown_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Two paths, both raising ``KeyboardInterrupt``: only the first is
        # re-raised, but the second must surface as a WARNING so the loss
        # is not silent.
        from langchain_backend_aws.s3 import _io

        def fail(path: str) -> Any:
            raise KeyboardInterrupt(f"interrupt on {path}")

        with (
            caplog.at_level(logging.WARNING, logger=_io.__name__),
            pytest.raises(KeyboardInterrupt),
        ):
            download_files(
                ["a", "b", "c"],
                download_one=fail,
                download_concurrency=4,
                max_pool_connections=10,
            )
        # At least one (and likely two) extra teardown signals were
        # discarded; the WARNING must surface so the loss is visible.
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING"
            and "additional teardown signal" in rec.getMessage()
        ]
        assert warnings, "expected at least one extra-teardown warning"


class TestBinaryReadModeOffsetLimitWarning:
    """Non-default ``offset``/``limit`` warn on the base64 path."""

    def _make_get_object(self, body: bytes) -> Any:
        client = MagicMock()
        client.get_object.return_value = {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "LastModified": MagicMock(),
            "ETag": '"deadbeef"',
        }
        client.get_object.return_value["LastModified"].astimezone.return_value = (
            MagicMock()
        )
        return client

    def test_non_default_offset_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_backend_aws import S3Backend

        client = self._make_get_object(b"\xff\xfe\x00\x01binary")
        backend = S3Backend(
            S3BackendConfig(bucket="b", binary_read_mode="base64"),
            client=client,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_backend_aws.s3._read"):
            result = backend.read("/binary.bin", offset=10, limit=5)
        assert result.file_data is not None
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING"
            and "non-UTF-8" in rec.getMessage()
            and "offset=10" in rec.getMessage()
        ]
        assert warnings, "expected a warning when offset/limit ignored on binary body"

    def test_default_offset_limit_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from langchain_backend_aws import S3Backend

        client = self._make_get_object(b"\xff\xfe\x00\x01binary")
        backend = S3Backend(
            S3BackendConfig(bucket="b", binary_read_mode="base64"),
            client=client,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_backend_aws.s3._read"):
            backend.read("/binary.bin")  # use defaults
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelname == "WARNING" and "non-UTF-8" in rec.getMessage()
        ]
        assert not warnings, "default offset/limit must not warn"


