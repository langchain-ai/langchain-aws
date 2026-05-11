"""Validation rules for ``S3BackendConfig.extra_boto_config``.

The dataclass normalizes botocore's loose ``Config`` surface into a
constrained dict so misconfiguration fails at construction time with a
configuration-shaped diagnostic, never as an opaque downstream error.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from langchain_backend_aws import S3BackendConfig


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


class TestExplicitKeyDroppedWarning:
    """Explicit boto keys passed via ``extra_boto_config`` warn at WARNING."""

    def test_dropped_key_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_backend_aws.s3._config import build_client

        config_logger = "langchain_backend_aws.s3._config"
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"retries": {"max_attempts": 99}},
        )
        with (
            caplog.at_level(logging.WARNING, logger=config_logger),
            patch(
                "langchain_backend_aws.s3._config.boto3.client",
                return_value=MagicMock(),
            ),
        ):
            build_client(config)
        assert any(
            "retries" in rec.getMessage() and rec.levelname == "WARNING"
            for rec in caplog.records
        )
