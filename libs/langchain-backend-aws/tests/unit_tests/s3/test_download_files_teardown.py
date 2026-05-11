"""``download_files`` teardown logging on multiple in-flight interrupts.

Python's implicit ``__context__`` chaining cannot link sibling-future
exceptions, so subsequent teardown signals are otherwise silently
dropped. They must surface as a WARNING for operator visibility.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from langchain_backend_aws.s3._io import download_files


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
