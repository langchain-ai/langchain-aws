"""``read_capped_object`` rejects a non-bytes body from a misbehaving stub.

The defense-in-depth ``TypeError`` in ``_internal.py`` catches the case
where a custom boto stub returns a ``str`` body — without the check the
downstream encoding path would silently corrupt the data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

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
