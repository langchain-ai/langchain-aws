"""Unit tests for AgentCore stream parsing functions.

These tests validate ``_extract_text_from_stream`` and
``_extract_files_from_stream`` against various response shapes without
making any network calls.
"""

from __future__ import annotations

import base64
from typing import Any

from langchain_agentcore_codeinterpreter.sandbox import (
    _extract_files_from_stream,
    _extract_text_from_stream,
)

# ------------------------------------------------------------------
# _extract_text_from_stream
# ------------------------------------------------------------------


def test_extract_text_simple_output() -> None:
    """Single text event should be returned as-is."""
    response: dict[str, Any] = {
        "stream": [{"result": {"content": [{"type": "text", "text": "hello world"}]}}]
    }
    text, code = _extract_text_from_stream(response)
    assert text == "hello world"
    assert code is None


def test_extract_text_with_exit_code() -> None:
    """Explicit exitCode should be captured."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "exitCode": 0,
                    "content": [{"type": "text", "text": "success"}],
                }
            }
        ]
    }
    text, code = _extract_text_from_stream(response)
    assert text == "success"
    assert code == 0


def test_extract_text_error_sets_exit_code() -> None:
    """An error content item should imply exit code 1."""
    response: dict[str, Any] = {
        "stream": [
            {"result": {"content": [{"type": "error", "text": "command not found"}]}}
        ]
    }
    text, code = _extract_text_from_stream(response)
    assert text == "Error: command not found"
    assert code == 1


def test_extract_text_empty_stream() -> None:
    """Empty stream should produce empty output and no exit code."""
    text, code = _extract_text_from_stream({"stream": []})
    assert text == ""
    assert code is None


def test_extract_text_no_stream_key() -> None:
    """Missing stream key should not raise."""
    text, code = _extract_text_from_stream({})
    assert text == ""
    assert code is None


def test_extract_text_multiple_events() -> None:
    """Multiple text events should be joined with newlines."""
    response: dict[str, Any] = {
        "stream": [
            {"result": {"content": [{"type": "text", "text": "line 1"}]}},
            {"result": {"content": [{"type": "text", "text": "line 2"}]}},
        ]
    }
    text, code = _extract_text_from_stream(response)
    assert text == "line 1\nline 2"


def test_extract_text_mixed_text_and_error() -> None:
    """Text and error in same event should both appear in output."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {"type": "text", "text": "partial output"},
                        {"type": "error", "text": "then failed"},
                    ]
                }
            }
        ]
    }
    text, code = _extract_text_from_stream(response)
    assert "partial output" in text
    assert "Error: then failed" in text
    assert code == 1


def test_extract_text_explicit_exit_code_overrides_error() -> None:
    """Explicit exitCode takes precedence over error-implied code."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "exitCode": 42,
                    "content": [{"type": "error", "text": "oops"}],
                }
            }
        ]
    }
    _, code = _extract_text_from_stream(response)
    assert code == 42


def test_extract_text_events_without_result_skipped() -> None:
    """Events that lack a 'result' key should be silently skipped."""
    response: dict[str, Any] = {
        "stream": [
            {"metadata": {"info": "ignored"}},
            {"result": {"content": [{"type": "text", "text": "real"}]}},
        ]
    }
    text, _ = _extract_text_from_stream(response)
    assert text == "real"


# ------------------------------------------------------------------
# _extract_files_from_stream
# ------------------------------------------------------------------


def test_extract_files_text_resource() -> None:
    """Text resource should be decoded to bytes."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///hello.txt",
                                "text": "hello content",
                            },
                        }
                    ]
                }
            }
        ]
    }
    files = _extract_files_from_stream(response, ["/hello.txt"])
    assert files["/hello.txt"] == b"hello content"


def test_extract_files_blob_resource() -> None:
    """Base64 blob resource should be decoded to original bytes."""
    encoded = base64.b64encode(b"binary data").decode()
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {"uri": "file:///data.bin", "blob": encoded},
                        }
                    ]
                }
            }
        ]
    }
    files = _extract_files_from_stream(response, ["/data.bin"])
    assert files["/data.bin"] == b"binary data"


def test_extract_files_path_normalization() -> None:
    """Relative requested paths should still match file:/// URIs."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///test.py",
                                "text": "print('hi')",
                            },
                        }
                    ]
                }
            }
        ]
    }
    files = _extract_files_from_stream(response, ["test.py"])
    assert "test.py" in files


def test_extract_files_missing_file() -> None:
    """Missing file should result in an empty dict."""
    response: dict[str, Any] = {"stream": [{"result": {"content": []}}]}
    files = _extract_files_from_stream(response, ["/missing.txt"])
    assert files == {}


def test_extract_files_empty_stream() -> None:
    """Empty stream should produce empty result."""
    files = _extract_files_from_stream({"stream": []}, ["/any.txt"])
    assert files == {}


def test_extract_files_skips_non_resource_types() -> None:
    """Non-resource content items should be ignored."""
    response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {"type": "text", "text": "not a file"},
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///real.txt",
                                "text": "real",
                            },
                        },
                    ]
                }
            }
        ]
    }
    files = _extract_files_from_stream(response, ["/real.txt"])
    assert len(files) == 1
    assert files["/real.txt"] == b"real"
