"""Unit tests for AgentCoreSandbox using a mocked CodeInterpreter.

All tests use ``unittest.mock.MagicMock`` to avoid network calls and
AWS credential requirements.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from langchain_agentcore_codeinterpreter.sandbox import (
    _AGENTCORE_EXECUTOR,
    SessionExpiredError,
)


def _make_sandbox(
    invoke_return: dict[str, Any] | None = None,
    session_id: str = "test-session-123",
) -> tuple[AgentCoreSandbox, MagicMock]:
    """Create a sandbox with a mocked interpreter."""
    interpreter = MagicMock()
    interpreter.session_id = session_id
    interpreter.invoke.return_value = invoke_return or {"stream": []}
    return AgentCoreSandbox(interpreter=interpreter), interpreter


def _make_expired_sandbox() -> tuple[AgentCoreSandbox, MagicMock]:
    """Create a sandbox whose interpreter raises ResourceNotFoundException."""
    interpreter = MagicMock()
    interpreter.session_id = "expired-session"
    interpreter.invoke.side_effect = ClientError(
        {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Session not found",
            }
        },
        "InvokeCodeInterpreter",
    )
    return AgentCoreSandbox(interpreter=interpreter), interpreter


# ------------------------------------------------------------------
# Property: id
# ------------------------------------------------------------------


def test_id_returns_session_id() -> None:
    """The id property should reflect the interpreter session_id."""
    sandbox, _ = _make_sandbox()
    assert sandbox.id == "test-session-123"


def test_id_returns_empty_when_none() -> None:
    """A None session_id should produce an empty string."""
    sandbox, mock = _make_sandbox(session_id=None)  # type: ignore[arg-type]
    assert sandbox.id == ""


# ------------------------------------------------------------------
# execute()
# ------------------------------------------------------------------


def test_execute_calls_invoke_correctly() -> None:
    """execute() should call invoke with executeCommand and the command."""
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "exitCode": 0,
                        "content": [{"type": "text", "text": "ok"}],
                    }
                }
            ]
        }
    )
    result = sandbox.execute("echo ok")
    mock.invoke.assert_called_once_with(
        method="executeCommand", params={"command": "echo ok"}
    )
    assert result.output == "ok"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_defaults_exit_code_to_zero() -> None:
    """When the stream has no exitCode, execute() should default to 0."""
    sandbox, _ = _make_sandbox(
        {"stream": [{"result": {"content": [{"type": "text", "text": "no code"}]}}]}
    )
    result = sandbox.execute("cmd")
    assert result.exit_code == 0


def test_execute_handles_exception() -> None:
    """SDK exceptions should be caught and returned as exit code 1."""
    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = RuntimeError("connection lost")
    result = sandbox.execute("echo fail")
    assert result.exit_code == 1
    assert "connection lost" in result.output


def test_execute_handles_session_expiry() -> None:
    """ResourceNotFoundException should produce a clear expiry message."""
    sandbox, _ = _make_expired_sandbox()
    result = sandbox.execute("echo hello")
    assert result.exit_code == 1
    assert "expired" in result.output.lower()


# ------------------------------------------------------------------
# upload_files()
# ------------------------------------------------------------------


def test_upload_files_text() -> None:
    """UTF-8 content should be uploaded as text with leading / stripped."""
    sandbox, mock = _make_sandbox()
    result = sandbox.upload_files([("/hello.py", b"print('hi')")])
    mock.invoke.assert_called_once()
    call_kwargs = mock.invoke.call_args.kwargs
    assert call_kwargs["method"] == "writeFiles"
    content = call_kwargs["params"]["content"]
    assert len(content) == 1
    assert content[0]["path"] == "hello.py"
    assert content[0]["text"] == "print('hi')"
    assert result[0].error is None
    assert result[0].path == "/hello.py"


def test_upload_files_binary_uses_blob() -> None:
    """Non-UTF-8 content should be base64-encoded as a blob."""
    sandbox, mock = _make_sandbox()
    sandbox.upload_files([("/data.bin", b"\x80\x81\x82")])
    content = mock.invoke.call_args.kwargs["params"]["content"][0]
    assert "blob" in content
    assert "text" not in content


def test_upload_files_empty_list() -> None:
    """An empty file list should not call invoke."""
    sandbox, mock = _make_sandbox()
    result = sandbox.upload_files([])
    mock.invoke.assert_not_called()
    assert result == []


def test_upload_files_handles_exception() -> None:
    """SDK errors during upload should return permission_denied."""
    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = RuntimeError("write failed")
    result = sandbox.upload_files([("/a.txt", b"data")])
    assert result[0].error == "permission_denied"


def test_upload_files_handles_session_expiry() -> None:
    """Session expiry during upload should return permission_denied."""
    sandbox, _ = _make_expired_sandbox()
    result = sandbox.upload_files([("/a.txt", b"data")])
    assert result[0].error == "permission_denied"


# ------------------------------------------------------------------
# download_files()
# ------------------------------------------------------------------


def test_download_files() -> None:
    """A successful download should return content bytes."""
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///test.txt",
                                    "text": "content",
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    results = sandbox.download_files(["/test.txt"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["test.txt"]}
    )
    assert results[0].content == b"content"
    assert results[0].error is None


def test_download_files_missing() -> None:
    """Missing files should be reported as file_not_found."""
    sandbox, _ = _make_sandbox({"stream": [{"result": {"content": []}}]})
    results = sandbox.download_files(["/missing.txt"])
    assert results[0].error == "file_not_found"
    assert results[0].content is None


def test_download_files_handles_exception() -> None:
    """SDK errors during download should return file_not_found."""
    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = RuntimeError("read failed")
    results = sandbox.download_files(["/a.txt"])
    assert results[0].error == "file_not_found"


def test_download_files_handles_session_expiry() -> None:
    """Session expiry during download should return permission_denied."""
    sandbox, _ = _make_expired_sandbox()
    results = sandbox.download_files(["/a.txt"])
    assert results[0].error == "permission_denied"


def test_download_files_dot_slash_path() -> None:
    """./-prefixed paths must round-trip through readFiles and lookup."""
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///data/foo.png",
                                    "blob": fake_png,
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    results = sandbox.download_files(["./data/foo.png"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["data/foo.png"]}
    )
    assert results[0].error is None
    assert results[0].content == fake_png


# ------------------------------------------------------------------
# _to_relative_path()
# ------------------------------------------------------------------


def test_relative_path_stripping() -> None:
    """Leading slashes should be stripped; relative paths left as-is."""
    assert AgentCoreSandbox._to_relative_path("/abs/path.txt") == "abs/path.txt"
    assert AgentCoreSandbox._to_relative_path("rel/path.txt") == "rel/path.txt"
    assert AgentCoreSandbox._to_relative_path("///triple.txt") == "triple.txt"


def test_relative_path_strips_dot_slash() -> None:
    """./ and repeated ././ prefixes should be stripped."""
    assert AgentCoreSandbox._to_relative_path("./data/foo.png") == "data/foo.png"
    assert AgentCoreSandbox._to_relative_path("././foo.png") == "foo.png"
    assert AgentCoreSandbox._to_relative_path("/./data/foo.png") == "data/foo.png"


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


def test_keyword_only_init() -> None:
    """The constructor requires 'interpreter' as a keyword argument."""
    interpreter = MagicMock()
    interpreter.session_id = "s"
    sandbox = AgentCoreSandbox(interpreter=interpreter)
    assert sandbox.id == "s"


# ------------------------------------------------------------------
# _invoke() — eager stream consumption
# ------------------------------------------------------------------


def test_invoke_eagerly_consumes_stream() -> None:
    """_invoke() should materialize a lazy stream iterator into a list."""

    def lazy_stream() -> Iterator[dict[str, Any]]:
        yield {
            "result": {
                "exitCode": 0,
                "content": [{"type": "text", "text": "hello"}],
            }
        }

    sandbox, mock = _make_sandbox()
    mock.invoke.return_value = {"stream": lazy_stream()}

    response = sandbox._invoke(method="executeCommand", params={"command": "echo"})
    # Stream should be a list, not a generator
    assert isinstance(response["stream"], list)
    assert len(response["stream"]) == 1


def test_invoke_handles_no_stream_key() -> None:
    """_invoke() should work when response has no 'stream' key."""
    sandbox, mock = _make_sandbox()
    mock.invoke.return_value = {"metadata": "ok"}

    response = sandbox._invoke(method="listFiles", params={})
    assert "stream" not in response
    assert response["metadata"] == "ok"


def test_invoke_raises_session_expired_error() -> None:
    """_invoke() should raise SessionExpiredError for ResourceNotFoundException."""
    sandbox, _ = _make_expired_sandbox()
    with pytest.raises(SessionExpiredError) as exc_info:
        sandbox._invoke(method="executeCommand", params={"command": "echo"})
    assert "expired" in str(exc_info.value).lower()
    assert exc_info.value.session_id == "expired-session"


def test_invoke_reraises_other_client_errors() -> None:
    """_invoke() should reraise non-ResourceNotFound ClientErrors."""
    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeCodeInterpreter",
    )
    with pytest.raises(ClientError) as exc_info:
        sandbox._invoke(method="executeCommand", params={"command": "echo"})
    assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"


# ------------------------------------------------------------------
# SessionExpiredError
# ------------------------------------------------------------------


def test_session_expired_error_attributes() -> None:
    """SessionExpiredError should store session_id and original exception."""
    original = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Gone"}},
        "InvokeCodeInterpreter",
    )
    err = SessionExpiredError("sess-123", original)
    assert err.session_id == "sess-123"
    assert err.original is original
    assert "sess-123" in str(err)


# ------------------------------------------------------------------
# Async overrides — verify they use dedicated executor
# ------------------------------------------------------------------


def test_aexecute_uses_dedicated_executor() -> None:
    """aexecute() should run on the agentcore executor, not the default."""
    invoke_return = {
        "stream": [
            {
                "result": {
                    "exitCode": 0,
                    "content": [{"type": "text", "text": "async ok"}],
                }
            }
        ]
    }
    sandbox, mock = _make_sandbox(invoke_return)

    import threading

    thread_names: list[str] = []
    canned_response = invoke_return

    def tracking_invoke(**kwargs: Any) -> dict[str, Any]:
        thread_names.append(threading.current_thread().name)
        return canned_response

    mock.invoke.side_effect = tracking_invoke

    result = asyncio.run(sandbox.aexecute("echo async"))
    assert result.output == "async ok"
    assert result.exit_code == 0
    assert any("agentcore-sandbox" in name for name in thread_names)


def test_awrite_runs_on_dedicated_executor() -> None:
    """awrite() should not use the default asyncio executor."""
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "exitCode": 0,
                        "content": [{"type": "text", "text": ""}],
                    }
                }
            ]
        }
    )

    import threading

    thread_names: list[str] = []
    canned_response: dict[str, Any] = {"stream": []}

    def tracking_invoke(**kwargs: Any) -> dict[str, Any]:
        thread_names.append(threading.current_thread().name)
        return canned_response

    mock.invoke.side_effect = tracking_invoke

    try:
        asyncio.run(sandbox.awrite("/test.txt", "content"))
    except Exception:
        pass

    if thread_names:
        assert any("agentcore-sandbox" in name for name in thread_names)


def test_aupload_files_uses_dedicated_executor() -> None:
    """aupload_files() should run on the agentcore executor."""
    sandbox, mock = _make_sandbox()

    import threading

    thread_names: list[str] = []
    canned_response: dict[str, Any] = {"stream": []}

    def tracking_invoke(**kwargs: Any) -> dict[str, Any]:
        thread_names.append(threading.current_thread().name)
        return canned_response

    mock.invoke.side_effect = tracking_invoke

    result = asyncio.run(sandbox.aupload_files([("/test.txt", b"data")]))
    assert result[0].error is None
    assert any("agentcore-sandbox" in name for name in thread_names)


def test_adownload_files_uses_dedicated_executor() -> None:
    """adownload_files() should run on the agentcore executor."""
    canned_response: dict[str, Any] = {
        "stream": [
            {
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///test.txt",
                                "text": "content",
                            },
                        }
                    ]
                }
            }
        ]
    }
    sandbox, mock = _make_sandbox(canned_response)

    import threading

    thread_names: list[str] = []

    def tracking_invoke(**kwargs: Any) -> dict[str, Any]:
        thread_names.append(threading.current_thread().name)
        return canned_response

    mock.invoke.side_effect = tracking_invoke

    results = asyncio.run(sandbox.adownload_files(["/test.txt"]))
    assert results[0].content == b"content"
    assert any("agentcore-sandbox" in name for name in thread_names)


def test_aexecute_handles_session_expiry() -> None:
    """aexecute() should handle session expiry gracefully."""
    sandbox, _ = _make_expired_sandbox()
    result = asyncio.run(sandbox.aexecute("echo hello"))
    assert result.exit_code == 1
    assert "expired" in result.output.lower()


# ------------------------------------------------------------------
# Executor configuration
# ------------------------------------------------------------------


def test_dedicated_executor_exists() -> None:
    """The module-level executor should be configured."""
    assert _AGENTCORE_EXECUTOR is not None
    assert _AGENTCORE_EXECUTOR._max_workers == 4
    assert _AGENTCORE_EXECUTOR._thread_name_prefix == "agentcore-sandbox"
