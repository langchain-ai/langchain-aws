"""Unit tests for AgentCoreSandbox using a mocked CodeInterpreter.

All tests use ``unittest.mock.MagicMock`` to avoid network calls and
AWS credential requirements.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_agentcore_codeinterpreter import AgentCoreSandbox


def _make_sandbox(
    invoke_return: dict[str, Any] | None = None,
    session_id: str = "test-session-123",
) -> tuple[AgentCoreSandbox, MagicMock]:
    """Create a sandbox with a mocked interpreter."""
    interpreter = MagicMock()
    interpreter.session_id = session_id
    interpreter.invoke.return_value = invoke_return or {"stream": []}
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


# ------------------------------------------------------------------
# _to_relative_path()
# ------------------------------------------------------------------


def test_relative_path_stripping() -> None:
    """Leading slashes should be stripped; relative paths left as-is."""
    assert AgentCoreSandbox._to_relative_path("/abs/path.txt") == "abs/path.txt"
    assert AgentCoreSandbox._to_relative_path("rel/path.txt") == "rel/path.txt"
    assert AgentCoreSandbox._to_relative_path("///triple.txt") == "triple.txt"


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


def test_keyword_only_init() -> None:
    """The constructor requires 'interpreter' as a keyword argument."""
    interpreter = MagicMock()
    interpreter.session_id = "s"
    sandbox = AgentCoreSandbox(interpreter=interpreter)
    assert sandbox.id == "s"
