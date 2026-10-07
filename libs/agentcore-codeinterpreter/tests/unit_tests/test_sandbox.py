"""Unit tests for AgentCoreSandbox using a mocked CodeInterpreter.

All tests use ``unittest.mock.MagicMock`` to avoid network calls and
AWS credential requirements.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from deepagents.backends.protocol import (
    EditResult,
    FileData,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from langchain_agentcore_codeinterpreter.sandbox import (
    _AGENTCORE_EXECUTOR,
    _BASE_GREP_HAS_MAX_COUNT,
    _BASE_HAS_DELETE,
    INTEGRATION_SOURCE,
    SessionExpiredError,
)

#: Extra kwargs BaseSandbox.grep receives, which depends on the deepagents version.
_GREP_KW: dict[str, Any] = {"max_count": None} if _BASE_GREP_HAS_MAX_COUNT else {}


def _make_sandbox(
    invoke_return: dict[str, Any] | None = None,
    session_id: str = "test-session-123",
    cwd: str | None = None,
) -> tuple[AgentCoreSandbox, MagicMock]:
    """Create a sandbox with a mocked interpreter."""
    interpreter = MagicMock()
    interpreter.session_id = session_id
    interpreter.invoke.return_value = invoke_return or {"stream": []}
    return AgentCoreSandbox(interpreter=interpreter, cwd=cwd), interpreter


def _make_expired_sandbox(cwd: str | None = None) -> tuple[AgentCoreSandbox, MagicMock]:
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
    return AgentCoreSandbox(interpreter=interpreter, cwd=cwd), interpreter


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
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": False,
                        "structuredContent": {"exitCode": 0, "stdout": "ok"},
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


def test_execute_unknown_exit_code_is_none() -> None:
    """With no exit code reported, execute() returns None, not success.

    Deep Agents reads None as "could not be determined". Reporting 0 told the
    agent that commands of unknown outcome had succeeded.
    """
    sandbox, _ = _make_sandbox(
        {"stream": [{"result": {"content": [{"type": "text", "text": "no code"}]}}]}
    )
    result = sandbox.execute("cmd")
    assert result.exit_code is None


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
    """Non-UTF-8 content should be uploaded as raw blob bytes."""
    sandbox, mock = _make_sandbox()
    binary_content = b"\x80\x81\x82"

    sandbox.upload_files([("/data.bin", binary_content)])

    content = mock.invoke.call_args.kwargs["params"]["content"][0]
    assert "blob" in content
    assert "text" not in content
    assert content["blob"] == binary_content


def test_upload_files_mixed_text_and_binary() -> None:
    """Text and binary files should use the correct writeFiles fields."""
    sandbox, mock = _make_sandbox()
    binary_content = b"\x00\xff\xfe"

    result = sandbox.upload_files(
        [
            ("/hello.py", b"print('hi')"),
            ("/data.bin", binary_content),
        ]
    )

    uploaded = mock.invoke.call_args.kwargs["params"]["content"]
    assert uploaded == [
        {"path": "hello.py", "text": "print('hi')"},
        {"path": "data.bin", "blob": binary_content},
    ]
    assert [response.path for response in result] == ["/hello.py", "/data.bin"]
    assert [response.error for response in result] == [None, None]


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
        },
        cwd="/",
    )
    results = sandbox.download_files(["/test.txt"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["test.txt"]}
    )
    assert results[0].content == b"content"
    assert results[0].error is None


def test_download_files_missing() -> None:
    """Missing files should be reported as file_not_found."""
    sandbox, _ = _make_sandbox({"stream": [{"result": {"content": []}}]}, cwd="/")
    results = sandbox.download_files(["/missing.txt"])
    assert results[0].error == "file_not_found"
    assert results[0].content is None


def test_download_files_handles_exception() -> None:
    """SDK errors during download should return file_not_found."""
    sandbox, mock = _make_sandbox(cwd="/")
    mock.invoke.side_effect = RuntimeError("read failed")
    results = sandbox.download_files(["/a.txt"])
    assert results[0].error == "file_not_found"


def test_download_files_handles_session_expiry() -> None:
    """Session expiry during download should return permission_denied."""
    sandbox, _ = _make_expired_sandbox(cwd="/")
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
        },
        cwd="/",
    )
    results = sandbox.download_files(["./data/foo.png"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["data/foo.png"]}
    )
    assert results[0].error is None
    assert results[0].content == fake_png


def test_download_files_strips_cwd_prefix() -> None:
    """Absolute paths under cwd should have the prefix stripped for readFiles."""
    cwd = "/opt/sandbox"
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///workspace/hello.py",
                                    "text": "hello",
                                },
                            }
                        ]
                    }
                }
            ]
        },
        cwd=cwd,
    )
    results = sandbox.download_files([f"{cwd}/workspace/hello.py"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["workspace/hello.py"]}
    )
    assert results[0].content == b"hello"
    assert results[0].error is None
    assert results[0].path == f"{cwd}/workspace/hello.py"


def test_download_files_virtual_path_with_cwd() -> None:
    """Virtual paths not under cwd should fall back to leading-slash stripping."""
    sandbox, mock = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///workspace/hello.py",
                                    "text": "hello",
                                },
                            }
                        ]
                    }
                }
            ]
        },
        cwd="/opt/sandbox",
    )
    results = sandbox.download_files(["/workspace/hello.py"])
    mock.invoke.assert_called_once_with(
        method="readFiles", params={"paths": ["workspace/hello.py"]}
    )
    assert results[0].content == b"hello"
    assert results[0].error is None


def test_download_files_lazy_cwd_detection() -> None:
    """When cwd is not provided, download_files should detect it via pwd."""
    cwd = "/opt/sandbox"

    def invoke(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("method") == "executeCommand":
            return {
                "stream": [
                    {
                        "result": {
                            "exitCode": 0,
                            "content": [{"type": "text", "text": cwd}],
                        }
                    }
                ]
            }
        return {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///workspace/hello.py",
                                    "text": "hello",
                                },
                            }
                        ]
                    }
                }
            ]
        }

    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = invoke

    results = sandbox.download_files([f"{cwd}/workspace/hello.py"])
    assert sandbox._cwd == cwd
    assert results[0].content == b"hello"
    assert results[0].error is None


# ------------------------------------------------------------------
# _to_relative_path()
# ------------------------------------------------------------------


def test_relative_path_stripping() -> None:
    """Leading slashes should be stripped; relative paths left as-is."""
    sandbox, _ = _make_sandbox()
    assert sandbox._to_relative_path("/abs/path.txt") == "abs/path.txt"
    assert sandbox._to_relative_path("rel/path.txt") == "rel/path.txt"
    assert sandbox._to_relative_path("///triple.txt") == "triple.txt"


def test_relative_path_strips_dot_slash() -> None:
    """./ and repeated ././ prefixes should be stripped."""
    sandbox, _ = _make_sandbox()
    assert sandbox._to_relative_path("./data/foo.png") == "data/foo.png"
    assert sandbox._to_relative_path("././foo.png") == "foo.png"
    assert sandbox._to_relative_path("/./data/foo.png") == "data/foo.png"


def test_relative_path_strips_cwd_prefix() -> None:
    """When cwd is known, paths under cwd should have the prefix stripped."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    result = sandbox._to_relative_path("/opt/sandbox/workspace/hello.py")
    assert result == "workspace/hello.py"
    assert sandbox._to_relative_path("/opt/sandbox/hello.py") == "hello.py"


def test_relative_path_virtual_path_falls_back_to_strip() -> None:
    """Paths outside cwd should fall back to leading-slash stripping."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    assert sandbox._to_relative_path("/workspace/hello.py") == "workspace/hello.py"


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


def test_keyword_only_init() -> None:
    """The constructor requires 'interpreter' as a keyword argument."""
    interpreter = MagicMock()
    interpreter.session_id = "s"
    sandbox = AgentCoreSandbox(interpreter=interpreter)
    assert sandbox.id == "s"


def test_cwd_constructor_stores_stripped_cwd() -> None:
    """cwd passed at construction should be stored with trailing slash removed."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox/")
    assert sandbox._cwd == "/opt/sandbox"


def test_cwd_defaults_to_none() -> None:
    """When cwd is not passed, _cwd should start as None (lazy detection)."""
    sandbox, _ = _make_sandbox()
    assert sandbox._cwd is None


# ------------------------------------------------------------------
# _to_absolute_path()
# ------------------------------------------------------------------


def test_to_absolute_path_already_under_cwd() -> None:
    """Paths already under the cwd should be returned unchanged."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    path = "/opt/sandbox/workspace/hello.py"
    assert sandbox._to_absolute_path(path) == path


def test_to_absolute_path_virtual_path_prepends_cwd() -> None:
    """Virtual paths like /workspace/hello.py should be resolved under cwd."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    result = sandbox._to_absolute_path("/workspace/hello.py")
    assert result == "/opt/sandbox/workspace/hello.py"


def test_to_absolute_path_relative_path_prepends_cwd() -> None:
    """Relative paths should be resolved under cwd."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    result = sandbox._to_absolute_path("workspace/hello.py")
    assert result == "/opt/sandbox/workspace/hello.py"


def test_to_absolute_path_root_cwd_returns_as_is() -> None:
    """When cwd is / (stored as empty string), absolute paths are returned as-is."""
    sandbox, _ = _make_sandbox(cwd="/")
    assert sandbox._to_absolute_path("/workspace/hello.py") == "/workspace/hello.py"


# ------------------------------------------------------------------
# write() — path normalization (issue #1055)
# ------------------------------------------------------------------


def _make_successful_invoke(cwd: str = "") -> Any:
    """Return an invoke side_effect that succeeds for all methods."""

    def invoke(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("method") == "executeCommand":
            return {
                "stream": [
                    {
                        "result": {
                            "exitCode": 0,
                            "content": [{"type": "text", "text": cwd}],
                        }
                    }
                ]
            }
        return {"stream": []}

    return invoke


def test_write_strips_cwd_prefix_from_upload_path() -> None:
    """upload path must be cwd-relative so writeFiles doesn't double the prefix."""
    cwd = "/opt/sandbox/var"
    sandbox, mock = _make_sandbox(cwd=cwd)
    mock.invoke.side_effect = _make_successful_invoke(cwd)

    abs_path = f"{cwd}/workspace/hello.py"
    result = sandbox.write(abs_path, "hello")

    write_files_calls = [
        c for c in mock.invoke.call_args_list if c.kwargs.get("method") == "writeFiles"
    ]
    assert len(write_files_calls) == 1
    uploaded_path = write_files_calls[0].kwargs["params"]["content"][0]["path"]
    assert uploaded_path == "workspace/hello.py"
    assert result.path == abs_path


def test_write_resolves_virtual_path_for_preflight() -> None:
    """Virtual paths must be resolved to real absolute paths before preflight."""
    cwd = "/opt/sandbox"
    sandbox, mock = _make_sandbox(cwd=cwd)
    mock.invoke.side_effect = _make_successful_invoke(cwd)

    result = sandbox.write("/workspace/hello.py", "hello")

    # The resolved absolute path should be returned and used for the upload.
    assert result.path == "/opt/sandbox/workspace/hello.py"
    write_files_calls = [
        c for c in mock.invoke.call_args_list if c.kwargs.get("method") == "writeFiles"
    ]
    assert len(write_files_calls) == 1
    uploaded_path = write_files_calls[0].kwargs["params"]["content"][0]["path"]
    assert uploaded_path == "workspace/hello.py"


def test_write_lazy_cwd_detection() -> None:
    """When cwd is not passed at construction, write() detects it via pwd."""
    cwd = "/opt/sandbox"
    sandbox, mock = _make_sandbox()
    mock.invoke.side_effect = _make_successful_invoke(cwd)

    result = sandbox.write("/workspace/hello.py", "hello")

    assert sandbox._cwd == cwd
    assert result.path == "/opt/sandbox/workspace/hello.py"


def test_write_returns_resolved_path_not_virtual_path() -> None:
    """WriteResult.path must be the resolved absolute path, not the virtual path.

    Regression test for the execute()-after-write() mismatch: when the LLM writes
    to "/tmp/script.py" and then runs execute("python /tmp/script.py"), it must
    use the same path that was returned by write() — otherwise the shell cannot
    find the file because AgentCore resolves uploads relative to cwd, not to "/".
    """
    cwd = "/opt/sandbox/var"
    sandbox, mock = _make_sandbox(cwd=cwd)
    mock.invoke.side_effect = _make_successful_invoke(cwd)

    result = sandbox.write("/tmp/script.py", "print('hello')")

    # The returned path must be the real location so execute() can find the file.
    assert result.path == f"{cwd}/tmp/script.py"
    write_files_calls = [
        c for c in mock.invoke.call_args_list if c.kwargs.get("method") == "writeFiles"
    ]
    uploaded_path = write_files_calls[0].kwargs["params"]["content"][0]["path"]
    # AgentCore receives a cwd-relative path, resolving to {cwd}/tmp/script.py.
    assert uploaded_path == "tmp/script.py"


def test_write_root_cwd_preserves_existing_behavior() -> None:
    """When cwd is /, write() should behave as it did before the fix."""
    sandbox, mock = _make_sandbox(cwd="/")
    mock.invoke.side_effect = _make_successful_invoke("/")

    result = sandbox.write("/hello.py", "hello")

    assert result.path == "/hello.py"
    write_files_calls = [
        c for c in mock.invoke.call_args_list if c.kwargs.get("method") == "writeFiles"
    ]
    uploaded_path = write_files_calls[0].kwargs["params"]["content"][0]["path"]
    assert uploaded_path == "hello.py"


# ------------------------------------------------------------------
# Read-plane path normalization (ls/read/grep/glob/edit)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "virtual_path", "expected_path", "extra_args", "return_value"),
    [
        (
            "ls",
            "/workspace",
            "/opt/sandbox/workspace",
            (),
            LsResult(entries=[]),
        ),
        (
            "read",
            "/workspace/hello.py",
            "/opt/sandbox/workspace/hello.py",
            (0, 2000),
            ReadResult(file_data=FileData(content="hello", encoding="utf-8")),
        ),
        (
            "grep",
            "/workspace",
            "/opt/sandbox/workspace",
            ("needle", None),
            GrepResult(matches=[]),
        ),
        (
            "glob",
            "/workspace",
            "/opt/sandbox/workspace",
            ("*.py",),
            GlobResult(matches=[]),
        ),
        (
            "edit",
            "/workspace/hello.py",
            "/opt/sandbox/workspace/hello.py",
            ("old", "new", False),
            EditResult(path="/opt/sandbox/workspace/hello.py", occurrences=1),
        ),
    ],
)
def test_read_plane_resolves_virtual_paths(
    method_name: str,
    virtual_path: str,
    expected_path: str,
    extra_args: tuple[Any, ...],
    return_value: Any,
) -> None:
    """Read-plane methods should resolve virtual paths against the sandbox cwd."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    result: Any

    with patch.object(
        BaseSandbox, method_name, return_value=return_value
    ) as mock_method:
        if method_name == "grep":
            result = sandbox.grep(extra_args[0], virtual_path)
            mock_method.assert_called_once_with(
                extra_args[0], expected_path, None, **_GREP_KW
            )
        elif method_name == "glob":
            result = sandbox.glob(extra_args[0], virtual_path)
            mock_method.assert_called_once_with(extra_args[0], expected_path)
        elif method_name == "read":
            result = sandbox.read(virtual_path, *extra_args)
            mock_method.assert_called_once_with(expected_path, *extra_args)
        elif method_name == "edit":
            result = sandbox.edit(virtual_path, *extra_args)
            mock_method.assert_called_once_with(expected_path, *extra_args)
        else:
            result = sandbox.ls(virtual_path)
            mock_method.assert_called_once_with(expected_path)

    assert result == return_value


def test_grep_with_none_path_passes_through() -> None:
    """grep() should not resolve when path is None (BaseSandbox defaults to '.')."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")

    with patch.object(
        BaseSandbox, "grep", return_value=GrepResult(matches=[])
    ) as mock_grep:
        sandbox.grep("needle")
        mock_grep.assert_called_once_with("needle", None, None, **_GREP_KW)


def test_glob_with_none_path_passes_through() -> None:
    """glob() should not resolve when path is None (BaseSandbox defaults to '/')."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")

    with patch.object(
        BaseSandbox, "glob", return_value=GlobResult(matches=[])
    ) as mock_glob:
        sandbox.glob("*.py")
        mock_glob.assert_called_once_with("*.py", None)


def test_read_root_cwd_preserves_virtual_path() -> None:
    """When cwd is /, read() should pass virtual paths through unchanged."""
    sandbox, _ = _make_sandbox(cwd="/")

    with patch.object(
        BaseSandbox,
        "read",
        return_value=ReadResult(file_data=FileData(content="ok", encoding="utf-8")),
    ) as mock_read:
        sandbox.read("/workspace/hello.py")
        mock_read.assert_called_once_with("/workspace/hello.py", 0, 2000)


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
    sandbox, mock = _make_sandbox(canned_response, cwd="/")

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


def test_als_routes_through_cwd_aware_sync_ls() -> None:
    """als() must resolve the virtual path against the cwd."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")

    with patch.object(BaseSandbox, "ls", return_value=LsResult(entries=[])) as mock_ls:
        asyncio.run(sandbox.als("/workspace"))
        mock_ls.assert_called_once_with("/opt/sandbox/workspace")


def test_agrep_routes_through_cwd_aware_sync_grep() -> None:
    """agrep() must resolve the virtual path against the cwd (not literal)."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")

    with patch.object(
        BaseSandbox, "grep", return_value=GrepResult(matches=[])
    ) as mock_grep:
        asyncio.run(sandbox.agrep("needle", "/workspace"))
        mock_grep.assert_called_once_with(
            "needle", "/opt/sandbox/workspace", None, **_GREP_KW
        )


def test_aglob_routes_through_cwd_aware_sync_glob() -> None:
    """aglob() must resolve the virtual path against the cwd (not literal)."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")

    with patch.object(
        BaseSandbox, "glob", return_value=GlobResult(matches=[])
    ) as mock_glob:
        asyncio.run(sandbox.aglob("*.py", "/workspace"))
        mock_glob.assert_called_once_with("*.py", "/opt/sandbox/workspace")


# ------------------------------------------------------------------
# Executor configuration
# ------------------------------------------------------------------


def test_dedicated_executor_exists() -> None:
    """The module-level executor should be configured."""
    assert _AGENTCORE_EXECUTOR is not None
    assert _AGENTCORE_EXECUTOR._max_workers == 16
    assert _AGENTCORE_EXECUTOR._thread_name_prefix == "agentcore-sandbox"


# ------------------------------------------------------------------
# Exit codes, in the shape the service returns them
# ------------------------------------------------------------------


def _result(exit_code: int | None, text: str = "", *, is_error: bool = False) -> dict:
    """A live-shaped executeCommand result: exit code under structuredContent."""
    content = [{"type": "text", "text": text}] if text else []
    structured: dict[str, Any] = {"stdout": text, "stderr": ""}
    if exit_code is not None:
        structured["exitCode"] = exit_code
    return {
        "stream": [
            {
                "result": {
                    "content": content,
                    "isError": is_error,
                    "structuredContent": structured,
                }
            }
        ]
    }


@pytest.mark.parametrize("code", [0, 1, 2, 3, 127])
def test_execute_reads_exit_code_from_structured_content(code: int) -> None:
    """The service reports exitCode only under structuredContent."""
    sandbox, _ = _make_sandbox(_result(code, is_error=code != 0))
    assert sandbox.execute("cmd").exit_code == code


def test_execute_uses_is_error_when_no_exit_code() -> None:
    """isError without any exit code still reports failure."""
    sandbox, _ = _make_sandbox(
        {"stream": [{"result": {"content": [], "isError": True}}]}
    )
    assert sandbox.execute("cmd").exit_code == 1


def test_write_refuses_existing_file() -> None:
    """write() must fail on an existing file.

    BaseSandbox decides this from the preflight command's exit code, so this
    only works once that exit code is read from the right field.
    """
    sandbox, mock = _make_sandbox(cwd="/work")
    mock.invoke.return_value = _result(1, "Error: File '/work/a.txt' already exists")
    result = sandbox.write("/a.txt", "x")
    assert result.error is not None
    assert not [
        c for c in mock.invoke.call_args_list if c.kwargs["method"] == "writeFiles"
    ]


# ------------------------------------------------------------------
# writeFiles failures returned as isError
# ------------------------------------------------------------------


def test_upload_files_reports_is_error_result() -> None:
    """A writeFiles result with isError must not be reported as success."""
    sandbox, _ = _make_sandbox(
        {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Error executing tool write_files: Invalid "
                                "file path: potential path traversal detected",
                            }
                        ],
                        "isError": True,
                    }
                }
            ]
        }
    )
    result = sandbox.upload_files([("../../etc/x", b"x")])
    assert result[0].error is not None


def test_non_write_is_error_does_not_raise() -> None:
    """Only writeFiles turns isError into an exception; commands report it."""
    sandbox, _ = _make_sandbox(_result(2, "ls: cannot access", is_error=True))
    result = sandbox.execute("ls /nope")
    assert result.exit_code == 2
    assert "cannot access" in result.output


# ------------------------------------------------------------------
# timeout
# ------------------------------------------------------------------


def test_execute_without_timeout_sends_command_unchanged() -> None:
    sandbox, mock = _make_sandbox(_result(0))
    sandbox.execute("echo hi")
    assert mock.invoke.call_args.kwargs["params"] == {"command": "echo hi"}


def test_execute_timeout_wraps_command() -> None:
    """timeout runs the command under GNU timeout, quoted as one argument."""
    sandbox, mock = _make_sandbox(_result(0))
    sandbox.execute("echo 'a b' && sleep 1", timeout=30)
    sent = mock.invoke.call_args.kwargs["params"]["command"]
    assert sent == "timeout -k 5 30s bash -c 'echo '\"'\"'a b'\"'\"' && sleep 1'"


def test_execute_zero_timeout_means_no_limit() -> None:
    sandbox, mock = _make_sandbox(_result(0))
    sandbox.execute("echo hi", timeout=0)
    assert mock.invoke.call_args.kwargs["params"] == {"command": "echo hi"}


def test_execute_timeout_expiry_is_explained() -> None:
    sandbox, _ = _make_sandbox(_result(124, is_error=True))
    result = sandbox.execute("sleep 99", timeout=2)
    assert result.exit_code == 124
    assert "timed out after 2 seconds" in result.output


def test_execute_exit_124_without_timeout_is_left_alone() -> None:
    sandbox, _ = _make_sandbox(_result(124, "own 124", is_error=True))
    result = sandbox.execute("exit 124")
    assert result.output == "own 124"


# ------------------------------------------------------------------
# grep max_count and delete, on deepagents versions that have them
# ------------------------------------------------------------------


@pytest.mark.skipif(not _BASE_GREP_HAS_MAX_COUNT, reason="deepagents without max_count")
def test_grep_forwards_max_count() -> None:
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    with patch.object(BaseSandbox, "grep", return_value=GrepResult(matches=[])) as g:
        sandbox.grep("needle", "/src", max_count=5)
        g.assert_called_once_with("needle", "/opt/sandbox/src", None, max_count=5)


@pytest.mark.skipif(not _BASE_GREP_HAS_MAX_COUNT, reason="deepagents without max_count")
def test_agrep_forwards_max_count() -> None:
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    with patch.object(BaseSandbox, "grep", return_value=GrepResult(matches=[])) as g:
        asyncio.run(sandbox.agrep("needle", "/src", max_count=5))
        g.assert_called_once_with("needle", "/opt/sandbox/src", None, max_count=5)


def test_grep_signature_advertises_max_count() -> None:
    """deepagents detects support from the signature, so it must be present."""
    import inspect

    assert "max_count" in inspect.signature(AgentCoreSandbox.grep).parameters
    assert "max_count" in inspect.signature(AgentCoreSandbox.agrep).parameters


@pytest.mark.skipif(not _BASE_HAS_DELETE, reason="deepagents without delete")
def test_delete_resolves_against_cwd() -> None:
    """delete() targets the same file read() would read."""
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    with patch.object(BaseSandbox, "delete", return_value=MagicMock()) as d:
        sandbox.delete("/notes.txt")
        d.assert_called_once_with("/opt/sandbox/notes.txt")


@pytest.mark.skipif(not _BASE_HAS_DELETE, reason="deepagents without delete")
def test_adelete_runs_on_dedicated_executor() -> None:
    sandbox, _ = _make_sandbox(cwd="/opt/sandbox")
    seen: list[str] = []

    def fake_delete(self: Any, path: str) -> Any:
        import threading

        seen.append(threading.current_thread().name)
        return MagicMock(path=path)

    with patch.object(BaseSandbox, "delete", fake_delete):
        asyncio.run(sandbox.adelete("/notes.txt"))
    assert seen and seen[0].startswith("agentcore-sandbox")


# ------------------------------------------------------------------
# create(), telemetry and lifecycle
# ------------------------------------------------------------------


def test_create_tags_session_and_starts_it() -> None:
    with patch(
        "bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter"
    ) as ci_cls:
        sandbox = AgentCoreSandbox.create(
            region="us-east-1", session_timeout_seconds=3600, cwd="/w"
        )
    ci_cls.assert_called_once_with(
        region="us-east-1", session=None, integration_source=INTEGRATION_SOURCE
    )
    ci_cls.return_value.start.assert_called_once_with(session_timeout_seconds=3600)
    assert sandbox._cwd == "/w"
    assert INTEGRATION_SOURCE == "langchain-deepagents"


def test_create_passes_identifier_and_session() -> None:
    session = object()
    with patch(
        "bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter"
    ) as ci_cls:
        AgentCoreSandbox.create(
            region="us-west-2", boto3_session=session, identifier="my-ci"
        )
    assert ci_cls.call_args.kwargs["session"] is session
    ci_cls.return_value.start.assert_called_once_with(
        session_timeout_seconds=900, identifier="my-ci"
    )


def test_context_manager_stops_owned_session_once() -> None:
    with patch(
        "bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter"
    ) as ci_cls:
        with AgentCoreSandbox.create(region="us-west-2") as sandbox:
            pass
        sandbox.close()
    ci_cls.return_value.stop.assert_called_once()


def test_async_context_manager_stops_owned_session() -> None:
    async def run() -> MagicMock:
        with patch(
            "bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter"
        ) as ci_cls:
            async with AgentCoreSandbox.create(region="us-west-2"):
                pass
            return ci_cls

    ci_cls = asyncio.run(run())
    ci_cls.return_value.stop.assert_called_once()


def test_caller_supplied_interpreter_is_left_running() -> None:
    sandbox, mock = _make_sandbox()
    with sandbox:
        pass
    mock.stop.assert_not_called()


def test_execute_undoes_pty_line_endings() -> None:
    """The service's pseudo-terminal turns "\\n" into "\\r\\n"; execute() undoes it."""
    sandbox, _ = _make_sandbox(_result(0, "Line 1\r\nLine 2\r\n"))
    assert sandbox.execute("cat f").output == "Line 1\nLine 2\n"


def test_execute_preserves_real_crlf() -> None:
    """A file's own "\\r\\n" arrives as "\\r\\r\\n" and must come back as "\\r\\n"."""
    sandbox, _ = _make_sandbox(_result(0, "a\r\r\nb\r\r\n"))
    assert sandbox.execute("cat crlf.txt").output == "a\r\nb\r\n"
