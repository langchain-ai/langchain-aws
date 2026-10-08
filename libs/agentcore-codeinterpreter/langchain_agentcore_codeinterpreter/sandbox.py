"""Amazon Bedrock AgentCore Code Interpreter sandbox backend implementation."""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from types import TracebackType

    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

logger = logging.getLogger(__name__)

# Dedicated thread pool for AgentCore boto3 calls. Isolates sandbox I/O
# from the default asyncio executor so long-running stream reads don't
# starve other async work (LLM calls, tool dispatch, etc.). Shared by every
# sandbox in the process, so it is sized for several subagents working at
# once; InvokeCodeInterpreter is throttled at 30 TPS per account, which
# bounds how much a larger pool could help.
_AGENTCORE_EXECUTOR = ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="agentcore-sandbox"
)

#: Reported to the AgentCore SDK so usage from Deep Agents is attributed
#: separately from the ``langchain`` source the langchain-aws toolkits use.
INTEGRATION_SOURCE = "langchain-deepagents"

#: Exit status GNU ``timeout`` returns when it stops a command.
_TIMEOUT_EXIT_CODE = 124

# Newer deepagents releases add methods and parameters to BaseSandbox. The
# overrides below only forward to them when the installed version has them,
# so the package keeps working on the minimum supported deepagents.
_BASE_HAS_DELETE = hasattr(BaseSandbox, "delete")
_BASE_GREP_HAS_MAX_COUNT = "max_count" in inspect.signature(BaseSandbox.grep).parameters


class _ToolError(Exception):
    """A tool call the service answered with ``isError`` instead of raising."""


def _normalize_relative_path(path: str) -> str:
    """Strip leading slashes and ``./`` prefixes to a canonical relative path.

    Args:
        path: File path (absolute or relative).

    Returns:
        Canonical relative path string with no leading ``/`` or ``./``.
    """
    path = path.lstrip("/")
    while path.startswith("./"):
        path = path[2:]
    return path


class SessionExpiredError(Exception):
    """Raised when the AgentCore session has expired or been terminated."""

    def __init__(self, session_id: str, original: ClientError) -> None:
        self.session_id = session_id
        self.original = original
        super().__init__(
            f"AgentCore session '{session_id}' has expired or was terminated. "
            f"Start a new session to continue."
        )


def _extract_text_from_stream(response: dict[str, Any]) -> tuple[str, int | None]:
    """Extract text output and exit code from a code interpreter response stream.

    Iterates through the streamed response events and collects text content,
    error messages, and the exit code.

    The service reports the exit code in ``result["structuredContent"]["exitCode"]``,
    which is where the InvokeCodeInterpreter API reference documents it. A
    top-level ``result["exitCode"]`` is still read as a fallback. When neither is
    present, ``isError`` is used to tell failure from an undetermined outcome.

    Args:
        response: Response dict from a code interpreter invocation.

    Returns:
        Tuple of (output_text, exit_code). The exit code is ``None`` when
        the response stream does not include one.
    """
    output_parts: list[str] = []
    exit_code: int | None = None

    for event in response.get("stream", []):
        if "result" not in event:
            continue

        result = event["result"]

        structured = result.get("structuredContent") or {}
        if "exitCode" in structured:
            exit_code = structured["exitCode"]
        elif "exitCode" in result:
            exit_code = result["exitCode"]
        elif result.get("isError") and exit_code is None:
            exit_code = 1

        for content_item in result.get("content", []):
            content_type = content_item.get("type")

            if content_type == "text":
                text = content_item.get("text", "")
                output_parts.append(text)
            elif content_type == "error":
                error_msg = content_item.get("text", "Unknown error")
                output_parts.append(f"Error: {error_msg}")
                if exit_code is None:
                    exit_code = 1

    return "\n".join(output_parts), exit_code


def _extract_files_from_stream(
    response: dict[str, Any],
    requested_paths: list[str],
) -> dict[str, bytes]:
    """Extract file contents from a code interpreter ``readFiles`` response.

    Matches ``file://`` URIs in the response back to the original requested
    paths by stripping leading slashes for comparison.

    Args:
        response: Response dict from a code interpreter ``readFiles``
            invocation.
        requested_paths: The original paths that were requested, used to
            map URIs back to caller-provided names.

    Returns:
        Dict mapping original requested paths to their contents as bytes.
    """
    path_lookup: dict[str, str] = {}
    for path in requested_paths:
        path_lookup[_normalize_relative_path(path)] = path

    files: dict[str, bytes] = {}

    for event in response.get("stream", []):
        if "result" not in event:
            continue
        for item in event["result"].get("content", []):
            if item.get("type") != "resource":
                continue
            resource = item.get("resource", {})
            uri = resource.get("uri", "")
            file_path = _normalize_relative_path(uri.replace("file://", ""))

            content: bytes | None = None
            if "text" in resource:
                content = resource["text"].encode("utf-8")
            elif "blob" in resource:
                blob = resource["blob"]
                # The AgentCore stream may deliver blob as already-decoded bytes.
                # Only base64-decode when it arrives as encoded text.
                content = blob if isinstance(blob, bytes) else base64.b64decode(blob)

            if content is not None:
                original_path = path_lookup.get(file_path, file_path)
                files[original_path] = content

    return files


class AgentCoreSandbox(BaseSandbox):
    """AgentCore Code Interpreter sandbox conforming to SandboxBackendProtocol.

    Wraps an active :class:`CodeInterpreter` session to execute shell commands
    and manage files in a secure, isolated MicroVM environment.

    This implementation inherits all file operation methods from
    :class:`BaseSandbox` and implements the required ``execute()``,
    ``download_files()``, and ``upload_files()`` methods using AgentCore's
    streaming API.

    Async methods (``aexecute``, ``awrite``, etc.) use a dedicated thread
    pool executor to avoid blocking the default ``asyncio`` executor with
    long-running boto3 stream reads.

    The caller is responsible for managing the interpreter lifecycle
    (``start()`` / ``stop()``).

    !!! note

        When the sandbox working directory is not ``/``, paths must be resolved
        against the real cwd before shell preflight commands and stripped of
        the cwd prefix before the AgentCore ``writeFiles``/``readFiles`` APIs.
        Pass the known cwd via the ``cwd`` constructor argument, or let the
        sandbox detect it automatically on the first path operation via ``pwd``.

    Example:
        .. code-block:: python

            from langchain_agentcore_codeinterpreter import AgentCoreSandbox

            with AgentCoreSandbox.create(region="us-west-2") as backend:
                result = backend.execute("echo hello")
                print(result.output)
    """

    def __init__(
        self,
        *,
        interpreter: CodeInterpreter,
        cwd: str | None = None,
    ) -> None:
        """Create a backend wrapping an active CodeInterpreter session.

        Prefer :meth:`create`, which starts the session, tags it for usage
        attribution and stops it on exit. Use this constructor when you need
        to configure the :class:`CodeInterpreter` yourself; its lifecycle then
        stays with you.

        Args:
            interpreter: A started :class:`CodeInterpreter` instance.
            cwd: The sandbox working directory. When provided,
                ``write()`` uses it to resolve virtual paths to real absolute
                paths and to strip the prefix before the AgentCore
                ``writeFiles`` API. When omitted, the cwd is detected
                automatically on the first path operation via ``pwd``.
        """
        self._interpreter = interpreter
        self._cwd: str | None = cwd.rstrip("/") if cwd is not None else None
        self._owns_interpreter = False

    @classmethod
    def create(
        cls,
        *,
        region: str,
        boto3_session: Any = None,
        identifier: str | None = None,
        session_timeout_seconds: int = 900,
        cwd: str | None = None,
    ) -> AgentCoreSandbox:
        """Start a Code Interpreter session and return a sandbox that owns it.

        The session is tagged so AgentCore can attribute its usage to Deep
        Agents. Closing the sandbox, directly or by leaving a ``with`` block,
        stops the session.

        Args:
            region: AWS region for the Code Interpreter.
            boto3_session: Session to take credentials from. Defaults to the
                standard boto3 credential chain.
            identifier: Code interpreter to start a session on. Defaults to
                the built-in ``aws.codeinterpreter.v1``; pass a custom one
                for VPC networking or an execution role.
            session_timeout_seconds: Session lifetime. AgentCore's default is
                15 minutes, which long agent runs can exceed; the maximum is
                28,800 (8 hours).
            cwd: The sandbox working directory, if already known.

        Returns:
            A sandbox wrapping the started session.
        """
        from bedrock_agentcore.tools.code_interpreter_client import (
            CodeInterpreter,
        )

        interpreter = CodeInterpreter(
            region=region,
            session=boto3_session,
            integration_source=INTEGRATION_SOURCE,
        )
        start_kwargs: dict[str, Any] = {
            "session_timeout_seconds": session_timeout_seconds
        }
        if identifier is not None:
            start_kwargs["identifier"] = identifier
        interpreter.start(**start_kwargs)
        sandbox = cls(interpreter=interpreter, cwd=cwd)
        sandbox._owns_interpreter = True
        return sandbox

    def close(self) -> None:
        """Stop the session if this sandbox started it through :meth:`create`.

        A sandbox built from a caller-supplied interpreter leaves it running,
        since the caller manages that lifecycle.
        """
        if self._owns_interpreter:
            self._owns_interpreter = False
            self._interpreter.stop()

    def __enter__(self) -> AgentCoreSandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> AgentCoreSandbox:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_AGENTCORE_EXECUTOR, self.close)

    def _get_cwd(self) -> str:
        """Return the sandbox working directory, detecting it lazily if needed.

        Returns:
            The working directory with any trailing slash stripped.
        """
        if self._cwd is None:
            result = self.execute("pwd")
            if result.exit_code not in (0, None) or not result.output.strip():
                raise RuntimeError(
                    f"Failed to detect sandbox working directory: "
                    f"exit_code={result.exit_code}, output={result.output!r}"
                )
            self._cwd = result.output.strip().rstrip("/")
        return self._cwd

    def _to_relative_path(self, path: str) -> str:
        """Strip the cwd prefix (or leading slashes) for AgentCore file APIs.

        When the sandbox cwd is known and ``path`` starts with it, the cwd
        prefix is removed so the AgentCore ``writeFiles``/``readFiles`` APIs
        receive a cwd-relative path. For paths that do not start with the cwd,
        the standard leading-slash stripping is applied.

        Args:
            path: File path (absolute or relative).

        Returns:
            Path relative to the sandbox cwd, with no leading ``/`` or ``./``.
        """
        if self._cwd is not None:
            cwd_prefix = self._cwd + "/"
            if path.startswith(cwd_prefix):
                return path[len(cwd_prefix) :]
        return _normalize_relative_path(path)

    def _to_absolute_path(self, path: str) -> str:
        """Resolve a path to a real absolute path under the sandbox cwd.

        Paths already under the cwd are returned unchanged. Virtual paths
        (e.g. ``/workspace/hello.py``) and relative paths are prepended with
        the cwd so that shell commands (``makedirs``, etc.) operate on the
        real filesystem location.

        Args:
            path: File path to resolve.

        Returns:
            Absolute path under the sandbox cwd.
        """
        cwd = self._get_cwd()
        if not cwd:
            return path
        if path.startswith(cwd + "/") or path == cwd:
            return path
        return cwd + "/" + path.lstrip("/")

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file in the sandbox, failing if it already exists.

        Normalizes ``file_path`` to a real absolute path under the sandbox cwd
        before the preflight shell command (so ``makedirs`` operates on the
        correct location) and before uploading (so the AgentCore ``writeFiles``
        API receives a cwd-relative path rather than a doubled absolute path).

        Args:
            file_path: Destination path for the new file. May be a real
                absolute path under the sandbox cwd, a virtual absolute path
                (e.g. ``/workspace/hello.py``), or a relative path.
            content: UTF-8 text content to write.

        Returns:
            ``WriteResult`` with ``path`` set to the resolved absolute path on
            success, or ``error`` on failure.
        """
        abs_path = self._to_absolute_path(file_path)
        preflight_error = self._write_preflight(abs_path)
        if preflight_error is not None:
            return preflight_error
        responses = self.upload_files([(abs_path, content.encode("utf-8"))])
        assert responses, "upload_files returned no responses"
        response = responses[0]
        if response.error:
            return WriteResult(
                error=f"Failed to write file '{abs_path}': {response.error}"
            )
        return WriteResult(path=abs_path)

    def _invoke(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke the interpreter and eagerly consume the response stream.

        AgentCore's ``invoke_code_interpreter`` returns a lazy EventStream
        that holds the HTTP connection open until fully iterated. Consuming
        it eagerly releases the connection promptly, which prevents thread
        starvation when multiple sandbox calls are in-flight under
        ``asyncio.to_thread`` or a thread pool executor.

        Args:
            method: The interpreter method name (e.g. ``executeCommand``).
            params: Parameters to pass to the method.

        Returns:
            Response dict with the ``"stream"`` key materialized as a list.

        Raises:
            SessionExpiredError: If the session has expired or been terminated.
            _ToolError: If ``writeFiles`` reports failure. The service returns
                such failures as an ``isError`` result rather than an
                exception, so without this an upload that wrote nothing would
                be reported as a success.
        """
        try:
            response = self._interpreter.invoke(method=method, params=params)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                raise SessionExpiredError(self.id, exc) from exc
            raise

        # Eagerly consume the lazy EventStream to release the HTTP connection.
        if "stream" in response:
            response["stream"] = list(response["stream"])

        if method == "writeFiles":
            for event in response.get("stream", []):
                result = event.get("result") or {}
                if result.get("isError"):
                    text, _ = _extract_text_from_stream({"stream": [event]})
                    raise _ToolError(text or "writeFiles failed")

        return response

    @property
    def id(self) -> str:
        """Return the AgentCore session ID."""
        return self._interpreter.session_id or ""

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Seconds to let the command run before it is stopped.
                ``None`` or ``0`` means no limit beyond the service's own
                request timeout. Enforced with GNU ``timeout`` inside the
                sandbox, which sends SIGTERM and then SIGKILL five seconds
                later.

        Returns:
            Response containing the command output, exit code, and truncation
            flag. ``exit_code`` is ``None`` when the service did not report
            one, which Deep Agents treats as an unknown outcome rather than a
            success.
        """
        to_run = command
        if timeout:
            # ``sh`` rather than ``bash``: commands normally run under /bin/sh,
            # and ``bash -c`` would leave POSIX mode and change their behavior.
            to_run = f"timeout -k 5 {int(timeout)}s sh -c {shlex.quote(command)}"
        try:
            started = time.monotonic()
            response = self._invoke(method="executeCommand", params={"command": to_run})
            elapsed = time.monotonic() - started
            output, exit_code = _extract_text_from_stream(response)
            # executeCommand runs under a pseudo-terminal, which turns every
            # "\n" the command writes into "\r\n". Undoing exactly that
            # mapping restores the original bytes, including any "\r\n" a
            # file really contains (the terminal renders those as "\r\r\n").
            output = output.replace("\r\n", "\n")
            # A command can exit 124 on its own, so only call it a timeout when
            # the call also lasted at least as long as the limit.
            if timeout and exit_code == _TIMEOUT_EXIT_CODE and elapsed >= int(timeout):
                note = f"Command timed out after {int(timeout)} seconds."
                output = f"{output}\n{note}" if output else note
            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=False,
            )
        except SessionExpiredError:
            logger.error(
                "AgentCore session expired while executing command: %s",
                command[:80],
            )
            return ExecuteResponse(
                output=(
                    "Error: AgentCore session has expired. "
                    "Start a new session to continue."
                ),
                exit_code=1,
                truncated=False,
            )
        except Exception as exc:
            logger.exception("Error executing command: %s", command[:80])
            msg = f"Error executing command: {exc}"
            return ExecuteResponse(
                output=msg,
                exit_code=1,
                truncated=False,
            )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the AgentCore sandbox.

        Uses AgentCore's ``readFiles`` API. Supports partial success —
        individual file downloads may fail without affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of :class:`FileDownloadResponse` objects in the same order
            as the input paths.
        """
        try:
            self._get_cwd()
            relative_paths = [self._to_relative_path(p) for p in paths]
            response = self._invoke(
                method="readFiles", params={"paths": relative_paths}
            )
            file_contents = _extract_files_from_stream(response, relative_paths)

            return [
                FileDownloadResponse(
                    path=original,
                    content=file_contents.get(rel),
                    error=None if rel in file_contents else "file_not_found",
                )
                for original, rel in zip(paths, relative_paths)
            ]
        except SessionExpiredError:
            logger.error("AgentCore session expired while downloading files: %s", paths)
            return [
                FileDownloadResponse(path=path, content=None, error="permission_denied")
                for path in paths
            ]
        except Exception:
            logger.exception("Error downloading files: %s", paths)
            return [
                FileDownloadResponse(path=path, content=None, error="file_not_found")
                for path in paths
            ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the AgentCore sandbox.

        Text files are sent directly; binary files are sent as raw blobs.

        Args:
            files: List of ``(path, content)`` tuples to upload.

        Returns:
            List of :class:`FileUploadResponse` objects in the same order
            as the input files.
        """
        file_list: list[dict[str, str | bytes]] = []

        for path, content in files:
            rel_path = self._to_relative_path(path)
            try:
                text_content = content.decode("utf-8")
                file_list.append({"path": rel_path, "text": text_content})
            except UnicodeDecodeError:
                file_list.append({"path": rel_path, "blob": content})

        try:
            if file_list:
                self._invoke(method="writeFiles", params={"content": file_list})
            return [FileUploadResponse(path=path, error=None) for path, _ in files]
        except SessionExpiredError:
            logger.error(
                "AgentCore session expired while uploading files: %s",
                [p for p, _ in files],
            )
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _ in files
            ]
        except Exception:
            logger.exception("Error uploading files: %s", [p for p, _ in files])
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _ in files
            ]

    def ls(self, path: str) -> LsResult:
        """List a directory, resolving ``path`` against the sandbox cwd."""
        return super().ls(self._to_absolute_path(path))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read a file, resolving ``file_path`` against the sandbox cwd."""
        return super().read(self._to_absolute_path(file_path), offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents, resolving ``path`` against the sandbox cwd.

        ``max_count`` is passed to ``BaseSandbox`` so the search stops early
        inside the sandbox, rather than returning every match for Deep Agents
        to trim afterwards. Older deepagents releases without the parameter
        ignore it.
        """
        resolved = self._to_absolute_path(path) if path else path
        if _BASE_GREP_HAS_MAX_COUNT:
            # Only reached on deepagents releases that accept max_count; the
            # type checker sees whichever version is locked.
            return super().grep(pattern, resolved, glob, max_count=max_count)  # type: ignore[call-arg]
        return super().grep(pattern, resolved, glob)

    if _BASE_HAS_DELETE:

        def delete(self, file_path: str) -> Any:
            """Delete a file, resolving ``file_path`` against the sandbox cwd.

            Without this, ``BaseSandbox`` would run the delete on the raw path,
            so ``delete("/notes.txt")`` targeted ``/notes.txt`` while
            ``read("/notes.txt")`` read ``<cwd>/notes.txt``.
            """
            return super().delete(self._to_absolute_path(file_path))  # type: ignore[misc]

        async def adelete(self, file_path: str) -> Any:
            """Async version of :meth:`delete`, run on the sandbox executor."""
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _AGENTCORE_EXECUTOR,
                lambda: self.delete(file_path),
            )

    def glob(
        self,
        pattern: str,
        path: str | None = None,
    ) -> GlobResult:
        """Match paths by glob, resolving ``path`` against the sandbox cwd."""
        resolved = self._to_absolute_path(path) if path else path
        return super().glob(pattern, resolved)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file, resolving ``file_path`` against the sandbox cwd."""
        return super().edit(
            self._to_absolute_path(file_path),
            old_string,
            new_string,
            replace_all,
        )

    # ------------------------------------------------------------------
    # Async overrides — use a dedicated executor to avoid starving the
    # default asyncio thread pool with long-running boto3 stream reads.
    # ------------------------------------------------------------------

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Async version of :meth:`execute`.

        Runs the sync method in a dedicated thread pool executor to avoid
        blocking the default ``asyncio`` executor.

        Args:
            command: Shell command string to execute.
            timeout: Unused. Accepted for interface compatibility.

        Returns:
            Response containing the command output, exit code, and truncation
            flag.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.execute(command, timeout=timeout),
        )

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of :meth:`read`.

        Runs the sync method in a dedicated thread pool executor.

        Args:
            file_path: Absolute path to the file to read.
            offset: Starting line number (0-indexed).
            limit: Maximum number of lines to return.

        Returns:
            ``ReadResult`` with ``file_data`` on success or ``error`` on
            failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.read(file_path, offset, limit),
        )

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of :meth:`write`.

        Runs the sync method in a dedicated thread pool executor.

        Args:
            file_path: Absolute path for the new file.
            content: UTF-8 text content to write.

        Returns:
            ``WriteResult`` with ``path`` on success or ``error`` on failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.write(file_path, content),
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of :meth:`edit`.

        Runs the sync method in a dedicated thread pool executor.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: The exact substring to find.
            new_string: The replacement string.
            replace_all: If ``True``, replace every occurrence.

        Returns:
            ``EditResult`` with ``path`` and ``occurrences`` on success,
            or ``error`` on failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.edit(file_path, old_string, new_string, replace_all),
        )

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Async version of :meth:`upload_files`.

        Runs the sync method in a dedicated thread pool executor.

        Args:
            files: List of ``(path, content)`` tuples to upload.

        Returns:
            List of :class:`FileUploadResponse` objects.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.upload_files(files),
        )

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of :meth:`download_files`.

        Runs the sync method in a dedicated thread pool executor.

        Args:
            paths: List of file paths to download.

        Returns:
            List of :class:`FileDownloadResponse` objects.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.download_files(paths),
        )

    async def als(self, path: str) -> LsResult:
        """Async version of :meth:`ls`.

        Args:
            path: Directory path to list, resolved against the sandbox cwd.

        Returns:
            ``LsResult`` with directory entries on success or ``error`` on
            failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.ls(path),
        )

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Async version of :meth:`grep`.

        Args:
            pattern: Literal string to search for.
            path: Directory or file to search in, resolved against the sandbox
                cwd. When ``None``, the ``BaseSandbox`` default is used.
            glob: Optional file-name glob to restrict the search.
            max_count: Stop after this many matches, when supported by the
                installed deepagents.

        Returns:
            ``GrepResult`` with a list of matches or ``error`` on failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.grep(pattern, path, glob, max_count=max_count),
        )

    async def aglob(
        self,
        pattern: str,
        path: str | None = None,
    ) -> GlobResult:
        """Async version of :meth:`glob`.

        Args:
            pattern: Glob pattern to match.
            path: Directory to search in, resolved against the sandbox cwd.
                When ``None``, the ``BaseSandbox`` default is used.

        Returns:
            ``GlobResult`` with a list of matches or ``error`` on failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _AGENTCORE_EXECUTOR,
            lambda: self.glob(pattern, path),
        )
