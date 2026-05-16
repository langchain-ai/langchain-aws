"""Amazon Bedrock AgentCore Code Interpreter sandbox backend implementation."""

from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

logger = logging.getLogger(__name__)

# Dedicated thread pool for AgentCore boto3 calls. Isolates sandbox I/O
# from the default asyncio executor so long-running stream reads don't
# starve other async work (LLM calls, tool dispatch, etc.).
_AGENTCORE_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="agentcore-sandbox"
)


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

        if "exitCode" in result:
            exit_code = result["exitCode"]

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

    Example:
        .. code-block:: python

            from bedrock_agentcore.tools.code_interpreter_client import (
                CodeInterpreter,
            )
            from langchain_agentcore_codeinterpreter import AgentCoreSandbox

            interpreter = CodeInterpreter(region="us-west-2")
            interpreter.start()

            backend = AgentCoreSandbox(interpreter=interpreter)
            result = backend.execute("echo hello")
            print(result.output)

            interpreter.stop()
    """

    def __init__(self, *, interpreter: CodeInterpreter) -> None:
        """Create a backend wrapping an active CodeInterpreter session.

        Args:
            interpreter: A started :class:`CodeInterpreter` instance.
        """
        self._interpreter = interpreter

    @staticmethod
    def _to_relative_path(path: str) -> str:
        """Strip leading slashes and ``./`` prefixes for AgentCore APIs.

        Args:
            path: File path (absolute or relative).

        Returns:
            Relative path string.
        """
        return _normalize_relative_path(path)

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
        timeout: int | None = None,  # noqa: ARG002
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Unused. AgentCore does not support per-command timeouts.
                Accepted for interface compatibility with
                :class:`SandboxBackendProtocol`.

        Returns:
            Response containing the command output, exit code, and truncation
            flag.
        """
        try:
            response = self._invoke(
                method="executeCommand", params={"command": command}
            )
            output, exit_code = _extract_text_from_stream(response)
            return ExecuteResponse(
                output=output,
                exit_code=exit_code if exit_code is not None else 0,
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
            relative_paths = [self._to_relative_path(p) for p in paths]
            response = self._invoke(
                method="readFiles", params={"paths": relative_paths}
            )
            file_contents = _extract_files_from_stream(response, paths)

            return [
                FileDownloadResponse(
                    path=path,
                    content=file_contents.get(path),
                    error=None if path in file_contents else "file_not_found",
                )
                for path in paths
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

        Text files are sent directly; binary files are base64-encoded.

        Args:
            files: List of ``(path, content)`` tuples to upload.

        Returns:
            List of :class:`FileUploadResponse` objects in the same order
            as the input files.
        """
        file_list: list[dict[str, str]] = []

        for path, content in files:
            rel_path = self._to_relative_path(path)
            try:
                text_content = content.decode("utf-8")
                file_list.append({"path": rel_path, "text": text_content})
            except UnicodeDecodeError:
                encoded = base64.b64encode(content).decode("ascii")
                file_list.append({"path": rel_path, "blob": encoded})

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
