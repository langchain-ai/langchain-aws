"""Amazon Bedrock AgentCore Code Interpreter sandbox backend implementation."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

logger = logging.getLogger(__name__)


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
        stripped = path.lstrip("/")
        path_lookup[stripped] = path

    files: dict[str, bytes] = {}

    for event in response.get("stream", []):
        if "result" not in event:
            continue
        for item in event["result"].get("content", []):
            if item.get("type") != "resource":
                continue
            resource = item.get("resource", {})
            uri = resource.get("uri", "")
            file_path = uri.replace("file://", "").lstrip("/")

            content: bytes | None = None
            if "text" in resource:
                content = resource["text"].encode("utf-8")
            elif "blob" in resource:
                content = base64.b64decode(resource["blob"])

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
        """Strip leading slashes so paths are relative for AgentCore APIs.

        Args:
            path: File path (absolute or relative).

        Returns:
            Relative path string.
        """
        return path.lstrip("/")

    @property
    def id(self) -> str:
        """Return the AgentCore session ID."""
        return self._interpreter.session_id or ""

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
            response = self._interpreter.invoke(
                method="executeCommand", params={"command": command}
            )
            output, exit_code = _extract_text_from_stream(response)
            return ExecuteResponse(
                output=output,
                exit_code=exit_code if exit_code is not None else 0,
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
            response = self._interpreter.invoke(
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
                self._interpreter.invoke(
                    method="writeFiles", params={"content": file_list}
                )
            return [FileUploadResponse(path=path, error=None) for path, _ in files]
        except Exception:
            logger.exception("Error uploading files: %s", [p for p, _ in files])
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _ in files
            ]
