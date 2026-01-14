"""Unit tests for code interpreter toolkit and tools."""

import pytest

# Skip all tests in this module if optional dependencies are not installed
pytest.importorskip("bedrock_agentcore", reason="Requires langchain-aws[tools]")

from typing import cast
from unittest.mock import MagicMock, patch

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool


class TestCodeInterpreterToolkit:
    """Tests for CodeInterpreterToolkit class."""

    @pytest.mark.asyncio
    async def test_create_code_interpreter_toolkit(self) -> None:
        """Test create_code_interpreter_toolkit factory function."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            create_code_interpreter_toolkit,
        )

        toolkit, tools = await create_code_interpreter_toolkit(region="us-west-2")

        assert toolkit is not None
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_toolkit_initializes_with_region(self) -> None:
        """Test toolkit initializes with specified region."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit(region="us-east-1")

        assert toolkit.region == "us-east-1"
        assert toolkit._code_interpreters == {}

    def test_toolkit_default_region(self) -> None:
        """Test toolkit uses default region."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        assert toolkit.region == "us-west-2"

    def test_get_tools_returns_list(self) -> None:
        """Test get_tools returns list of tools."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()
        # Note: tools list is empty until _setup() is called
        tools = toolkit.get_tools()

        assert isinstance(tools, list)

    def test_get_tools_by_name_returns_dict(self) -> None:
        """Test get_tools_by_name returns dictionary."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()
        tools_dict = toolkit.get_tools_by_name()

        assert isinstance(tools_dict, dict)

    @pytest.mark.asyncio
    async def test_expected_tools_present(self) -> None:
        """Test all expected code interpreter tools are present."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            create_code_interpreter_toolkit,
        )

        toolkit, tools = await create_code_interpreter_toolkit(region="us-west-2")
        tools_dict = toolkit.get_tools_by_name()

        expected_tools = [
            "execute_code",
            "execute_command",
            "read_files",
            "write_files",
            "list_files",
            "delete_files",
            "upload_file",
            "install_packages",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools_dict, f"Missing tool: {tool_name}"

    @pytest.mark.asyncio
    async def test_cleanup_all_sessions(self) -> None:
        """Test cleanup stops all code interpreter sessions."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        # Add mock interpreters
        mock_interpreter1 = MagicMock()
        mock_interpreter1.stop = MagicMock()
        mock_interpreter2 = MagicMock()
        mock_interpreter2.stop = MagicMock()

        toolkit._code_interpreters = {
            "thread-1": mock_interpreter1,
            "thread-2": mock_interpreter2,
        }

        await toolkit.cleanup()

        mock_interpreter1.stop.assert_called_once()
        mock_interpreter2.stop.assert_called_once()
        assert toolkit._code_interpreters == {}

    @pytest.mark.asyncio
    async def test_cleanup_specific_thread(self) -> None:
        """Test cleanup can stop specific thread's session."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        mock_interpreter1 = MagicMock()
        mock_interpreter1.stop = MagicMock()
        mock_interpreter2 = MagicMock()
        mock_interpreter2.stop = MagicMock()

        toolkit._code_interpreters = {
            "thread-1": mock_interpreter1,
            "thread-2": mock_interpreter2,
        }

        await toolkit.cleanup(thread_id="thread-1")

        mock_interpreter1.stop.assert_called_once()
        mock_interpreter2.stop.assert_not_called()
        assert "thread-1" not in toolkit._code_interpreters
        assert "thread-2" in toolkit._code_interpreters


class TestCodeInterpreterToolInputSchemas:
    """Tests for code interpreter tool input schemas."""

    def test_execute_code_input(self) -> None:
        """Test ExecuteCodeInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import ExecuteCodeInput

        input_data = ExecuteCodeInput(code="print('hello')")
        assert input_data.code == "print('hello')"
        assert input_data.language == "python"
        assert input_data.clear_context is False

    def test_execute_code_input_with_language(self) -> None:
        """Test ExecuteCodeInput with different language."""
        from langchain_aws.tools.code_interpreter_toolkit import ExecuteCodeInput

        input_data = ExecuteCodeInput(
            code="console.log('hello')", language="javascript"
        )
        assert input_data.language == "javascript"

    def test_execute_command_input(self) -> None:
        """Test ExecuteCommandInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import ExecuteCommandInput

        input_data = ExecuteCommandInput(command="ls -la")
        assert input_data.command == "ls -la"

    def test_read_files_input(self) -> None:
        """Test ReadFilesInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import ReadFilesInput

        input_data = ReadFilesInput(paths=["file1.txt", "file2.txt"])
        assert input_data.paths == ["file1.txt", "file2.txt"]

    def test_write_files_input(self) -> None:
        """Test WriteFilesInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import WriteFilesInput

        files = [{"path": "test.txt", "text": "content"}]
        input_data = WriteFilesInput(files=files)
        assert input_data.files == files

    def test_list_files_input_defaults(self) -> None:
        """Test ListFilesInput default values."""
        from langchain_aws.tools.code_interpreter_toolkit import ListFilesInput

        input_data = ListFilesInput()
        assert input_data.directory_path == ""

    def test_delete_files_input(self) -> None:
        """Test DeleteFilesInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import DeleteFilesInput

        input_data = DeleteFilesInput(paths=["file1.txt", "file2.txt"])
        assert input_data.paths == ["file1.txt", "file2.txt"]

    def test_upload_file_input(self) -> None:
        """Test UploadFileInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import UploadFileInput

        input_data = UploadFileInput(
            path="data.csv",
            content="a,b,c\n1,2,3",
            description="Test CSV file",
        )
        assert input_data.path == "data.csv"
        assert input_data.content == "a,b,c\n1,2,3"
        assert input_data.description == "Test CSV file"

    def test_upload_file_input_default_description(self) -> None:
        """Test UploadFileInput default description."""
        from langchain_aws.tools.code_interpreter_toolkit import UploadFileInput

        input_data = UploadFileInput(path="file.txt", content="hello")
        assert input_data.description == ""

    def test_install_packages_input(self) -> None:
        """Test InstallPackagesInput schema."""
        from langchain_aws.tools.code_interpreter_toolkit import InstallPackagesInput

        input_data = InstallPackagesInput(packages=["pandas", "numpy"])
        assert input_data.packages == ["pandas", "numpy"]
        assert input_data.upgrade is False

    def test_install_packages_input_with_upgrade(self) -> None:
        """Test InstallPackagesInput with upgrade flag."""
        from langchain_aws.tools.code_interpreter_toolkit import InstallPackagesInput

        input_data = InstallPackagesInput(packages=["pandas"], upgrade=True)
        assert input_data.upgrade is True


class TestGetOrCreateInterpreter:
    """Tests for _get_or_create_interpreter method."""

    def test_creates_new_interpreter(self) -> None:
        """Test creates new interpreter for new thread."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        with patch(
            "langchain_aws.tools.code_interpreter_toolkit.CodeInterpreter"
        ) as mock_class:
            mock_interpreter = MagicMock()
            mock_interpreter.start = MagicMock()
            mock_interpreter.session_id = "test-session"
            mock_class.return_value = mock_interpreter

            config: RunnableConfig = cast(
                RunnableConfig, {"configurable": {"thread_id": "thread-1"}}
            )
            interpreter = toolkit._get_or_create_interpreter(config)

            assert interpreter is mock_interpreter
            mock_class.assert_called_once_with(
                region="us-west-2",
                integration_source="langchain"
            )
            mock_interpreter.start.assert_called_once()
            assert "thread-1" in toolkit._code_interpreters

    def test_reuses_existing_interpreter(self) -> None:
        """Test reuses existing interpreter for same thread."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        mock_interpreter = MagicMock()
        toolkit._code_interpreters["thread-1"] = mock_interpreter

        config: RunnableConfig = cast(
            RunnableConfig, {"configurable": {"thread_id": "thread-1"}}
        )
        interpreter = toolkit._get_or_create_interpreter(config)

        assert interpreter is mock_interpreter

    def test_uses_default_thread_id(self) -> None:
        """Test uses 'default' thread_id when not specified."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        with patch(
            "langchain_aws.tools.code_interpreter_toolkit.CodeInterpreter"
        ) as mock_class:
            mock_interpreter = MagicMock()
            mock_interpreter.start = MagicMock()
            mock_interpreter.session_id = "test-session"
            mock_class.return_value = mock_interpreter

            config: RunnableConfig = cast(RunnableConfig, {"configurable": {}})
            # This should raise KeyError since thread_id is required
            with pytest.raises(KeyError):
                toolkit._get_or_create_interpreter(config)


class TestGetThreadId:
    """Tests for _get_thread_id helper function."""

    def test_extracts_thread_id_from_config(self) -> None:
        """Test extracts thread_id from valid config."""
        from langchain_aws.tools.code_interpreter_toolkit import _get_thread_id

        config: RunnableConfig = cast(
            RunnableConfig, {"configurable": {"thread_id": "my-thread"}}
        )
        thread_id = _get_thread_id(config)

        assert thread_id == "my-thread"

    def test_returns_default_when_config_none(self) -> None:
        """Test returns 'default' when config is None."""
        from langchain_aws.tools.code_interpreter_toolkit import _get_thread_id

        thread_id = _get_thread_id(None)

        assert thread_id == "default"


class TestExtractOutputFromStream:
    """Tests for _extract_output_from_stream helper function."""

    def test_extracts_text_output(self) -> None:
        """Test extracts text content from stream."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            _extract_output_from_stream,
        )

        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "Hello World"},
                        ]
                    }
                }
            ]
        }

        output = _extract_output_from_stream(response)

        assert "Hello World" in output

    def test_extracts_file_resource(self) -> None:
        """Test extracts file resource from stream."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            _extract_output_from_stream,
        )

        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file://output.txt",
                                    "text": "file content",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        output = _extract_output_from_stream(response)

        assert "output.txt" in output
        assert "file content" in output

    def test_handles_binary_file(self) -> None:
        """Test handles binary file resource."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            _extract_output_from_stream,
        )

        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file://image.png",
                                    "blob": "base64data",
                                },
                            }
                        ]
                    }
                }
            ]
        }

        output = _extract_output_from_stream(response)

        assert "Binary File" in output
        assert "image.png" in output

    def test_handles_multiple_outputs(self) -> None:
        """Test handles multiple content items."""
        from langchain_aws.tools.code_interpreter_toolkit import (
            _extract_output_from_stream,
        )

        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ]
                    }
                }
            ]
        }

        output = _extract_output_from_stream(response)

        assert "Line 1" in output
        assert "Line 2" in output


class TestUploadFileValidation:
    """Tests for upload_file validation."""

    def test_rejects_absolute_path(self) -> None:
        """Test upload_file rejects absolute paths."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        with patch(
            "langchain_aws.tools.code_interpreter_toolkit.CodeInterpreter"
        ) as mock_class:
            mock_interpreter = MagicMock()
            mock_interpreter.start = MagicMock()
            mock_interpreter.session_id = "test-session"
            mock_class.return_value = mock_interpreter

            config: RunnableConfig = cast(
                RunnableConfig, {"configurable": {"thread_id": "test"}}
            )

            with pytest.raises(ValueError, match="Path must be relative"):
                toolkit._upload_file(
                    path="/absolute/path.txt",
                    content="test",
                    config=config,
                )


class TestInstallPackagesValidation:
    """Tests for install_packages validation."""

    def test_rejects_empty_packages_list(self) -> None:
        """Test install_packages rejects empty list."""
        from langchain_aws.tools.code_interpreter_toolkit import CodeInterpreterToolkit

        toolkit = CodeInterpreterToolkit()

        with patch(
            "langchain_aws.tools.code_interpreter_toolkit.CodeInterpreter"
        ) as mock_class:
            mock_interpreter = MagicMock()
            mock_interpreter.start = MagicMock()
            mock_interpreter.session_id = "test-session"
            mock_class.return_value = mock_interpreter

            config: RunnableConfig = cast(
                RunnableConfig, {"configurable": {"thread_id": "test"}}
            )

            with pytest.raises(ValueError, match="At least one package"):
                toolkit._install_packages(packages=[], config=config)
