import logging
from typing import Any, Dict, List, Optional, Tuple

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecuteCodeInput(BaseModel):
    """Input schema for execute_code tool."""

    code: str = Field(
        description="Python/JavaScript/TypeScript code to execute. Can include "
        "imports, function definitions, data analysis, and visualizations. "
        "Variables and imports persist across calls within the same session."
    )
    language: str = Field(
        default="python",
        description="Programming language: 'python' (default), 'javascript', "
        "or 'typescript'",
    )
    clear_context: bool = Field(
        default=False,
        description="If True, clears all previous variable state before execution. "
        "Use this to start fresh or free memory.",
    )


class ExecuteCommandInput(BaseModel):
    """Input schema for execute_command tool."""

    command: str = Field(
        description="Shell command to execute "
        "(e.g., 'ls -la', 'pip list', 'cat file.txt'). "
        "Runs in a bash shell environment."
    )


class ReadFilesInput(BaseModel):
    """Input schema for read_files tool."""

    paths: List[str] = Field(
        description="List of file paths to read "
        "(e.g., ['data.csv', 'results/output.json'])"
    )


class WriteFilesInput(BaseModel):
    """Input schema for write_files tool."""

    files: List[Dict[str, str]] = Field(
        description="List of files to write. Each dict must have 'path' (relative path "
        "like 'data.csv' or 'scripts/analyze.py') and 'text' (file content). "
        "Cannot use absolute paths starting with '/'."
    )


class ListFilesInput(BaseModel):
    """Input schema for list_files tool."""

    directory_path: str = Field(
        default="",
        description="Directory path to list. Empty string or '.' for current "
        "directory.",
    )


class DeleteFilesInput(BaseModel):
    """Input schema for delete_files tool."""

    paths: List[str] = Field(description="List of file paths to delete")


class UploadFileInput(BaseModel):
    """Input schema for upload_file tool."""

    path: str = Field(
        description="Relative path where file should be saved "
        "(e.g., 'data.csv', 'scripts/analyze.py')"
    )
    content: str = Field(description="File content as string")
    description: str = Field(
        default="",
        description="Optional semantic description of the file contents to help "
        "understand the data structure "
        "(e.g., 'CSV with columns: date, revenue, product_id')",
    )


class InstallPackagesInput(BaseModel):
    """Input schema for install_packages tool."""

    packages: List[str] = Field(
        description="List of Python packages to install. Can include version "
        "specifiers (e.g., ['pandas>=2.0', 'numpy', 'scikit-learn==1.3.0'])"
    )
    upgrade: bool = Field(
        default=False, description="If True, upgrades packages if already installed"
    )


class CodeInterpreterToolkit:
    """Toolkit for working with AWS code interpreter environment.

    This toolkit provides a set of tools for working with a remote code interpreter environment:

    * execute_code - Run code in various languages (primarily Python)
    * execute_command - Run shell commands
    * read_files - Read content of files in the environment
    * list_files - List files in directories
    * delete_files - Remove files from the environment
    * write_files - Create or update files
    * start_command_execution - Start long-running commands asynchronously
    * get_task - Check status of async tasks
    * stop_task - Stop running tasks
    * upload_file - Upload files with semantic descriptions
    * install_packages - Install Python packages

    The toolkit lazily initializes the code interpreter session on first use.
    It supports multiple threads by maintaining separate code interpreter sessions for each thread ID.

    Example:
        ```python
        import asyncio
        from langchain.agents import create_agent
        from langchain_aws.tools import create_code_interpreter_toolkit

        async def main():
            # Create and setup the code interpreter toolkit
            toolkit, code_tools = await create_code_interpreter_toolkit(region="us-west-2")

            # Create a ReAct agent using the code interpreter tools
            agent = create_agent(
                "bedrock_converse:us.anthropic.claude-3-5-haiku-20241022-v1:0",
                tools=code_tools
            )

            # Create runnable config with thread ID
            config = {
                "configurable": {
                    "thread_id": "session123"
                }
            }

            # Invoke the agent with a specific task using thread ID
            result = await agent.ainvoke(
                "Create a simple Python function that calculates the factorial of a number.",
                config=config
            )

            # Cleanup when done
            await toolkit.cleanup()

            return result

        # Run the example
        asyncio.run(main())
        ```

    """  # noqa: E501

    def __init__(self, region: str = "us-west-2"):
        """
        Initialize the toolkit

        Args:
            region: AWS region for the code interpreter

        """
        self.region = region
        self._code_interpreters: Dict[str, CodeInterpreter] = {}
        self.tools: List[BaseTool] = []

    def get_tools(self) -> List[BaseTool]:
        """
        Get the list of code interpreter tools

        Returns:
            List of LangChain tools

        """
        return self.tools

    def get_tools_by_name(self) -> Dict[str, BaseTool]:
        """
        Get a dictionary of tools mapped by their names

        Returns:
            Dictionary of {tool_name: tool}

        """
        return {tool.name: tool for tool in self.tools}

    def _get_or_create_interpreter(self, config: RunnableConfig) -> CodeInterpreter:
        """
        Get or create a code interpreter for a specific config

        The config is expected to have a 'configurable' with
        'thread_id', otherwise it creates a session with 'default'
        thread ID.

        Args:
            config: Runnable config that may contain a thread_id

        Returns:
            CodeInterpreter instance for the specified thread

        """
        # Extract thread ID from config if available
        thread_id = _get_thread_id(config)

        if thread_id in self._code_interpreters:
            return self._code_interpreters[thread_id]

        # Create a new code interpreter for this thread
        # Pass integration_source for telemetry attribution
        code_interpreter = CodeInterpreter(
            region=self.region, integration_source="langchain"
        )
        code_interpreter.start()
        logger.info(
            f"Started code interpreter with session_id:{code_interpreter.session_id} "
            f"for thread:{thread_id}"
        )

        # Store the interpreter
        self._code_interpreters[thread_id] = code_interpreter
        return code_interpreter

    async def _setup(self) -> List[BaseTool]:
        """
        Setup and initialize code execution tools.

        Returns:
            List of LangChain tools for code execution

        """
        if self.tools:
            return self.tools

        # Create the basic tools for code execution - this doesn't initialize any
        # code interpreter yet
        self.tools = self._create_tools()

        # Return the list of tools
        return self.tools

    def _create_tools(self) -> List[BaseTool]:
        """
        Create LangChain tools for code execution

        Returns:
            List of code execution tools

        """
        tools: List[BaseTool] = []

        execute_code_tool = StructuredTool.from_function(
            name="execute_code",
            func=self._execute_code,
            args_schema=ExecuteCodeInput,
            description="""Execute code in a secure AWS sandbox environment.

Use this tool for:
- Data analysis and transformation (pandas, numpy)
- Mathematical calculations and statistics
- File processing (CSV, JSON, Excel, text files)
- Generating visualizations (matplotlib, plotly, seaborn)
- Running algorithms and data pipelines

Variables and imports persist across calls within the same session.
Use clear_context=True to reset state and free memory.""",
        )
        tools.append(execute_code_tool)

        execute_command_tool = StructuredTool.from_function(
            name="execute_command",
            func=self._execute_command,
            args_schema=ExecuteCommandInput,
            description="""Execute a shell command in the sandbox environment.

Use this tool for:
- Listing files and directories (ls, find)
- Checking installed packages (pip list)
- System information (python --version, which python)
- File operations (cat, head, tail, wc)
- Running scripts (python script.py, bash script.sh)""",
        )
        tools.append(execute_command_tool)

        read_files_tool = StructuredTool.from_function(
            name="read_files",
            func=self._read_files,
            args_schema=ReadFilesInput,
            description="""Read content of one or more files from the sandbox.

Use this tool to:
- Read data files before analysis
- Check contents of generated files
- Verify file modifications""",
        )
        tools.append(read_files_tool)

        list_files_tool = StructuredTool.from_function(
            name="list_files",
            func=self._list_files,
            args_schema=ListFilesInput,
            description="""List files and directories in the sandbox.

Use this tool to:
- See what files are available
- Check output directories
- Explore the sandbox structure""",
        )
        tools.append(list_files_tool)

        delete_files_tool = StructuredTool.from_function(
            name="delete_files",
            func=self._remove_files,
            args_schema=DeleteFilesInput,
            description="""Delete files from the sandbox environment.

Use this tool to:
- Clean up temporary files
- Remove old outputs
- Free disk space""",
        )
        tools.append(delete_files_tool)

        write_files_tool = StructuredTool.from_function(
            name="write_files",
            func=self._write_files,
            args_schema=WriteFilesInput,
            description="""Write/create files in the sandbox environment.

Use this tool to:
- Save analysis results
- Create data files for processing
- Write scripts or configuration files

Paths must be relative (e.g., 'output.csv', 'scripts/analyze.py').
Absolute paths starting with '/' are not allowed.""",
        )
        tools.append(write_files_tool)

        upload_file_tool = StructuredTool.from_function(
            name="upload_file",
            func=self._upload_file,
            args_schema=UploadFileInput,
            description="""Upload a file with optional semantic description.

This is a convenience tool for creating files with context.
The description helps track what the file contains.

Example:
- path: 'sales_data.csv'
- content: 'date,revenue\\n2024-01-01,1000'
- description: 'Daily sales with columns: date, revenue'""",
        )
        tools.append(upload_file_tool)

        install_packages_tool = StructuredTool.from_function(
            name="install_packages",
            func=self._install_packages,
            args_schema=InstallPackagesInput,
            description="""Install Python packages in the sandbox.

Use this tool before running code that requires packages not pre-installed.

Examples:
- ['pandas', 'matplotlib'] - Install multiple packages
- ['scikit-learn==1.3.0'] - Install specific version
- ['tensorflow'], upgrade=True - Upgrade if exists""",
        )
        tools.append(install_packages_tool)

        start_command_tool = StructuredTool.from_function(
            name="start_command_execution",
            func=self._start_command_execution,
            description="Start a long-running command asynchronously. "
            "Returns a task_id to check status.",
        )
        tools.append(start_command_tool)

        get_task_tool = StructuredTool.from_function(
            name="get_task",
            func=self._get_task,
            description="Check status of an async task by task_id.",
        )
        tools.append(get_task_tool)

        stop_task_tool = StructuredTool.from_function(
            name="stop_task",
            func=self._stop_task,
            description="Stop a running async task by task_id.",
        )
        tools.append(stop_task_tool)

        return tools

    def _execute_code(
        self,
        code: str,
        config: RunnableConfig,
        language: str = "python",
        clear_context: bool = False,
    ) -> str:
        """
        Executes code in the AWS code interpreter environment

        Args:
            code:
                Code to execute
            language:
                Programming language, default is python
            clear_context:
                Whether to clear execution context, default is False
            config:
                Runnable config that may contain a thread_id

        Returns:
            String containing execution results

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="executeCode",
            params={"code": code, "language": language, "clearContext": clear_context},
        )

        return _extract_output_from_stream(response)

    def _execute_command(self, command: str, config: RunnableConfig) -> str:
        """
        Execute a command synchronously

        Args:
            command: Command to execute
            config: Runnable config that may contain a thread_id

        Returns:
            String containing execution results

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="executeCommand", params={"command": command}
        )

        return _extract_output_from_stream(response)

    def _read_files(self, paths: List[str], config: RunnableConfig) -> str:
        """
        Read content of files

        Args:
            paths: List of file paths to read
            config: Runnable config that may contain a thread_id

        Returns:
            String containing file contents

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(method="readFiles", params={"paths": paths})

        return _extract_output_from_stream(response)

    def _list_files(self, config: RunnableConfig, directory_path: str = "") -> str:
        """
        List files in a directory

        Args:
            directory_path: Path to the directory to list, defaults to current directory
            config: Runnable config that may contain a thread_id

        Returns:
            String containing list of files

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="listFiles", params={"directoryPath": directory_path}
        )

        return _extract_output_from_stream(response)

    def _remove_files(self, paths: List[str], config: RunnableConfig) -> str:
        """
        Remove files from the system

        Args:
            paths: List of file paths to remove
            config: Runnable config that may contain a thread_id

        Returns:
            String containing removal result

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="removeFiles", params={"paths": paths}
        )

        return _extract_output_from_stream(response)

    def _write_files(self, files: List[Dict[str, str]], config: RunnableConfig) -> str:
        """
        Writes file content to the specified path in code env

        Is limited to writing files in the current working dir
        of the code interpreter environment. Absolute paths such
        as beginning with / are not allowed, only paths relative
        to current working dir, e.g., file.txt or dir/file.txt are
        valid paths.

        Args:
            files: List of dictionaries with path and text fields
            config: Runnable config that may contain a thread_id

        Returns:
            String containing write results

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="writeFiles", params={"content": files}
        )

        return _extract_output_from_stream(response)

    def _upload_file(
        self,
        path: str,
        content: str,
        config: RunnableConfig,
        description: str = "",
    ) -> str:
        """
        Upload a file with optional semantic description.

        Args:
            path: Relative path where file should be saved
            content: File content as string
            description: Optional description of file contents
            config: Runnable config that may contain a thread_id

        Returns:
            String confirming upload

        """
        if path.startswith("/"):
            raise ValueError(
                f"Path must be relative, not absolute. Got: {path}. "
                "Use paths like 'data.csv' or 'scripts/analyze.py'."
            )

        code_interpreter = self._get_or_create_interpreter(config=config)
        response = code_interpreter.upload_file(path, content, description)
        return _extract_output_from_stream(response)

    def _install_packages(
        self,
        packages: List[str],
        config: RunnableConfig,
        upgrade: bool = False,
    ) -> str:
        """
        Install Python packages in the code interpreter environment.

        Args:
            packages: List of package names to install
            upgrade: If True, upgrades existing packages
            config: Runnable config that may contain a thread_id

        Returns:
            String containing installation results

        """
        if not packages:
            raise ValueError("At least one package name must be provided")

        code_interpreter = self._get_or_create_interpreter(config=config)
        response = code_interpreter.install_packages(packages, upgrade=upgrade)
        return _extract_output_from_stream(response)

    def _start_command_execution(self, command: str, config: RunnableConfig) -> str:
        """
        Start a long-running command asynchronously

        Args:
            command: Command to execute
            config: Runnable config that may contain a thread_id

        Returns:
            String containing task ID and status

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="startCommandExecution", params={"command": command}
        )

        return _extract_output_from_stream(response)

    def _get_task(self, task_id: str, config: RunnableConfig) -> str:
        """
        Get status of an async task

        Args:
            task_id: ID of the task to check
            config: Runnable config that may contain a thread_id

        Returns:
            String containing task status

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(method="getTask", params={"taskId": task_id})

        return _extract_output_from_stream(response)

    def _stop_task(self, task_id: str, config: RunnableConfig) -> str:
        """
        Stop a running task

        Args:
            task_id: ID of the task to stop
            config: Runnable config that may contain a thread_id

        Returns:
            String containing stop result

        """
        # Get or create code interpreter for this thread using the config
        code_interpreter = self._get_or_create_interpreter(config=config)

        response = code_interpreter.invoke(
            method="stopTask", params={"taskId": task_id}
        )

        return _extract_output_from_stream(response)

    async def cleanup(self, thread_id: Optional[str] = None) -> None:
        """Clean up resources

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all sessions.

        """
        if thread_id:
            # Clean up a specific thread's session
            if thread_id in self._code_interpreters:
                try:
                    self._code_interpreters[thread_id].stop()
                    del self._code_interpreters[thread_id]
                    logger.info(
                        f"Code interpreter session for thread {thread_id} cleaned up"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error stopping code interpreter for thread {thread_id}: {e}"
                    )
        else:
            # Clean up all sessions
            thread_ids = list(self._code_interpreters.keys())
            for tid in thread_ids:
                try:
                    self._code_interpreters[tid].stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping code interpreter for thread {tid}: {e}"
                    )

            self._code_interpreters = {}
            logger.info("All code interpreter sessions cleaned up")


async def create_code_interpreter_toolkit(
    region: str = "us-west-2",
) -> Tuple["CodeInterpreterToolkit", List[BaseTool]]:
    """Create and setup a CodeInterpreterToolkit.

    Args:
        region: AWS region for code interpreter

    Returns:
        Tuple of (toolkit, tools)

    Example:
        >>> toolkit, tools = await create_code_interpreter_toolkit()
        >>> # Use tools with an agent
        >>> agent = create_react_agent(model, tools=tools)
        >>> await toolkit.cleanup()  # When done
    """
    toolkit = CodeInterpreterToolkit(region=region)
    tools = await toolkit._setup()
    return toolkit, tools


def _get_thread_id(config: Optional[RunnableConfig] = None) -> str:
    thread_id = "default"

    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {})["thread_id"]

    return thread_id


def _extract_output_from_stream(response: Any) -> str:
    """
    Extract output from code interpreter response stream

    Args:
        response: Response from code interpreter execution

    Returns:
        Extracted output as string

    """
    output = []
    for event in response["stream"]:
        if "result" in event:
            result = event["result"]
            for content_item in result["content"]:
                if content_item["type"] == "text":
                    output.append(content_item["text"])
                if content_item["type"] == "resource":
                    resource = content_item["resource"]
                    file_path = resource.get("uri", "").replace("file://", "")
                    if "text" in resource:
                        file_content = resource["text"]
                        output.append(f"==== File: {file_path} ====\n{file_content}\n")
                    elif "blob" in resource:
                        # Binary file (images, etc.) - just note it was created
                        output.append(f"==== Binary File: {file_path} ====\n")
                    else:
                        output.append(f"==== File: {file_path} ====\n")

    return "\n".join(output)
