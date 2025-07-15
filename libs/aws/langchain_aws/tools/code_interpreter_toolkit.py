import json
import logging
from typing import Dict, List, Optional, Tuple

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)


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

    The toolkit lazily initializes the code interpreter session on first use.
    It supports multiple threads by maintaining separate code interpreter sessions for each thread ID.

    Example:
        ```python

        import asyncio
        from langgraph.prebuilt import create_react_agent
        from langchain_aws.tools import create_code_interpreter_toolkit

        async def main():
            # Create and setup the code interpreter toolkit
            toolkit, code_tools = await create_code_interpreter_toolkit(region="us-west-2")

            # Create a ReAct agent using the code interpreter tools
            agent = create_react_agent(
                model="bedrock_converse:us.anthropic.claude-3-5-haiku-20241022-v1:0",
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

    def _get_or_create_interpreter(
        self, config: RunnableConfig
    ) -> CodeInterpreter:
        """
        Get or create a code interpreter for a specific config

        The config is expected to have a 'configurable' with 
        'thread_id', otherwise it creates a sesion with 'default'
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
        code_interpreter = CodeInterpreter(region=self.region)
        code_interpreter.start()
        logger.info(
            f"Started code interpreter with session_id:{code_interpreter.session_id} for thread:{thread_id}"  # noqa: E501
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

        # Create the basic tools for code execution - this doesn't initialize any code interpreter yet
        self.tools = self._create_tools()

        # Return the list of tools
        return self.tools

    def _create_tools(self) -> List[BaseTool]:
        """
        Create LangChain tools for code execution

        Returns:
            List of code execution tools
        """
        tools = []

        # Execute code tool
        execute_code_tool = StructuredTool.from_function(
            name="execute_code",
            func=self._execute_code,
        )
        tools.append(execute_code_tool)

        # Execute command tool
        execute_command_tool = StructuredTool.from_function(
            name="execute_command",
            func=self._execute_command,
        )
        tools.append(execute_command_tool)

        # Read files tool
        read_files_tool = StructuredTool.from_function(
            name="read_files",
            func=self._read_files,
        )
        tools.append(read_files_tool)

        # List files tool
        list_files_tool = StructuredTool.from_function(
            name="list_files",
            func=self._list_files,
        )
        tools.append(list_files_tool)

        # Remove files tool
        delete_files_tool = StructuredTool.from_function(
            name="delete_files",
            func=self._remove_files,
        )
        tools.append(delete_files_tool)

        # Write files tool
        write_files_tool = StructuredTool.from_function(
            name="write_files",
            func=self._write_files,
        )
        tools.append(write_files_tool)

        # Start command execution tool
        start_command_tool = StructuredTool.from_function(
            name="start_command_execution",
            func=self._start_command_execution,
        )
        tools.append(start_command_tool)

        # Get task status tool
        get_task_tool = StructuredTool.from_function(
            name="get_task",
            func=self._get_task,
        )
        tools.append(get_task_tool)

        # Stop task tool
        stop_task_tool = StructuredTool.from_function(
            name="stop_task",
            func=self._stop_task,
        )
        tools.append(stop_task_tool)

        return tools

    def _execute_code(
        self,
        code: str,
        config: RunnableConfig,
        language: str = "python",
        clearContext: bool = False,
    ) -> str:
        """
        Executes code in the AWS code interpreter environment

        Args:
            code:
                Code to execute
            language:
                Programming language, default is python
            clearContext:
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
            params={"code": code, "language": language, "clearContext": clearContext},
        )

        return _extract_output_from_stream(response)

    def _execute_command(
        self, command: str, config: RunnableConfig
    ) -> str:
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

    def _read_files(
        self, paths: List[str], config: RunnableConfig
    ) -> str:
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

    def _list_files(
        self, config: RunnableConfig, directory_path: str = ""
    ) -> str:
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

    def _remove_files(
        self, paths: List[str], config: RunnableConfig
    ) -> str:
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

    def _write_files(
        self, files: List[Dict[str, str]], config: RunnableConfig
    ) -> str:
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

    def _start_command_execution(
        self, command: str, config: RunnableConfig
    ) -> str:
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

    async def cleanup(self, thread_id: str = None) -> None:
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
) -> Tuple[CodeInterpreterToolkit, List[BaseTool]]:
    """
    Create and setup a CodeInterpreterToolkit

    Args:
        region: AWS region for code interpreter

    Returns:
        Tuple of (toolkit, tools)
    """
    toolkit = CodeInterpreterToolkit(region=region)
    # Create tools without immediately initializing the code interpreter
    tools = await toolkit._setup()
    # Code interpreter will be initialized lazily when first used
    return toolkit, tools


def _get_thread_id(config: Optional[RunnableConfig] = None):
    thread_id = "default"

    if config and isinstance(config, dict):
        thread_id = config["configurable"]["thread_id"]
    
    return thread_id

def _extract_output_from_stream(response):
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
                    if "text" in resource:
                        file_path = resource["uri"].replace("file://", "")
                        file_content = resource["text"]
                        output.append(f"==== File: {file_path} ====\n{file_content}\n")
                    else:
                        output.append(json.dumps(resource))

    return "\n".join(output)
