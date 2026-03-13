from langchain_core.tools import BaseTool

from langchain_aws.tools.nova_tools import (
    NovaCodeInterpreterTool,
    NovaGroundingTool,
    NovaSystemTool,
)


class TestNovaSystemTool:
    """Tests for NovaSystemTool base class."""

    def test_initialization(self) -> None:
        """Test NovaSystemTool initialization."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        assert tool.name == "test_tool"
        assert tool.type == "system_tool"

    def test_to_bedrock_format(self) -> None:
        """Test to_bedrock_format returns correct structure."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        result = tool.to_bedrock_format()

        assert isinstance(result, dict)
        assert "systemTool" in result
        assert "name" in result["systemTool"]
        assert result["systemTool"]["name"] == "test_tool"

    def test_to_bedrock_format_structure(self) -> None:
        """Test to_bedrock_format returns exact expected structure."""
        tool = NovaSystemTool(name="my_tool", description="My tool")
        expected = {"systemTool": {"name": "my_tool"}}
        assert tool.to_bedrock_format() == expected

    def test_is_base_tool(self) -> None:
        """Test that NovaSystemTool is a BaseTool instance."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        assert isinstance(tool, BaseTool)

    def test_run_returns_message(self) -> None:
        """Test that _run returns a server-side execution message."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        result = tool._run()
        assert "server-side" in result
        assert "test_tool" in result


class TestNovaGroundingTool:
    """Tests for NovaGroundingTool."""

    def test_initialization(self) -> None:
        """Test NovaGroundingTool initialization."""
        tool = NovaGroundingTool()
        assert tool.name == "nova_grounding"
        assert tool.type == "system_tool"

    def test_inherits_from_nova_system_tool(self) -> None:
        """Test that NovaGroundingTool inherits from NovaSystemTool."""
        tool = NovaGroundingTool()
        assert isinstance(tool, NovaSystemTool)

    def test_is_base_tool(self) -> None:
        """Test that NovaGroundingTool is a BaseTool (create_agent compat)."""
        tool = NovaGroundingTool()
        assert isinstance(tool, BaseTool)

    def test_to_bedrock_format(self) -> None:
        """Test NovaGroundingTool formats correctly."""
        tool = NovaGroundingTool()
        result = tool.to_bedrock_format()

        expected = {"systemTool": {"name": "nova_grounding"}}
        assert result == expected

    def test_has_description(self) -> None:
        """Test that NovaGroundingTool has a non-empty description."""
        tool = NovaGroundingTool()
        assert tool.description
        assert "grounding" in tool.description.lower()


class TestNovaCodeInterpreterTool:
    """Tests for NovaCodeInterpreterTool."""

    def test_initialization(self) -> None:
        """Test NovaCodeInterpreterTool initialization."""
        tool = NovaCodeInterpreterTool()
        assert tool.name == "nova_code_interpreter"
        assert tool.type == "system_tool"

    def test_inherits_from_nova_system_tool(self) -> None:
        """Test that NovaCodeInterpreterTool inherits from NovaSystemTool."""
        tool = NovaCodeInterpreterTool()
        assert isinstance(tool, NovaSystemTool)

    def test_is_base_tool(self) -> None:
        """Test that NovaCodeInterpreterTool is a BaseTool (create_agent compat)."""
        tool = NovaCodeInterpreterTool()
        assert isinstance(tool, BaseTool)

    def test_to_bedrock_format(self) -> None:
        """Test NovaCodeInterpreterTool formats correctly."""
        tool = NovaCodeInterpreterTool()
        result = tool.to_bedrock_format()

        expected = {"systemTool": {"name": "nova_code_interpreter"}}
        assert result == expected

    def test_has_description(self) -> None:
        """Test that NovaCodeInterpreterTool has a non-empty description."""
        tool = NovaCodeInterpreterTool()
        assert tool.description
        assert "code" in tool.description.lower()
