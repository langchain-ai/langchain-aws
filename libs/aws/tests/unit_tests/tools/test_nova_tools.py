from langchain_aws.tools.nova_tools import (
    NovaCodeInterpreterTool,
    NovaGroundingTool,
    NovaSystemTool,
)


class TestNovaSystemTool:
    """Tests for NovaSystemTool base class."""

    def test_initialization(self) -> None:
        """Test NovaSystemTool initialization."""
        tool = NovaSystemTool("test_tool")
        assert tool.name == "test_tool"
        assert tool.type == "system_tool"

    def test_to_bedrock_format(self) -> None:
        """Test to_bedrock_format returns correct structure."""
        tool = NovaSystemTool("test_tool")
        result = tool.to_bedrock_format()

        assert isinstance(result, dict)
        assert "systemTool" in result
        assert "name" in result["systemTool"]
        assert result["systemTool"]["name"] == "test_tool"

    def test_to_bedrock_format_structure(self) -> None:
        """Test to_bedrock_format returns exact expected structure."""
        tool = NovaSystemTool("my_tool")
        expected = {"systemTool": {"name": "my_tool"}}
        assert tool.to_bedrock_format() == expected


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

    def test_to_bedrock_format(self) -> None:
        """Test NovaGroundingTool formats correctly."""
        tool = NovaGroundingTool()
        result = tool.to_bedrock_format()

        expected = {"systemTool": {"name": "nova_grounding"}}
        assert result == expected


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

    def test_to_bedrock_format(self) -> None:
        """Test NovaCodeInterpreterTool formats correctly."""
        tool = NovaCodeInterpreterTool()
        result = tool.to_bedrock_format()

        expected = {"systemTool": {"name": "nova_code_interpreter"}}
        assert result == expected
