from langchain_core.tools import BaseTool

from langchain_aws.tools.nova_tools import (
    NovaCodeInterpreterTool,
    NovaGroundingTool,
    NovaSystemTool,
)


class TestNovaSystemTool:
    """Tests for NovaSystemTool base class."""

    def test_initialization(self) -> None:
        """Test NovaSystemTool initialization with keyword args."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        assert tool.name == "test_tool"
        assert tool.type == "system_tool"

    def test_is_base_tool(self) -> None:
        """Test that NovaSystemTool is a BaseTool instance."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        assert isinstance(tool, BaseTool)

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
        tool = NovaSystemTool(name="my_tool", description="A tool")
        expected = {"systemTool": {"name": "my_tool"}}
        assert tool.to_bedrock_format() == expected

    def test_run_returns_server_side_message(self) -> None:
        """Test that _run returns an informative server-side message."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        result = tool._run()
        assert "test_tool" in result
        assert "server-side" in result

    def test_empty_args_schema(self) -> None:
        """Test that system tools have an empty args schema."""
        tool = NovaSystemTool(name="test_tool", description="A test tool")
        schema = tool.args_schema.model_json_schema()
        # Should have no required properties — empty input
        assert schema.get("required") is None or schema.get("required") == []


class TestNovaGroundingTool:
    """Tests for NovaGroundingTool."""

    def test_initialization(self) -> None:
        """Test NovaGroundingTool initialization with defaults."""
        tool = NovaGroundingTool()
        assert tool.name == "nova_grounding"
        assert tool.type == "system_tool"

    def test_has_description(self) -> None:
        """Test NovaGroundingTool has a meaningful description."""
        tool = NovaGroundingTool()
        assert tool.description
        assert len(tool.description) > 0

    def test_is_base_tool(self) -> None:
        """Test that NovaGroundingTool is a BaseTool instance."""
        tool = NovaGroundingTool()
        assert isinstance(tool, BaseTool)

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

    def test_no_args_constructor(self) -> None:
        """Test that NovaGroundingTool requires no constructor arguments."""
        tool = NovaGroundingTool()
        assert tool.name == "nova_grounding"


class TestNovaCodeInterpreterTool:
    """Tests for NovaCodeInterpreterTool."""

    def test_initialization(self) -> None:
        """Test NovaCodeInterpreterTool initialization with defaults."""
        tool = NovaCodeInterpreterTool()
        assert tool.name == "nova_code_interpreter"
        assert tool.type == "system_tool"

    def test_has_description(self) -> None:
        """Test NovaCodeInterpreterTool has a meaningful description."""
        tool = NovaCodeInterpreterTool()
        assert tool.description
        assert len(tool.description) > 0

    def test_is_base_tool(self) -> None:
        """Test that NovaCodeInterpreterTool is a BaseTool instance."""
        tool = NovaCodeInterpreterTool()
        assert isinstance(tool, BaseTool)

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

    def test_no_args_constructor(self) -> None:
        """Test that NovaCodeInterpreterTool requires no constructor arguments."""
        tool = NovaCodeInterpreterTool()
        assert tool.name == "nova_code_interpreter"


class TestToolNodeCompatibility:
    """Tests verifying system tools work with LangGraph's ToolNode."""

    def test_tool_node_accepts_nova_grounding(self) -> None:
        """Test that ToolNode accepts NovaGroundingTool without error."""
        from langgraph.prebuilt import ToolNode

        tool = NovaGroundingTool()
        # Should not raise — this was the original crash in issue #921
        node = ToolNode([tool])
        assert "nova_grounding" in node.tools_by_name

    def test_tool_node_accepts_nova_code_interpreter(self) -> None:
        """Test that ToolNode accepts NovaCodeInterpreterTool without error."""
        from langgraph.prebuilt import ToolNode

        tool = NovaCodeInterpreterTool()
        node = ToolNode([tool])
        assert "nova_code_interpreter" in node.tools_by_name

    def test_tool_node_accepts_mixed_tools(self) -> None:
        """Test ToolNode accepts system tools alongside regular tools."""
        from langchain_core.tools import tool as tool_decorator
        from langgraph.prebuilt import ToolNode

        @tool_decorator
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"

        node = ToolNode([get_weather, NovaGroundingTool()])
        assert "get_weather" in node.tools_by_name
        assert "nova_grounding" in node.tools_by_name
