"""Unit tests for browser toolkit and tools."""

import pytest

# Skip all tests in this module if optional dependencies are not installed
pytest.importorskip("bedrock_agentcore", reason="Requires langchain-aws[tools]")
pytest.importorskip("playwright", reason="Requires langchain-aws[tools]")

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool


class TestBrowserToolkit:
    """Tests for BrowserToolkit class."""

    def test_create_browser_toolkit(self) -> None:
        """Test create_browser_toolkit factory function."""
        from langchain_aws.tools.browser_toolkit import create_browser_toolkit

        toolkit, tools = create_browser_toolkit(region="us-west-2")

        assert toolkit is not None
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_toolkit_initializes_with_region(self) -> None:
        """Test toolkit initializes with specified region."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit(region="us-east-1")

        assert toolkit.region == "us-east-1"
        assert toolkit.session_manager is not None
        assert toolkit.session_manager.region == "us-east-1"

    def test_toolkit_default_region(self) -> None:
        """Test toolkit uses default region."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit()

        assert toolkit.region == "us-west-2"

    def test_get_tools_returns_list(self) -> None:
        """Test get_tools returns list of tools."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit()
        tools = toolkit.get_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_by_name_returns_dict(self) -> None:
        """Test get_tools_by_name returns dictionary."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit()
        tools_dict = toolkit.get_tools_by_name()

        assert isinstance(tools_dict, dict)
        assert len(tools_dict) > 0
        assert all(isinstance(name, str) for name in tools_dict.keys())
        assert all(isinstance(tool, BaseTool) for tool in tools_dict.values())

    def test_expected_tools_present(self) -> None:
        """Test all expected browser tools are present."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit()
        tools_dict = toolkit.get_tools_by_name()

        # Updated to match actual tool.name values from browser_tools.py
        expected_tools = [
            "navigate_browser",  # ThreadAwareNavigateTool.name
            "click_element",  # ThreadAwareClickTool.name (was "click")
            "navigate_back",  # ThreadAwareNavigateBackTool.name
            "extract_text",  # ThreadAwareExtractTextTool.name
            "extract_hyperlinks",  # ThreadAwareExtractHyperlinksTool.name
            "get_elements",  # ThreadAwareGetElementsTool.name
            "current_webpage",  # ThreadAwareCurrentWebPageTool.name
            "type_text",  # ThreadAwareTypeTool.name
            "take_screenshot",  # ThreadAwareScreenshotTool.name (was "screenshot")
            "scroll_page",  # ThreadAwareScrollTool.name
            "wait_for_element",  # ThreadAwareWaitForElementTool.name
        ]

        for tool_name in expected_tools:
            assert tool_name in tools_dict, f"Missing tool: {tool_name}"

    @pytest.mark.asyncio
    async def test_cleanup_calls_session_manager(self) -> None:
        """Test cleanup calls session manager's close_all_browsers."""
        from langchain_aws.tools.browser_toolkit import BrowserToolkit

        toolkit = BrowserToolkit()

        with patch.object(
            toolkit.session_manager, "close_all_browsers", new_callable=AsyncMock
        ) as mock_close:
            await toolkit.cleanup()
            mock_close.assert_called_once()


class TestBrowserSessionManager:
    """Tests for BrowserSessionManager class."""

    def test_initialization(self) -> None:
        """Test session manager initializes correctly."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        assert manager.region == "us-west-2"
        assert manager._async_sessions == {}
        assert manager._sync_sessions == {}

    @pytest.mark.asyncio
    async def test_get_async_browser_creates_session(self) -> None:
        """Test get_async_browser creates a new session."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        # Patch at the correct locations:
        # - BrowserClient is imported at module level
        # - async_playwright is imported inside the method from playwright.async_api
        with (
            patch(
                "langchain_aws.tools.browser_session_manager.BrowserClient"
            ) as mock_client_class,
            patch("playwright.async_api.async_playwright") as mock_playwright,
        ):
            # Setup BrowserClient mock
            mock_client = MagicMock()
            mock_client.start = MagicMock()
            mock_client.generate_ws_headers = MagicMock(
                return_value=("ws://test", {"header": "value"})
            )
            mock_client_class.return_value = mock_client

            # Setup playwright mock
            mock_browser = AsyncMock()
            mock_pw_instance = AsyncMock()
            mock_pw_instance.chromium.connect_over_cdp = AsyncMock(
                return_value=mock_browser
            )

            mock_pw_context = MagicMock()
            mock_pw_context.start = AsyncMock(return_value=mock_pw_instance)
            mock_playwright.return_value = mock_pw_context

            browser = await manager.get_async_browser("thread-1")

            assert browser is not None
            mock_client.start.assert_called_once()
            mock_client.generate_ws_headers.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_async_browser_reuses_session(self) -> None:
        """Test get_async_browser reuses existing session after release."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        # Manually add a session that's not in use
        mock_client = MagicMock()
        mock_browser = AsyncMock()
        manager._async_sessions["thread-1"] = (mock_client, mock_browser, False)

        browser = await manager.get_async_browser("thread-1")

        assert browser is mock_browser
        # Verify session is now marked as in use
        assert manager._async_sessions["thread-1"][2] is True

    @pytest.mark.asyncio
    async def test_get_async_browser_raises_if_in_use(self) -> None:
        """Test get_async_browser raises error if session is in use."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        # Add a session that's in use
        mock_client = MagicMock()
        mock_browser = AsyncMock()
        manager._async_sessions["thread-1"] = (mock_client, mock_browser, True)

        with pytest.raises(RuntimeError, match="already in use"):
            await manager.get_async_browser("thread-1")

    @pytest.mark.asyncio
    async def test_release_async_browser(self) -> None:
        """Test release_async_browser marks session as available."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        # Add a session that's in use
        mock_client = MagicMock()
        mock_browser = AsyncMock()
        manager._async_sessions["thread-1"] = (mock_client, mock_browser, True)

        await manager.release_async_browser("thread-1")

        assert manager._async_sessions["thread-1"][2] is False

    @pytest.mark.asyncio
    async def test_release_async_browser_raises_if_not_found(self) -> None:
        """Test release_async_browser raises error if session not found."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        with pytest.raises(KeyError, match="No async browser session found"):
            await manager.release_async_browser("nonexistent-thread")

    @pytest.mark.asyncio
    async def test_close_async_browser(self) -> None:
        """Test close_async_browser cleans up resources."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        mock_client = MagicMock()
        mock_client.stop = MagicMock()
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        manager._async_sessions["thread-1"] = (mock_client, mock_browser, True)

        await manager.close_async_browser("thread-1")

        mock_browser.close.assert_called_once()
        mock_client.stop.assert_called_once()
        assert "thread-1" not in manager._async_sessions

    @pytest.mark.asyncio
    async def test_close_all_browsers(self) -> None:
        """Test close_all_browsers cleans up all sessions."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager

        manager = BrowserSessionManager(region="us-west-2")

        # Add multiple async sessions
        for i in range(3):
            mock_client = MagicMock()
            mock_client.stop = MagicMock()
            mock_browser = AsyncMock()
            mock_browser.close = AsyncMock()
            manager._async_sessions[f"thread-{i}"] = (mock_client, mock_browser, False)

        await manager.close_all_browsers()

        assert len(manager._async_sessions) == 0
        assert len(manager._sync_sessions) == 0


class TestBrowserToolInputSchemas:
    """Tests for browser tool input schemas."""

    def test_navigate_tool_input(self) -> None:
        """Test NavigateToolInput schema."""
        from langchain_aws.tools.browser_tools import NavigateToolInput

        input_data = NavigateToolInput(url="https://example.com")
        assert input_data.url == "https://example.com"

    def test_click_tool_input(self) -> None:
        """Test ClickToolInput schema."""
        from langchain_aws.tools.browser_tools import ClickToolInput

        input_data = ClickToolInput(selector="#button")
        assert input_data.selector == "#button"

    def test_type_text_input(self) -> None:
        """Test TypeTextInput schema."""
        from langchain_aws.tools.browser_tools import TypeTextInput

        input_data = TypeTextInput(selector="input[name='search']", text="test query")
        assert input_data.selector == "input[name='search']"
        assert input_data.text == "test query"

    def test_screenshot_input_defaults(self) -> None:
        """Test ScreenshotInput default values."""
        from langchain_aws.tools.browser_tools import ScreenshotInput

        input_data = ScreenshotInput()
        assert input_data.capture_type == "viewport"

    def test_scroll_input_defaults(self) -> None:
        """Test ScrollInput default values."""
        from langchain_aws.tools.browser_tools import ScrollInput

        input_data = ScrollInput()
        assert input_data.direction == "down"
        assert input_data.amount == 500

    def test_wait_for_element_input_defaults(self) -> None:
        """Test WaitForElementInput default values."""
        from langchain_aws.tools.browser_tools import WaitForElementInput

        input_data = WaitForElementInput(selector=".element")
        assert input_data.selector == ".element"
        assert input_data.timeout == 30000
        assert input_data.state == "visible"


class TestThreadAwareBaseTool:
    """Tests for ThreadAwareBaseTool base class."""

    def test_get_thread_id_from_config(self) -> None:
        """Test extracting thread_id from config."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager
        from langchain_aws.tools.browser_tools import ThreadAwareNavigateTool

        manager = BrowserSessionManager()
        tool = ThreadAwareNavigateTool(_session_manager=manager)

        config: RunnableConfig = cast(
            RunnableConfig, {"configurable": {"thread_id": "my-thread"}}
        )
        thread_id = tool.get_thread_id(config)

        assert thread_id == "my-thread"

    def test_get_thread_id_default(self) -> None:
        """Test default thread_id when config is None."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager
        from langchain_aws.tools.browser_tools import ThreadAwareNavigateTool

        manager = BrowserSessionManager()
        tool = ThreadAwareNavigateTool(_session_manager=manager)

        thread_id = tool.get_thread_id(None)

        assert thread_id == "default"


class TestNavigateTool:
    """Tests for NavigateTool."""

    def test_tool_name_and_description(self) -> None:
        """Test tool has correct name and description."""
        from langchain_aws.tools.browser_session_manager import BrowserSessionManager
        from langchain_aws.tools.browser_tools import ThreadAwareNavigateTool

        manager = BrowserSessionManager()
        tool = ThreadAwareNavigateTool(_session_manager=manager)

        assert tool.name == "navigate_browser"
        assert "Navigate" in tool.description or "navigate" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_navigate_validates_url_scheme(self) -> None:
        """Test navigate rejects invalid URL schemes."""
        from langchain_core.tools import ToolException

        from langchain_aws.tools.browser_session_manager import BrowserSessionManager
        from langchain_aws.tools.browser_tools import ThreadAwareNavigateTool

        manager = BrowserSessionManager()
        tool = ThreadAwareNavigateTool(_session_manager=manager)

        with patch.object(
            manager, "get_async_browser", new_callable=AsyncMock
        ) as mock_get:
            mock_page = AsyncMock()
            mock_browser = AsyncMock()
            mock_browser.contexts = [MagicMock(pages=[mock_page])]
            mock_get.return_value = mock_browser

            with patch.object(manager, "release_async_browser", new_callable=AsyncMock):
                config: RunnableConfig = cast(
                    RunnableConfig, {"configurable": {"thread_id": "test"}}
                )
                with pytest.raises(ToolException, match="scheme must be"):
                    await tool._arun(
                        url="ftp://invalid.com",
                        config=config,
                    )
