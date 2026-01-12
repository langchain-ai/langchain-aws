"""Thread-aware browser tools that work with the browser session manager."""

import base64
import json
import logging
from typing import Any, Dict, Literal, Optional, Type
from urllib.parse import urlparse

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .browser_session_manager import BrowserSessionManager
from .utils import aget_current_page, get_current_page

logger = logging.getLogger(__name__)


def _get_scroll_deltas(direction: str, amount: int) -> tuple[int, int]:
    """Convert direction and amount to delta_x, delta_y."""
    direction = direction.lower()
    if direction == "down":
        return 0, amount
    elif direction == "up":
        return 0, -amount
    elif direction == "right":
        return amount, 0
    elif direction == "left":
        return -amount, 0
    else:
        raise ValueError(
            f"Invalid direction: {direction}. Use 'up', 'down', 'left', or 'right'."
        )


class NavigateToolInput(BaseModel):
    """Input for NavigateTool."""

    url: str = Field(description="URL to navigate to")


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    selector: str = Field(description="CSS selector for the element to click on")


class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""

    selector: str = Field(description="CSS selector for elements to get")


class ExtractTextToolInput(BaseModel):
    """Input for ExtractTextTool."""

    pass


class ExtractHyperlinksToolInput(BaseModel):
    """Input for ExtractHyperlinksTool."""

    pass


class NavigateBackToolInput(BaseModel):
    """Input for NavigateBackTool."""

    pass


class CurrentWebPageToolInput(BaseModel):
    """Input for CurrentWebPageTool."""

    pass


class TypeTextInput(BaseModel):
    """Input for TypeTextTool."""

    selector: str = Field(
        description="CSS selector for the input field to type into "
        "(e.g., '#search', 'input[name=\"email\"]', '.search-box')"
    )
    text: str = Field(description="Text to type into the input field")


class ScreenshotInput(BaseModel):
    """Input for ScreenshotTool."""

    capture_type: Literal["viewport", "full_page"] = Field(
        default="viewport",
        description=(
            "Type of screenshot: 'viewport' (visible area, default) "
            "or 'full_page' (entire scrollable page)"
        ),
    )


class ScrollInput(BaseModel):
    """Input for ScrollTool."""

    direction: Literal["up", "down", "left", "right"] = Field(
        default="down",
        description="Scroll direction",
    )
    amount: int = Field(default=500, description="Number of pixels to scroll", gt=0)


class WaitForElementInput(BaseModel):
    """Input for WaitForElementTool."""

    selector: str = Field(description="CSS selector for the element to wait for")
    timeout: int = Field(
        default=30000,
        description="Maximum time to wait in milliseconds "
        "(default: 30000ms = 30 seconds)",
        gt=0,
        le=300000,
    )
    state: Literal["attached", "detached", "visible", "hidden"] = Field(
        default="visible",
        description="State to wait for",
    )


class ThreadAwareBaseTool(BaseTool):
    """Base class for thread-aware browser tools."""

    _session_manager: BrowserSessionManager

    def __init__(self, _session_manager: BrowserSessionManager, **kwargs: Any):
        """Initialize with a session manager."""
        super().__init__(**kwargs)
        self._session_manager = _session_manager

    def get_thread_id(self, config: Optional[RunnableConfig] = None) -> str:
        """Extract thread ID from config."""
        thread_id = "default"

        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {})["thread_id"]

        return thread_id

    async def get_async_page(self, thread_id: str) -> Any:
        """Get or create a page for the specified thread."""
        browser = await self._session_manager.get_async_browser(thread_id)
        page = await aget_current_page(browser)
        return page

    def get_sync_page(self, thread_id: str) -> Any:
        """Get or create a page for the specified thread."""
        browser = self._session_manager.get_sync_browser(thread_id)
        page = get_current_page(browser)
        return page

    async def release_async_browser(self, thread_id: str) -> None:
        """Release the async browser session after use."""
        try:
            await self._session_manager.release_async_browser(thread_id)
            logger.debug(f"Released async browser for thread {thread_id}")
        except Exception as e:
            logger.warning(f"Error releasing async browser for thread {thread_id}: {e}")

    def release_sync_browser(self, thread_id: str) -> None:
        """Release the sync browser session after use."""
        try:
            self._session_manager.release_sync_browser(thread_id)
            logger.debug(f"Released sync browser for thread {thread_id}")
        except Exception as e:
            logger.warning(f"Error releasing sync browser for thread {thread_id}: {e}")


class ThreadAwareNavigateTool(ThreadAwareBaseTool):
    """Tool for navigating a browser to a URL with thread support."""

    name: str = "navigate_browser"
    description: str = "Navigate a browser to the specified URL"
    args_schema: Type[BaseModel] = NavigateToolInput

    def _run(
        self,
        url: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            logger.info(f"Passed config is {config}")

            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get page for this thread
            page = self.get_sync_page(thread_id)

            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError("URL scheme must be 'http' or 'https'")

            # Navigate to URL
            response = page.goto(url)
            status = response.status if response else "unknown"

            self.release_sync_browser(thread_id)

            return f"Navigating to {url} returned status code {status}"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error navigating to {url}: {str(e)}")

    async def _arun(
        self,
        url: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get page for this thread
            page = await self.get_async_page(thread_id)

            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError("URL scheme must be 'http' or 'https'")

            # Navigate to URL
            response = await page.goto(url)
            status = response.status if response else "unknown"

            await self.release_async_browser(thread_id)

            return f"Navigating to {url} returned status code {status}"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error navigating to {url}: {str(e)}")


class ThreadAwareClickTool(ThreadAwareBaseTool):
    """Tool for clicking on an element with the given CSS selector."""

    name: str = "click_element"
    description: str = "Click on an element with the given CSS selector"
    args_schema: Type[BaseModel] = ClickToolInput

    visible_only: bool = True
    """Whether to consider only visible elements."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when clicking on elements."""
    playwright_timeout: float = 1_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _run(
        self,
        selector: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Click on the element
            selector_effective = self._selector_effective(selector=selector)
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

            try:
                page.click(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
            except PlaywrightTimeoutError:
                self.release_sync_browser(thread_id)
                return f"Unable to click on element '{selector}'"
            except Exception as click_error:
                self.release_sync_browser(thread_id)
                return f"Unable to click on element '{selector}': {str(click_error)}"

            self.release_sync_browser(thread_id)

            return f"Clicked element '{selector}'"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error clicking on element: {str(e)}")

    async def _arun(
        self,
        selector: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Click on the element
            selector_effective = self._selector_effective(selector=selector)
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError

            try:
                await page.click(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
            except PlaywrightTimeoutError:
                await self.release_async_browser(thread_id)
                return f"Unable to click on element '{selector}'"
            except Exception as click_error:
                await self.release_async_browser(thread_id)
                return f"Unable to click on element '{selector}': {str(click_error)}"

            await self.release_async_browser(thread_id)

            return f"Clicked element '{selector}'"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error clicking on element: {str(e)}")


class ThreadAwareNavigateBackTool(ThreadAwareBaseTool):
    name: str = "navigate_back"
    description: str = "Navigate back to the previous page"
    args_schema: Type[BaseModel] = NavigateBackToolInput
    """Tool for navigating back in browser history."""

    def _run(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Navigate back
            try:
                page.go_back()
                self.release_sync_browser(thread_id)
                return "Navigated back to the previous page"
            except Exception as nav_error:
                self.release_sync_browser(thread_id)
                return f"Unable to navigate back: {str(nav_error)}"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error navigating back: {str(e)}")

    async def _arun(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Navigate back
            try:
                await page.go_back()
                await self.release_async_browser(thread_id)
                return "Navigated back to the previous page"
            except Exception as nav_error:
                await self.release_async_browser(thread_id)
                return f"Unable to navigate back: {str(nav_error)}"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error navigating back: {str(e)}")


class ThreadAwareExtractTextTool(ThreadAwareBaseTool):
    name: str = "extract_text"
    description: str = "Extract all the text on the current webpage"
    args_schema: Type[BaseModel] = ExtractTextToolInput
    """Tool for extracting text from a webpage."""

    def _run(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Extract text
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")

            self.release_sync_browser(thread_id)

            return soup.get_text(separator="\n").strip()
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error extracting text: {str(e)}")

    async def _arun(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Extract text
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            await self.release_async_browser(thread_id)

            return soup.get_text(separator="\n").strip()
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error extracting text: {str(e)}")


class ThreadAwareExtractHyperlinksTool(ThreadAwareBaseTool):
    name: str = "extract_hyperlinks"
    description: str = "Extract all hyperlinks on the current webpage"
    args_schema: Type[BaseModel] = ExtractHyperlinksToolInput
    """Tool for extracting hyperlinks from a webpage."""

    def _run(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Extract hyperlinks
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            links = []
            for link in soup.find_all("a", href=True):
                text = link.get_text().strip()
                href = link["href"]
                if isinstance(href, str) and (
                    href.startswith("http") or href.startswith("https")
                ):
                    links.append({"text": text, "url": href})

            self.release_sync_browser(thread_id)

            if not links:
                return "No hyperlinks found on the current page."

            return json.dumps(links, indent=2)
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error extracting hyperlinks: {str(e)}")

    async def _arun(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Extract hyperlinks
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            links = []
            for link in soup.find_all("a", href=True):
                text = link.get_text().strip()
                href = link["href"]
                if isinstance(href, str) and (
                    href.startswith("http") or href.startswith("https")
                ):
                    links.append({"text": text, "url": href})

            await self.release_async_browser(thread_id)

            if not links:
                return "No hyperlinks found on the current page."

            return json.dumps(links, indent=2)
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error extracting hyperlinks: {str(e)}")


class ThreadAwareGetElementsTool(ThreadAwareBaseTool):
    name: str = "get_elements"
    description: str = "Get elements from the webpage using a CSS selector"
    args_schema: Type[BaseModel] = GetElementsToolInput
    """Tool for getting elements from a webpage."""

    def _run(
        self,
        selector: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Get elements
            elements = page.query_selector_all(selector)
            if not elements:
                self.release_sync_browser(thread_id)
                return f"No elements found with selector '{selector}'"

            elements_text = []
            for i, element in enumerate(elements):
                text = element.text_content()
                elements_text.append(f"Element {i + 1}: {text.strip()}")

            self.release_sync_browser(thread_id)

            return "\n".join(elements_text)
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error getting elements: {str(e)}")

    async def _arun(
        self,
        selector: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Get elements
            elements = await page.query_selector_all(selector)
            if not elements:
                await self.release_async_browser(thread_id)
                return f"No elements found with selector '{selector}'"

            elements_text = []
            for i, element in enumerate(elements):
                text = await element.text_content()
                elements_text.append(f"Element {i + 1}: {text.strip()}")

            await self.release_async_browser(thread_id)

            return "\n".join(elements_text)
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error getting elements: {str(e)}")


class ThreadAwareCurrentWebPageTool(ThreadAwareBaseTool):
    name: str = "current_webpage"
    description: str = "Get information about the current webpage"
    args_schema: Type[BaseModel] = CurrentWebPageToolInput
    """Tool for getting information about the current webpage."""

    def _run(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Get information
            url = page.url
            title = page.title()

            self.release_sync_browser(thread_id)

            return f"URL: {url}\nTitle: {title}"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error getting current webpage info: {str(e)}")

    async def _arun(
        self,
        config: Optional[RunnableConfig] = None,
        **_: Any,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Get thread ID from config
            thread_id = self.get_thread_id(config)

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Get information
            url = page.url
            title = await page.title()

            await self.release_async_browser(thread_id)

            return f"URL: {url}\nTitle: {title}"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error getting current webpage info: {str(e)}")


class ThreadAwareTypeTool(ThreadAwareBaseTool):
    """Tool for typing text into input fields on a webpage."""

    name: str = "type_text"
    description: str = """Type text into an input field on the webpage.

Use this tool to:
- Fill in search boxes
- Enter form data (usernames, emails, passwords)
- Type into text areas
- Fill any input field identified by CSS selector

The selector should identify a single input element. Common patterns:
- '#search' - element with id="search"
- 'input[name="email"]' - input with name attribute
- '.search-input' - element with class "search-input"
- 'input[type="text"]' - first text input on page

This clears existing content before typing."""

    args_schema: Type[BaseModel] = TypeTextInput

    def _run(
        self,
        selector: str,
        text: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = self.get_sync_page(thread_id)
            page.fill(selector, text)
            self.release_sync_browser(thread_id)
            return f"Successfully typed '{text}' into element '{selector}'"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error typing text into '{selector}': {str(e)}")

    async def _arun(
        self,
        selector: str,
        text: str,
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = await self.get_async_page(thread_id)
            await page.fill(selector, text)
            await self.release_async_browser(thread_id)
            return f"Successfully typed '{text}' into element '{selector}'"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error typing text into '{selector}': {str(e)}")


class ThreadAwareScreenshotTool(ThreadAwareBaseTool):
    """Tool for capturing screenshots of the current webpage."""

    name: str = "take_screenshot"
    description: str = """Capture a screenshot of the current webpage.

Use this tool to:
- Verify that navigation or actions completed successfully
- Capture visual state for debugging
- Get visual context for decision-making
- Document the current page state

Returns a base64-encoded PNG image. Set full_page=True to capture
the entire scrollable page, or False (default) for just the visible viewport."""

    args_schema: Type[BaseModel] = ScreenshotInput

    def _run(
        self,
        capture_type: str = "viewport",
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = self.get_sync_page(thread_id)
            full_page = capture_type == "full_page"
            screenshot_bytes = page.screenshot(full_page=full_page, type="png")
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            self.release_sync_browser(thread_id)
            page_type = "full page" if full_page else "viewport"
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Screenshot captured ({page_type})",
                    "image_base64": screenshot_base64,
                    "format": "png",
                }
            )
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error taking screenshot: {str(e)}")

    async def _arun(
        self,
        capture_type: str = "viewport",
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = await self.get_async_page(thread_id)
            full_page = capture_type == "full_page"
            screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            await self.release_async_browser(thread_id)
            page_type = "full page" if full_page else "viewport"
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Screenshot captured ({page_type})",
                    "image_base64": screenshot_base64,
                    "format": "png",
                }
            )
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error taking screenshot: {str(e)}")


class ThreadAwareScrollTool(ThreadAwareBaseTool):
    """Tool for scrolling the webpage."""

    name: str = "scroll_page"
    description: str = """Scroll the webpage in the specified direction.

Directions: 'up', 'down', 'left', 'right'. Amount: pixels to scroll (default 500)."""

    args_schema: Type[BaseModel] = ScrollInput

    def _run(
        self,
        direction: str = "down",
        amount: int = 500,
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = self.get_sync_page(thread_id)

            delta_x, delta_y = _get_scroll_deltas(direction, amount)

            page.mouse.wheel(delta_x, delta_y)
            self.release_sync_browser(thread_id)
            return f"Scrolled {direction} by {amount} pixels"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error scrolling page: {str(e)}")

    async def _arun(
        self,
        direction: str = "down",
        amount: int = 500,
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = await self.get_async_page(thread_id)

            delta_x, delta_y = _get_scroll_deltas(direction, amount)

            await page.mouse.wheel(delta_x, delta_y)
            await self.release_async_browser(thread_id)
            return f"Scrolled {direction} by {amount} pixels"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")
            raise ToolException(f"Error scrolling page: {str(e)}")


class ThreadAwareWaitForElementTool(ThreadAwareBaseTool):
    """Tool for waiting until an element appears or reaches a specific state."""

    name: str = "wait_for_element"
    description: str = """Wait for an element to appear or reach a specific state.

States: 'visible' (default), 'attached', 'detached', 'hidden'.
Timeout in milliseconds (default: 30000)."""

    args_schema: Type[BaseModel] = WaitForElementInput

    def _run(
        self,
        selector: str,
        timeout: int = 30000,
        state: str = "visible",
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = self.get_sync_page(thread_id)

            page.wait_for_selector(selector, state=state, timeout=timeout)
            self.release_sync_browser(thread_id)
            return f"Element '{selector}' is now {state}"
        except Exception as e:
            if thread_id:
                try:
                    self.release_sync_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")

            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise ToolException(
                    f"Timeout waiting for element '{selector}' to be {state} "
                    f"after {timeout}ms. The element may not exist or may not "
                    "reach the expected state."
                )
            raise ToolException(f"Error waiting for element '{selector}': {error_msg}")

    async def _arun(
        self,
        selector: str,
        timeout: int = 30000,
        state: str = "visible",
        config: Optional[RunnableConfig] = None,
        **_: Any,
    ) -> str:
        thread_id = None
        try:
            thread_id = self.get_thread_id(config)
            page = await self.get_async_page(thread_id)

            await page.wait_for_selector(selector, state=state, timeout=timeout)
            await self.release_async_browser(thread_id)
            return f"Element '{selector}' is now {state}"
        except Exception as e:
            if thread_id:
                try:
                    await self.release_async_browser(thread_id)
                except Exception as release_error:
                    logger.warning(f"Error releasing browser: {release_error}")

            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise ToolException(
                    f"Timeout waiting for element '{selector}' to be {state} "
                    f"after {timeout}ms. The element may not exist or may not "
                    "reach the expected state."
                )
            raise ToolException(f"Error waiting for element '{selector}': {error_msg}")


def create_thread_aware_tools(
    session_manager: BrowserSessionManager,
) -> Dict[str, ThreadAwareBaseTool]:
    """
    Create thread-aware browser tools that use the session manager.

    Args:
        session_manager: The session manager to use for browser access

    Returns:
        Dictionary of thread-aware tools
    """
    # Import all required tool inputs
    try:
        import bs4  # noqa: F401  # type: ignore[import]
    except ImportError:
        import warnings

        warnings.warn(
            "The 'beautifulsoup4' package is required for extract_text and "
            "extract_hyperlinks tools. Please install it with "
            "'pip install beautifulsoup4'."
        )
    return {
        "navigate": ThreadAwareNavigateTool(_session_manager=session_manager),
        "click": ThreadAwareClickTool(_session_manager=session_manager),
        "navigate_back": ThreadAwareNavigateBackTool(_session_manager=session_manager),
        "extract_text": ThreadAwareExtractTextTool(_session_manager=session_manager),
        "extract_hyperlinks": ThreadAwareExtractHyperlinksTool(
            _session_manager=session_manager
        ),
        "get_elements": ThreadAwareGetElementsTool(_session_manager=session_manager),
        "current_webpage": ThreadAwareCurrentWebPageTool(
            _session_manager=session_manager
        ),
        "type_text": ThreadAwareTypeTool(_session_manager=session_manager),
        "screenshot": ThreadAwareScreenshotTool(_session_manager=session_manager),
        "scroll": ThreadAwareScrollTool(_session_manager=session_manager),
        "wait_for_element": ThreadAwareWaitForElementTool(
            _session_manager=session_manager
        ),
    }
