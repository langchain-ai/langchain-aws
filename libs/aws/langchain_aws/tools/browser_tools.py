"""Thread-aware browser tools that work with the browser session manager."""

import logging
from typing import Any, Dict, Optional, Type
from urllib.parse import urlparse

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .browser_session_manager import BrowserSessionManager
from .utils import aget_current_page, get_current_page

logger = logging.getLogger(__name__)

class NavigateToolInput(BaseModel):
    """Input for NavigateTool."""
    url: str = Field(description="URL to navigate to")


class ClickToolInput(BaseModel):
    """Input for ClickTool."""
    selector: str = Field(
        description="CSS selector for the element to click on"
    )


class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""
    selector: str = Field(
        description="CSS selector for elements to get"
    )


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


class ThreadAwareBaseTool(BaseTool):
    """Base class for thread-aware browser tools."""
    
    _session_manager: BrowserSessionManager
        
    def __init__(self, session_manager: BrowserSessionManager):
        """Initialize with a session manager."""
        super().__init__()
        self._session_manager = session_manager
        
    def get_thread_id(self, config: Optional[RunnableConfig] = None) -> str:
        """Extract thread ID from config."""
        thread_id = "default"

        if config and isinstance(config, dict):
            thread_id = config["configurable"]["thread_id"]
        
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
    ) -> str:
        """Use the sync tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                import json

                from bs4 import BeautifulSoup
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
                if href.startswith("http") or href.startswith("https"):
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
    ) -> str:
        """Use the async tool."""
        thread_id = None
        try:
            # Import BeautifulSoup
            try:
                import json

                from bs4 import BeautifulSoup
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
                if href.startswith("http") or href.startswith("https"):
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
                elements_text.append(f"Element {i+1}: {text.strip()}")
            
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
                elements_text.append(f"Element {i+1}: {text.strip()}")
            
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        config: RunnableConfig,
        **_,  # Ignore additional arguments
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
        import bs4  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn(
            "The 'beautifulsoup4' package is required for extract_text and extract_hyperlinks tools."
            " Please install it with 'pip install beautifulsoup4'."
        )
    return {
        "navigate": ThreadAwareNavigateTool(session_manager=session_manager),
        "click": ThreadAwareClickTool(session_manager=session_manager),
        "navigate_back": ThreadAwareNavigateBackTool(session_manager=session_manager),
        "extract_text": ThreadAwareExtractTextTool(session_manager=session_manager),
        "extract_hyperlinks": ThreadAwareExtractHyperlinksTool(session_manager=session_manager),
        "get_elements": ThreadAwareGetElementsTool(session_manager=session_manager),
        "current_webpage": ThreadAwareCurrentWebPageTool(session_manager=session_manager),
    }
