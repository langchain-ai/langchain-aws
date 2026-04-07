"""Utilities for the Playwright browser tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.runnables.config import RunnableConfig

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage


async def aget_current_page(browser: AsyncBrowser) -> AsyncPage:
    """
    Asynchronously get the current page of the browser.

    Args:
        browser: The browser (AsyncBrowser) to get the current page from.

    Returns:
        AsyncPage: The current page.

    """
    if not browser.contexts:
        context = await browser.new_context()
        return await context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return await context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


def get_current_page(browser: SyncBrowser) -> SyncPage:
    """
    Get the current page of the browser.
    Args:
        browser: The browser to get the current page from.

    Returns:
        SyncPage: The current page.

    """
    if not browser.contexts:
        context = browser.new_context()
        return context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


def get_session_key(config: Optional[RunnableConfig] = None) -> str:
    """Build a session key from RunnableConfig.

    Uses thread_id as the base key and appends checkpoint_ns
    when present.  This ensures parallel LangGraph subgraphs get
    separate sessions even when they share a thread_id.

    Examples::

        Top-level agent:   "thread-001"
        Subagent A:        "thread-001:research-acme:abc123"
        Subagent B:        "thread-001:research-beta:def456"

    Args:
        config: LangGraph ``RunnableConfig`` passed to tool invocations.

    Returns:
        String key for indexing session dictionaries.
    """
    if not config or not isinstance(config, dict):
        return "default"

    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default")
    checkpoint_ns = configurable.get("checkpoint_ns", "")

    if checkpoint_ns:
        return f"{thread_id}:{checkpoint_ns}"
    return thread_id
