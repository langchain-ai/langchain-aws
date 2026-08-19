"""Utilities for the Playwright browser tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from langchain_core.runnables.config import RunnableConfig

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage


_CHECKPOINT_NS_SEP = "|"


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


def get_session_key(
    config: Optional[RunnableConfig] = None,
    *,
    checkpoint_ns_scope: Literal["full", "parent"] = "full",
) -> str:
    """Build a session key from RunnableConfig.

    ``checkpoint_ns_scope`` controls how much of ``checkpoint_ns`` is used:
    - ``"full"`` (default): append the entire ``checkpoint_ns``, so every tool
      call gets its own session.
    - ``"parent"``: drop the innermost (per-tool-call) ``checkpoint_ns`` segment
      and append only the enclosing scope.

    Examples::

        Top-level agent (ns "tools:abc"):
            full   -> "thread-001:tools:abc"
            parent -> "thread-001"
        Subagent (ns "sub-a:1|tools:xyz"):
            full   -> "thread-001:sub-a:1|tools:xyz"
            parent -> "thread-001:sub-a:1"

    Args:
        config: LangGraph ``RunnableConfig`` passed to tool invocations.
        checkpoint_ns_scope: How much of ``checkpoint_ns`` to include in the key.

    Returns:
        String key for indexing session dictionaries.
    """
    if not config or not isinstance(config, dict):
        return "default"

    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default")
    checkpoint_ns = configurable.get("checkpoint_ns", "")

    if checkpoint_ns_scope == "parent":
        checkpoint_ns = (
            checkpoint_ns.rsplit(_CHECKPOINT_NS_SEP, 1)[0]
            if _CHECKPOINT_NS_SEP in checkpoint_ns
            else ""
        )

    if checkpoint_ns:
        return f"{thread_id}:{checkpoint_ns}"
    return thread_id
