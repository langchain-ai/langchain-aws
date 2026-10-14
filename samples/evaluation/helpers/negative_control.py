"""Deliberately broken agents, for proving a check can actually fail.

STAND-IN TOOLS, stated plainly. Nothing here touches AgentCore Browser or Code
Interpreter, and nothing here says anything about answer accuracy. These agents exist
only so that "the check failed when the behavior broke" is a demonstrated fact rather
than an assumption.

A check that cannot fail is decoration. This is the cheapest way to find out which kind
you have written.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

__all__ = ["forced_screenshot_agent", "forced_serial_agent", "standin_browser"]


class _Url(BaseModel):
    url: str = Field(description="URL to navigate to")


class _NoArg(BaseModel):
    pass


def standin_browser(latency_s: float = 1.0) -> list[StructuredTool]:
    """Stand-in browser using the real toolkit's tool names.

    Non-zero latency is essential. With instant tools, serial and parallel execution are
    indistinguishable in wall clock, so a concurrency check would pass for the wrong
    reason. `take_screenshot` is present so that "the agent avoided it" is falsifiable.
    """
    async def navigate(url: str) -> str:
        await asyncio.sleep(latency_s)
        return f"Navigating to {url} returned status code 200"

    async def extract() -> str:
        await asyncio.sleep(latency_s / 4)
        return "Acme Corp. Revenue 1,000,000. Operating income 250,000. ($ in Thousands)"

    async def screenshot() -> str:
        await asyncio.sleep(latency_s / 4)
        return "[screenshot bytes omitted]"

    return [
        StructuredTool.from_function(coroutine=navigate, name="navigate_browser",
                                     description="Navigate the browser to a URL.",
                                     args_schema=_Url),
        StructuredTool.from_function(coroutine=extract, name="extract_text",
                                     description="Extract all visible text.",
                                     args_schema=_NoArg),
        StructuredTool.from_function(coroutine=screenshot, name="take_screenshot",
                                     description="Capture a screenshot.",
                                     args_schema=_NoArg),
    ]


def _agent(model: Any, coordinator: str, researcher: str, latency_s: float) -> Any:
    subagents = [{
        "name": f"research-{n}",
        "description": f"Researches company {n}.",
        "system_prompt": researcher,
        "tools": standin_browser(latency_s),
    } for n in ("alpha", "beta", "gamma")]
    return create_deep_agent(model=model, subagents=subagents, tools=[],
                             system_prompt=coordinator, name="negative-control",
                             checkpointer=None)


def forced_screenshot_agent(model: Any) -> Any:
    """A researcher instructed TO screenshot, so `no screenshots` must fail."""
    return _agent(
        model,
        "You are a research coordinator. Delegate to research subagents via task().",
        "You are a researcher. FIRST call take_screenshot. Then navigate_browser to "
        "https://acme.example, then take_screenshot again, then extract_text. You MUST "
        "call take_screenshot at least twice.",
        latency_s=0.4,
    )


def forced_serial_agent(model: Any) -> Any:
    """A coordinator forced to delegate one at a time.

    It still emits three `task()` calls, so a check that counts calls per turn scores
    this as parallel. Peak concurrency is 1, which is what actually happened.
    """
    return _agent(
        model,
        "You are a research coordinator. Research companies STRICTLY ONE AT A TIME. Call "
        "task() for exactly one company, wait for its result, and only then call task() "
        "for the next. NEVER emit two task() calls in the same response.",
        "You are a researcher. navigate_browser to https://acme.example then extract_text.",
        latency_s=1.2,
    )
