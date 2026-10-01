"""AgentCore Runtime entrypoint for the research Deep Agent.

Packaged and deployed by Part 2. Two details here exist for evaluation rather than
for the agent working:

- The deploy helper sets the entry point to `["opentelemetry-instrument", "main.py"]`,
  which activates ADOT auto-instrumentation. That is what produces the `gen_ai.*`
  spans every evaluation step reads. Deploy without it and the agent runs fine but
  cannot be evaluated.
- The system prompt is read from an attached configuration bundle when one is
  present, which is what makes prompt A/B testing possible without redeploying.
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
log = app.logger
REGION = os.environ.get("AWS_REGION", "us-west-2")


def _prompt_from_bundle(default: str) -> str:
    """Prefer an attached configuration bundle's prompt over the baked-in one.

    Returns `default` when no bundle is attached, which is the normal case outside
    an A/B test.
    """
    try:
        from bedrock_agentcore.runtime import BedrockAgentCoreContext

        bundle = BedrockAgentCoreContext.get_config_bundle()
    except Exception as exc:  # noqa: BLE001 - a missing bundle must not break the run
        log.info("no configuration bundle (%s)", exc)
        return default
    if not bundle:
        return default
    config = bundle.get("configuration", bundle) if isinstance(bundle, dict) else {}
    prompt = config.get("system_prompt") or config.get("systemPrompt")
    if prompt:
        log.info("using bundle system prompt (%d chars)", len(prompt))
        return prompt
    return default


def _final_text(messages: list) -> str:
    """Last assistant message with text and no pending tool calls."""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        text = (
            content
            if isinstance(content, str)
            else "".join(
                block.get("text", "")
                for block in (content or [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
        )
        if text and text.strip() and not getattr(message, "tool_calls", None):
            return text
    return ""


@app.entrypoint
async def invoke(payload, context):
    """Handle one invocation end to end."""
    from research_agent import (
        COMPANIES,
        COORDINATOR_PROMPT,
        build_model,
        build_research_agent,
        cleanup,
    )

    prompt = payload.get("prompt") or (
        "Compare Snowflake, Datadog and MongoDB on Rule of 40 for their most recent "
        "fiscal year."
    )
    session_id = getattr(context, "session_id", None) or "no-session"
    log.info("invoke session=%s", session_id)

    default_prompt = COORDINATOR_PROMPT.format(
        company_list=", ".join(c.name for c in COMPANIES)
    )
    agent, toolkits = await build_research_agent(
        build_model(region=REGION),
        region=REGION,
        system_prompt=_prompt_from_bundle(default_prompt),
    )

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={
                # thread_id keys the browser and interpreter sessions, so tying it to
                # the runtime session keeps one caller's sandboxes together.
                "configurable": {"thread_id": session_id},
                # 60 rather than LangGraph's default 25: a four-subagent fan-out that
                # retries anything trips the default ceiling and dies mid-run.
                "recursion_limit": 60,
            },
        )
        return {"result": _final_text(result.get("messages", []))}
    except Exception as exc:  # noqa: BLE001
        # Surface failures. Insights clusters real errors, so swallowing them hides
        # the signal the improvement loop needs.
        log.exception("invocation failed session=%s", session_id)
        return {"result": "", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        await cleanup(toolkits)


if __name__ == "__main__":
    app.run()
