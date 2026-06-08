"""Regression test pinning that user-facing JSON serialization in
`langchain-aws` preserves non-ASCII characters.

Python's `json.dumps` defaults to `ensure_ascii=True` and escapes every
non-ASCII character to `\\uXXXX`. For end-user payloads — tool-call
arguments, tool schemas, Bedrock agent return-control events, trace
logs — that escaping is observable: CJK / emoji / accented characters
land in model prompts, returned strings, or persisted logs as
escape sequences and are unreadable on inspection. The same
convention is already established in `langchain-openai`'s chat model
and `langchain-core`'s `messages/utils.py:1810`.

Rather than mock the heavy Bedrock client surface, this test reads the
source of the affected modules and asserts every `json.dumps(` call
inside the relevant scope passes `ensure_ascii=False`. The check
survives any future refactor that keeps the call shape but flips the
kwarg back.
"""

from __future__ import annotations

import inspect
import re


def _calls_without_ensure_ascii(source: str) -> list[str]:
    """Return a list of `json.dumps(...)` calls in `source` that do NOT
    include `ensure_ascii=False` anywhere inside the call's parentheses.
    """
    pattern = re.compile(r"json\.dumps\(")
    offenders: list[str] = []
    for match in pattern.finditer(source):
        idx = match.end()
        depth = 1
        while depth > 0 and idx < len(source):
            ch = source[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            idx += 1
        call_text = source[match.start() : idx]
        if "ensure_ascii=False" not in call_text:
            offenders.append(re.sub(r"\s+", " ", call_text)[:140])
    return offenders


def test_function_calling_anyof_allof_schema_dumps_preserve_non_ascii() -> None:
    """`function_calling._get_type` serializes tool parameter schemas
    (including any CJK descriptions / titles) into the Anthropic tool
    spec. Without `ensure_ascii=False`, those descriptions land in the
    Bedrock request body as `\\uXXXX`.
    """
    from langchain_aws import function_calling as fc

    source = inspect.getsource(fc)
    offenders = _calls_without_ensure_ascii(source)
    assert not offenders, (
        f"{len(offenders)} `json.dumps` call(s) in "
        "`langchain_aws/function_calling.py` are missing "
        "`ensure_ascii=False`:\n  - " + "\n  - ".join(offenders)
    )


def test_bedrock_converse_tool_input_dump_preserves_non_ascii() -> None:
    """`ChatBedrockConverse` synthesizes a `[Called tool_name with
    parameters: {...}]` text block when converting tool calls for models
    without native tool support. Non-ASCII parameter values must reach
    the model verbatim, not as `\\uXXXX`."""
    from langchain_aws.chat_models import bedrock_converse

    source = inspect.getsource(bedrock_converse)
    offenders = _calls_without_ensure_ascii(source)
    assert not offenders, (
        f"{len(offenders)} `json.dumps` call(s) in "
        "`langchain_aws/chat_models/bedrock_converse.py` are missing "
        "`ensure_ascii=False`:\n  - " + "\n  - ".join(offenders)
    )


def test_agents_utils_response_and_trace_dumps_preserve_non_ascii() -> None:
    """The Bedrock agent stream parser dumps `returnControl` events into
    `response_text` (returned to the caller) and the accumulated
    `trace_log` for logging. Both should preserve non-ASCII."""
    from langchain_aws.agents import utils as agent_utils

    source = inspect.getsource(agent_utils)
    offenders = _calls_without_ensure_ascii(source)
    assert not offenders, (
        f"{len(offenders)} `json.dumps` call(s) in "
        "`langchain_aws/agents/utils.py` are missing `ensure_ascii=False`:\n  - "
        + "\n  - ".join(offenders)
    )
