from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import content as types


def _convert_annotation_from_v1_to_converse(annotation: types.Annotation) -> dict[str, Any]:
    """Convert LangChain annotation format to Converse's native citation format."""
    if annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    out: dict[str, Any] = {}
    if "title" in annotation:
        out["title"] = annotation["title"]
    if "cited_text" in annotation:
        out["source_content"] = [{"text": annotation["cited_text"]}]

    for key, value in annotation.get("extras", {}).items():
        if key not in out:
            out[key] = value
    
    return out


def _convert_from_v1_to_converse(
    content: list[types.ContentBlock],
    model_provider: Optional[str],
) -> list[dict[str, Any]]:
    new_content: list = []
    for block in content:
        if block["type"] == "text":
            if model_provider == "bedrock_converse" and "annotations" in block:
                new_block: dict[str, Any] = {"type": "text"}
                new_block["citations"] = [
                    _convert_annotation_from_v1_to_converse(a) for a in block["annotations"]
                ]
                if "text" in block:
                    new_block["text"] = block["text"]
            else:
                new_block = {"text": block.get("text", ""), "type": "text"}
            new_content.append(new_block)

        elif block["type"] == "tool_call":
            new_content.append(
                {
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": block.get("args", {}),
                    "id": block.get("id", ""),
                }
            )

        elif block["type"] == "tool_call_chunk":
            if isinstance(block["args"], str):
                try:
                    input_ = json.loads(block["args"] or "{}")
                except json.JSONDecodeError:
                    input_ = {}
            else:
                input_ = block.get("args") or {}
            new_content.append(
                {
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": input_,
                    "id": block.get("id", ""),
                }
            )

        elif block["type"] == "reasoning" and model_provider == "bedrock_converse":
            new_block = {"type": "reasoning_content"}
            if "reasoning" in block:
                if "reasoning_content" not in new_block:
                    new_block["reasoning_content"] = {}
                new_block["reasoning_content"]["text"] = block["reasoning"]
            if signature := block.get("extras", {}).get("signature"):
                if "reasoning_content" not in new_block:
                    new_block["reasoning_content"] = {}
                new_block["reasoning_content"]["signature"] = signature

            new_content.append(new_block)

        elif (
            block["type"] == "non_standard"
            and "value" in block
            and model_provider == "bedrock_converse"
        ):
            new_content.append(block["value"])
        else:
            new_content.append(block)

    return new_content
