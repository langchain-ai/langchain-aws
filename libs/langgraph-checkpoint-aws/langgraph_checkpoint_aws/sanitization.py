"""Checkpoint sanitization utilities for Bedrock tool_call serialization.

When using AWS Bedrock Converse API, checkpoints containing tool_use blocks
may have their input/args fields serialized as JSON strings instead of dicts.
This module provides sanitization functions to fix this on checkpoint load.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)


def _parse_tool_args(args: Any) -> Any:
    """Safely deserialize tool args that may be JSON strings.

    Bedrock Converse API sometimes returns nested objects as JSON strings
    instead of proper dicts. This function handles both cases.

    Args:
        args: Tool arguments (can be dict, string, or other)

    Returns:
        Parsed arguments as dict if possible, original value otherwise
    """
    if args is None:
        return {}

    if isinstance(args, dict):
        return {
            k: _parse_tool_args(v) if isinstance(v, str) and v.startswith("{") else v
            for k, v in args.items()
        }

    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return args


def _sanitize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure all tool call args are dicts, not strings.

    Args:
        tool_calls: List of tool call dicts with 'name', 'args', 'id' keys

    Returns:
        Sanitized list with args properly deserialized
    """
    if not tool_calls:
        return tool_calls

    sanitized = []
    for tc in tool_calls:
        tc_copy = tc.copy()
        if "args" in tc_copy:
            original_args = tc_copy["args"]
            tc_copy["args"] = _parse_tool_args(original_args)

            if tc_copy["args"] != original_args and isinstance(original_args, str):
                logger.debug(
                    "Sanitized tool_call args for %s: string -> dict",
                    tc_copy.get("name", "unknown"),
                )

        sanitized.append(tc_copy)

    return sanitized


def _sanitize_content_blocks(content: Any) -> Any:
    """Sanitize content blocks in message content.

    AIMessage content can be a list with blocks like:
    [
        {"type": "text", "text": "Let me..."},
        {"type": "tool_use", "name": "...", "id": "...", "input": {...}}
    ]

    The 'input' field of tool_use blocks must be a dict, not a string.
    """
    if not isinstance(content, list):
        return content

    sanitized = []
    modified = False

    for block in content:
        if not isinstance(block, dict):
            sanitized.append(block)
            continue

        if block.get("type") == "tool_use" and "input" in block:
            original_input = block["input"]
            sanitized_input = _parse_tool_args(original_input)

            if sanitized_input != original_input and isinstance(original_input, str):
                logger.debug(
                    "Sanitized tool_use.input for %s: string -> dict",
                    block.get("name", "unknown"),
                )
                block_copy = block.copy()
                block_copy["input"] = sanitized_input
                sanitized.append(block_copy)
                modified = True
            else:
                sanitized.append(block)
        else:
            sanitized.append(block)

    return sanitized if modified else content


def _sanitize_additional_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Sanitize additional_kwargs which may contain tool_calls in Bedrock format."""
    if not kwargs or "tool_calls" not in kwargs:
        return kwargs

    kwargs_copy = kwargs.copy()
    sanitized_calls = []
    for tc in kwargs_copy.get("tool_calls", []):
        tc_copy = tc.copy()
        if "input" in tc_copy:
            tc_copy["input"] = _parse_tool_args(tc_copy["input"])
        if "args" in tc_copy:
            tc_copy["args"] = _parse_tool_args(tc_copy["args"])
        sanitized_calls.append(tc_copy)

    kwargs_copy["tool_calls"] = sanitized_calls
    return kwargs_copy


def sanitize_message(msg: BaseMessage) -> BaseMessage:
    """Sanitize a single message to ensure tool calls have dict args.

    Handles:
    1. msg.tool_calls - LangChain tool_calls attribute
    2. msg.additional_kwargs.tool_calls - Bedrock format
    3. msg.content - When content is a list with tool_use blocks

    Args:
        msg: A LangChain message

    Returns:
        Sanitized message with properly formatted tool calls
    """
    if not isinstance(msg, AIMessage):
        return msg

    needs_sanitization = False

    sanitized_tool_calls = None
    if msg.tool_calls:
        sanitized_tool_calls = _sanitize_tool_calls(msg.tool_calls)
        if sanitized_tool_calls != msg.tool_calls:
            needs_sanitization = True

    sanitized_kwargs = _sanitize_additional_kwargs(msg.additional_kwargs)
    if sanitized_kwargs != msg.additional_kwargs:
        needs_sanitization = True

    sanitized_content = msg.content
    if isinstance(msg.content, list):
        sanitized_content = _sanitize_content_blocks(msg.content)
        if sanitized_content != msg.content:
            needs_sanitization = True

    if needs_sanitization:
        return AIMessage(
            content=sanitized_content,
            tool_calls=sanitized_tool_calls or msg.tool_calls,
            additional_kwargs=sanitized_kwargs,
            response_metadata=getattr(msg, "response_metadata", {}),
            id=msg.id,
        )

    return msg


def sanitize_checkpoint(checkpoint: dict | None) -> dict | None:
    """Sanitize a checkpoint's messages to fix malformed tool_call args.

    This is the main entry point for checkpoint sanitization.

    Args:
        checkpoint: Raw checkpoint dict

    Returns:
        Sanitized checkpoint with properly formatted tool calls
    """
    if not checkpoint:
        return checkpoint

    if "channel_values" not in checkpoint:
        return checkpoint

    channel_values = checkpoint["channel_values"]
    if not isinstance(channel_values, dict) or "messages" not in channel_values:
        return checkpoint

    messages = channel_values["messages"]
    if not messages:
        return checkpoint

    sanitized_messages = []
    sanitized_count = 0

    for msg in messages:
        original = msg
        sanitized_msg = sanitize_message(msg)
        if sanitized_msg is not original:
            sanitized_count += 1
        sanitized_messages.append(sanitized_msg)

    if sanitized_count == 0:
        return checkpoint

    logger.info(
        "Sanitized %d message(s) with malformed tool_call args", sanitized_count
    )

    sanitized_checkpoint = checkpoint.copy()
    sanitized_checkpoint["channel_values"] = channel_values.copy()
    sanitized_checkpoint["channel_values"]["messages"] = sanitized_messages
    return sanitized_checkpoint
