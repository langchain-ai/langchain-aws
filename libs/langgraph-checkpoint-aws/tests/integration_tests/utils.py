"""Shared utilities for integration tests.

Provides common helpers for AWS resource setup and permission handling
so individual test modules don't duplicate boilerplate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def skip_on_aws_403(call_fn: Callable, description: str) -> Any:
    """Execute *call_fn*; skip the test on AWS permission errors.

    Args:
        call_fn: Zero-argument callable that makes an AWS API call.
        description: Human-readable label for the operation (used in
            the skip message).

    Returns:
        Whatever *call_fn* returns on success.

    Raises:
        ClientError: Re-raised if the error is not a permission error.
    """
    try:
        return call_fn()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("AccessDenied", "AccessDeniedException", "403"):
            pytest.skip(f"Insufficient permissions for {description}, skipping.")
        else:
            raise


def ensure_dynamodb_table(
    table_name: str,
    region_name: str,
) -> str:
    """Ensure a DynamoDB table exists, creating it if needed.

    The table uses a composite key (PK/SK) with PAY_PER_REQUEST billing,
    matching the schema expected by ``DynamoDBSaver``.

    Args:
        table_name: Name of the DynamoDB table.
        region_name: AWS region for the table.

    Returns:
        The table name (pass-through for convenience).
    """
    dynamodb = boto3.client("dynamodb", region_name=region_name)

    try:
        skip_on_aws_403(
            lambda: dynamodb.describe_table(TableName=table_name),
            f"DynamoDB DescribeTable on {table_name}",
        )
        logger.info("DynamoDB table '%s' already exists", table_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info("Creating DynamoDB table '%s'...", table_name)
            skip_on_aws_403(
                lambda: dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {"AttributeName": "PK", "KeyType": "HASH"},
                        {"AttributeName": "SK", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "PK", "AttributeType": "S"},
                        {"AttributeName": "SK", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                ),
                "DynamoDB CreateTable",
            )
            waiter = dynamodb.get_waiter("table_exists")
            skip_on_aws_403(
                lambda: waiter.wait(TableName=table_name),
                "DynamoDB WaitForTable",
            )
            logger.info("DynamoDB table '%s' created successfully", table_name)
        else:
            raise

    return table_name


def ensure_agentcore_memory(
    memory_id: str,
    region_name: str,
) -> str:
    """Ensure an AgentCore memory is accessible, creating one if needed.

    Attempts to probe the given memory ID.  If the server rejects it
    (validation error, not found), falls back to discovering an
    existing ACTIVE memory or creating a new one.

    Args:
        memory_id: The memory ID to verify or create.
        region_name: AWS region for the AgentCore service.

    Returns:
        A valid, reachable memory ID.
    """
    from langgraph_checkpoint_aws.agentcore.saver import (
        AgentCoreMemorySaver,
    )

    saver = AgentCoreMemorySaver(memory_id=memory_id, region_name=region_name)
    try:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "probe0",
                "actor_id": "probe0",
                "checkpoint_ns": "",
            }
        }
        list(saver.list(config))
        logger.info("AgentCore memory '%s' is reachable", memory_id)
        return memory_id
    except Exception as e:
        error_code = ""
        if hasattr(e, "response"):
            error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in (
            "AccessDenied",
            "AccessDeniedException",
            "403",
        ):
            pytest.skip(
                f"Insufficient permissions for AgentCore memory "
                f"'{memory_id}', skipping."
            )
        if error_code in (
            "ValidationException",
            "ResourceNotFoundException",
        ):
            logger.info(
                "Memory '%s' unavailable (%s), discovering or creating one.",
                memory_id,
                error_code,
            )
            return _discover_or_create_memory(region_name)
        raise


def _discover_or_create_memory(region_name: str) -> str:
    """Find an existing AgentCore memory or create a new one.

    Args:
        region_name: AWS region for the AgentCore control plane.

    Returns:
        A valid memory ID.
    """
    control = skip_on_aws_403(
        lambda: boto3.client("bedrock-agentcore-control", region_name=region_name),
        "AgentCore control plane client",
    )

    # Try to reuse an existing ACTIVE memory
    try:
        resp = skip_on_aws_403(
            lambda: control.list_memories(maxResults=5),
            "AgentCore ListMemories",
        )
        for mem in resp.get("memories", []):
            if mem.get("status") == "ACTIVE":
                logger.info("Reusing existing AgentCore memory: %s", mem["id"])
                return mem["id"]
    except Exception:
        logger.warning("Failed to list AgentCore memories, will create one.")

    # No existing memory found — create one
    try:
        create_resp = skip_on_aws_403(
            lambda: control.create_memory(
                name="langraphIntegTest",
            ),
            "AgentCore CreateMemory",
        )
        new_id = create_resp["memory"]["id"]
        logger.info("Created AgentCore memory: %s", new_id)

        # Wait for ACTIVE status
        import time

        for _ in range(30):
            status_resp = control.get_memory(memoryId=new_id)
            if status_resp["memory"]["status"] == "ACTIVE":
                return new_id
            time.sleep(2)

        pytest.skip(
            f"AgentCore memory '{new_id}' did not become ACTIVE within timeout."
        )
    except Exception as exc:
        pytest.skip(f"Failed to create AgentCore memory: {exc}")


def create_bedrock_session(
    region_name: str,
) -> tuple[Any, str, Callable[[], None]]:
    """Create a Bedrock session saver with a pre-created session.

    ``BedrockSessionSaver`` requires a pre-created session whose UUID
    is used as the ``thread_id``.  This helper creates the saver, a
    session, and returns a cleanup callable.

    Args:
        region_name: AWS region for the Bedrock runtime.

    Returns:
        A tuple of ``(saver, session_id, cleanup_fn)`` where
        *cleanup_fn* accepts no arguments and tears down the session.
    """
    from langgraph_checkpoint_aws.saver import BedrockSessionSaver

    saver = BedrockSessionSaver(region_name=region_name)
    client = saver.session_client.client

    try:
        session_resp = skip_on_aws_403(
            lambda: client.create_session(),
            "Bedrock CreateSession",
        )
    except ClientError:
        pytest.skip("Failed to create Bedrock session, skipping.")

    session_id = session_resp["sessionId"]
    logger.info("Created Bedrock session: %s", session_id)

    def cleanup() -> None:
        try:
            client.end_session(sessionIdentifier=session_id)
            client.delete_session(sessionIdentifier=session_id)
            logger.info("Deleted Bedrock session: %s", session_id)
        except Exception:
            logger.warning("Failed to delete Bedrock session: %s", session_id)

    return saver, session_id, cleanup
