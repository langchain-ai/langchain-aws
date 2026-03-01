import os
from collections import namedtuple

import boto3
import pytest
from langchain_aws import ChatBedrock
from langchain_core.tools import Tool

from tests.integration_tests.utils import add, get_weather, multiply

##########################################################
# AWS
##########################################################


@pytest.fixture(scope="session")
def aws_region() -> str:
    """Get AWS region from environment or use default."""
    return os.environ.get("AWS_REGION", "us-west-2")


##########################################################
# Bedrock
##########################################################


@pytest.fixture(scope="session")
def bedrock_model(aws_region: str) -> ChatBedrock:
    """Create ChatBedrock model instance for integration tests."""
    return ChatBedrock(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region=aws_region,
    )


##########################################################
# DynamoDB
##########################################################


@pytest.fixture(scope="session")
def dynamodb_client(aws_region: str):
    """Create DynamoDB client."""
    return boto3.client("dynamodb", region_name=aws_region)


##########################################################
# S3
##########################################################


@pytest.fixture(scope="session")
def s3_client(aws_region: str):
    """Create S3 client."""
    return boto3.client("s3", region_name=aws_region)


##########################################################
# Agents
##########################################################


@pytest.fixture(scope="session")
def agent_tools() -> tuple[Tool, ...]:
    """Return a named tuple of tools for agent tests."""

    _TestTools = namedtuple(
        "_TestTools",
        [
            "add",
            "multiply",
            "get_weather",
        ],
    )
    return _TestTools(
        add=add,
        multiply=multiply,
        get_weather=get_weather,
    )
