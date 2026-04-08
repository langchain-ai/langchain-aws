"""Conformance tests for DynamoDBSaver against DynamoDB Local.

Requires a DynamoDB Local instance running on localhost:8000.
Start one with:  docker run -d -p 8000:8000 amazon/dynamodb-local
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import boto3
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

from langgraph_checkpoint_aws.checkpoint.dynamodb import DynamoDBSaver

DYNAMODB_ENDPOINT = os.getenv("DYNAMODB_ENDPOINT", "http://localhost:8000")
DYNAMODB_REGION = "us-east-1"
TABLE_NAME = f"conformance-{uuid4().hex[:8]}"


def _dynamodb_available() -> bool:
    try:
        client = boto3.client(
            "dynamodb",
            region_name=DYNAMODB_REGION,
            endpoint_url=DYNAMODB_ENDPOINT,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        client.list_tables(Limit=1)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _dynamodb_available(),
        reason="DynamoDB Local not available",
    ),
    pytest.mark.allow_hosts(["127.0.0.1", "localhost", "::1"]),
]


def _create_table() -> None:
    client = boto3.client(
        "dynamodb",
        region_name=DYNAMODB_REGION,
        endpoint_url=DYNAMODB_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    client.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=TABLE_NAME)


def _delete_table() -> None:
    client = boto3.client(
        "dynamodb",
        region_name=DYNAMODB_REGION,
        endpoint_url=DYNAMODB_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    try:
        client.delete_table(TableName=TABLE_NAME)
    except Exception:
        pass


async def dynamodb_lifespan() -> AsyncGenerator[None, None]:
    _create_table()
    yield
    _delete_table()


@checkpointer_test(name="DynamoDBSaver", lifespan=dynamodb_lifespan)
async def dynamodb_checkpointer() -> AsyncGenerator[DynamoDBSaver, None]:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    saver = DynamoDBSaver(
        table_name=TABLE_NAME,
        region_name=DYNAMODB_REGION,
        endpoint_url=DYNAMODB_ENDPOINT,
    )
    yield saver


@pytest.mark.asyncio
async def test_conformance() -> None:
    report = await validate(
        dynamodb_checkpointer,
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base(), f"Base tests failed: {report.to_dict()}"
