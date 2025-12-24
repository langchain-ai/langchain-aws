import base64
import datetime
import json
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from botocore.client import BaseClient
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.constants import TASKS

from langgraph_checkpoint_aws.constants import CHECKPOINT_PREFIX, WRITES_PREFIX
from langgraph_checkpoint_aws.models import (
    BedrockSessionContentBlock,
    InvocationStepPayload,
    SessionCheckpoint,
    SessionPendingWrite,
)


@pytest.fixture
def mock_boto_client():
    mock_client = Mock(spec=BaseClient)

    mock_client.create_session = MagicMock()
    mock_client.get_session = MagicMock()
    mock_client.end_session = MagicMock()
    mock_client.delete_session = MagicMock()

    mock_client.create_invocation = MagicMock()
    mock_client.list_invocations = MagicMock()

    mock_client.put_invocation_step = MagicMock()
    mock_client.get_invocation_step = MagicMock()
    mock_client.list_invocation_steps = MagicMock()
    return mock_client


@pytest.fixture
def sample_session_id():
    return str(uuid4())


@pytest.fixture
def sample_invocation_id():
    return str(uuid4())


@pytest.fixture
def sample_invocation_step_id():
    return str(uuid4())


@pytest.fixture
def sample_session_arn():
    return f"arn:aws:bedrock:us-west-2:123456789012:session/{uuid4()}"


@pytest.fixture
def sample_kms_key_arn():
    return f"arn:aws:kms:us-west-2:123456789012:key/{uuid4()}"


@pytest.fixture
def sample_timestamp():
    return datetime.datetime.now(datetime.timezone.utc)


@pytest.fixture
def sample_metadata():
    return {"key1": "value1", "key2": "value2"}


@pytest.fixture
def sample_create_session_response(
    sample_session_id, sample_session_arn, sample_timestamp
):
    return {
        "sessionId": sample_session_id,
        "sessionArn": sample_session_arn,
        "createdAt": sample_timestamp,
        "sessionStatus": "ACTIVE",
    }


@pytest.fixture
def sample_get_session_response(
    sample_session_id, sample_session_arn, sample_timestamp, sample_kms_key_arn
):
    return {
        "sessionId": sample_session_id,
        "sessionArn": sample_session_arn,
        "sessionStatus": "ACTIVE",
        "createdAt": sample_timestamp,
        "lastUpdatedAt": sample_timestamp,
        "sessionMetadata": {},
        "encryptionKeyArn": sample_kms_key_arn,
    }


@pytest.fixture
def sample_create_invocation_response(
    sample_invocation_id,
):
    return {
        "invocationId": sample_invocation_id,
    }


@pytest.fixture
def sample_list_invocation_response(
    sample_session_id, sample_invocation_id, sample_timestamp
):
    return {
        "invocationSummaries": [
            {
                "sessionId": sample_session_id,
                "invocationId": sample_invocation_id,
                "createdAt": sample_timestamp,
            }
        ],
        "nextToken": None,
    }


@pytest.fixture
def sample_invocation_step_payload():
    return InvocationStepPayload(
        content_blocks=[BedrockSessionContentBlock(text="sample text")]
    )


@pytest.fixture
def sample_invocation_step_summary(
    sample_session_id, sample_invocation_id, sample_timestamp, sample_invocation_step_id
):
    return {
        "sessionId": sample_session_id,
        "invocationId": sample_invocation_id,
        "invocationStepId": sample_invocation_step_id,
        "invocationStepTime": sample_timestamp,
    }


@pytest.fixture
def sample_put_invocation_step_response(sample_invocation_step_id):
    return {
        "invocationStepId": sample_invocation_step_id,
    }


@pytest.fixture
def sample_get_invocation_step_response(sample_invocation_step_summary):
    return {
        "invocationStep": {
            **sample_invocation_step_summary,
            "payload": {"contentBlocks": [{"text": "sample text"}]},
        }
    }


@pytest.fixture
def sample_list_invocation_steps_response(sample_invocation_step_summary):
    return {
        "invocationStepSummaries": [sample_invocation_step_summary],
        "nextToken": None,
    }


@pytest.fixture
def sample_session_pending_write(sample_invocation_step_summary):
    return SessionPendingWrite(
        step_type=WRITES_PREFIX,
        thread_id=sample_invocation_step_summary["sessionId"],
        checkpoint_ns=sample_invocation_step_summary["invocationId"],
        checkpoint_id=sample_invocation_step_summary["invocationStepId"],
        task_id=str(uuid4()),
        channel="test_channel",
        value=["json", base64.b64encode(json.dumps({"test": "test"}).encode())],
        task_path="/test/path",
        write_idx=1,
    )


@pytest.fixture
def sample_session_pending_write_with_sends(sample_invocation_step_summary):
    return [
        SessionPendingWrite(
            step_type=WRITES_PREFIX,
            thread_id=sample_invocation_step_summary["sessionId"],
            checkpoint_ns=sample_invocation_step_summary["invocationId"],
            checkpoint_id=sample_invocation_step_summary["invocationStepId"],
            task_id="3",
            channel="test_channel",
            value=["json", base64.b64encode(json.dumps({"k1": "v1"}).encode())],
            task_path="/test1/path1",
            write_idx=1,
        ),
        SessionPendingWrite(
            step_type=WRITES_PREFIX,
            thread_id=sample_invocation_step_summary["sessionId"],
            checkpoint_ns=sample_invocation_step_summary["invocationId"],
            checkpoint_id=sample_invocation_step_summary["invocationStepId"],
            task_id="3",
            channel=TASKS,
            value=["json", base64.b64encode(json.dumps({"k3": "v3"}).encode())],
            task_path="/test3/path3",
            write_idx=1,
        ),
        SessionPendingWrite(
            step_type=WRITES_PREFIX,
            thread_id=sample_invocation_step_summary["sessionId"],
            checkpoint_ns=sample_invocation_step_summary["invocationId"],
            checkpoint_id=sample_invocation_step_summary["invocationStepId"],
            task_id="2",
            channel=TASKS,
            value=["json", base64.b64encode(json.dumps({"k2": "v2"}).encode())],
            task_path="/test2/path2",
            write_idx=1,
        ),
    ]


@pytest.fixture
def sample_session_checkpoint(sample_invocation_step_summary):
    return SessionCheckpoint(
        step_type=CHECKPOINT_PREFIX,
        thread_id=sample_invocation_step_summary["sessionId"],
        checkpoint_ns=sample_invocation_step_summary["invocationId"],
        checkpoint_id=sample_invocation_step_summary["invocationStepId"],
        checkpoint=("json", b"e30="),
        metadata=(
            "json",
            base64.b64encode(json.dumps({"key": "value"}).encode()).decode(),
        ),
        parent_checkpoint_id=None,
        channel_values={},
        version={},
    )


@pytest.fixture
def sample_checkpoint(sample_timestamp):
    return Checkpoint(
        v=1,
        id="checkpoint_123",
        ts=sample_timestamp.isoformat(),
        channel_values={
            "default": "value1",
            "tasks": ["task1", "task2"],
            "results": {"status": "completed"},
        },
        channel_versions={"default": "v1", "tasks": "v2", "results": "v1"},
        versions_seen={
            "node1": {"default": "v1", "tasks": "v2"},
            "node2": {"results": "v1"},
        },
        pending_sends=[],
    )


@pytest.fixture
def sample_checkpoint_metadata(sample_timestamp):
    return CheckpointMetadata(
        source="input",
        step=-1,
        writes={"node1": ["write1", "write2"], "node2": {"key": "value"}},
        parents={
            "namespace1": "parent_checkpoint_1",
            "namespace2": "parent_checkpoint_2",
        },
    )
