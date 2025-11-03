"""Shared fixtures and test utilities for checkpoint tests."""

from unittest.mock import Mock, patch

import pytest

# Test Constants
TEST_TABLE_NAME = "test-table"
TEST_BASE_TABLE_NAME = "base_table"
TEST_THREAD_ID = "thread_123"
TEST_CHECKPOINT_NS = "ns"
TEST_CHECKPOINT_ID = "cp_id"
TEST_TTL_SECONDS = 3600
TEST_S3_BUCKET = "test-bucket"
TEST_TASK_ID = "task1"
TEST_CHANNEL = "channel1"


@pytest.fixture
def mock_dynamodb_client():
    """Create a mock DynamoDB client."""
    return Mock()


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    return Mock()


@pytest.fixture
def mock_serializer():
    """Create a mock CheckpointSerializer."""
    return Mock()


@pytest.fixture
def mock_storage_strategy():
    """Create a mock StorageStrategy."""
    return Mock()


@pytest.fixture
def mock_repository_setup():
    """Mock setup for UnifiedRepository tests.

    Returns a dictionary with all mocked dependencies for repository tests.
    """
    mock_client = Mock()
    mock_serializer = Mock()
    mock_storage = Mock()

    # Set up common return values
    mock_serializer.dumps.return_value = (b"data", "msgpack")
    mock_serializer.loads.return_value = {"test": "data"}
    mock_storage.store_data.return_value = ("DYNAMODB", "key123")
    mock_storage.retrieve_data.return_value = b"data"

    return {
        "client": mock_client,
        "serializer": mock_serializer,
        "storage": mock_storage,
        "base_table": TEST_BASE_TABLE_NAME,
    }


@pytest.fixture
def mock_saver_dependencies():
    """Mock all DynamoDBSaver dependencies.

    Returns a dictionary with all mocked dependencies.
    Use this fixture to reduce repetitive patching in saver tests.
    """
    with (
        patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.create_dynamodb_client"
        ) as mock_ddb,
        patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.create_s3_client"
        ) as mock_s3,
        patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.CheckpointSerializer"
        ) as mock_ser,
        patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.StorageStrategy"
        ) as mock_stor,
        patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.UnifiedRepository"
        ) as mock_repo,
    ):
        # Set up default return values
        mock_ddb_instance = Mock()
        mock_ddb.return_value = mock_ddb_instance
        mock_s3.return_value = Mock()

        yield {
            "create_dynamodb_client": mock_ddb,
            "create_s3_client": mock_s3,
            "serializer": mock_ser,
            "storage": mock_stor,
            "repo": mock_repo,
            "ddb_instance": mock_ddb_instance,
        }


def build_checkpoint_item(
    thread_id: str = TEST_THREAD_ID,
    checkpoint_ns: str = TEST_CHECKPOINT_NS,
    checkpoint_id: str = TEST_CHECKPOINT_ID,
    ref_loc: str = "DYNAMODB",
    ref_key: str | None = None,
    compression_type: str | None = None,
) -> dict:
    """Build a DynamoDB checkpoint item for testing.

    Args:
        thread_id: Thread identifier
        checkpoint_ns: Checkpoint namespace
        checkpoint_id: Checkpoint identifier
        ref_loc: Storage location (DYNAMODB or S3)
        ref_key: Reference key for payload
        compression_type: Compression type string ("gzip" or None)

    Returns:
        Dictionary representing a DynamoDB item
    """
    if ref_key is None:
        ref_key = f"key_{checkpoint_id}"

    item = {
        "PK": {"S": f"CHECKPOINT_{thread_id}"},
        "SK": {"S": f"{checkpoint_ns}#{checkpoint_id}"},
        "id": {"S": checkpoint_id},
        "ns": {"S": checkpoint_ns},
        "ref_key": {"S": ref_key},
        "ref_loc": {"S": ref_loc},
        "type": {"S": "msgpack"},
    }

    if compression_type:
        item["compression"] = {"S": compression_type}

    return item


def build_write_item(
    thread_id: str = TEST_THREAD_ID,
    checkpoint_ns: str = TEST_CHECKPOINT_NS,
    checkpoint_id: str = TEST_CHECKPOINT_ID,
    task_id: str = "task1",
    idx: int = 0,
    channel: str = "channel1",
    ref_loc: str = "DYNAMODB",
    ref_key: str | None = None,
) -> dict:
    """Build a DynamoDB write item for testing.

    Args:
        thread_id: Thread identifier
        checkpoint_ns: Checkpoint namespace
        checkpoint_id: Checkpoint identifier
        task_id: Task identifier
        idx: Write index
        channel: Channel name
        ref_loc: Storage location
        ref_key: Reference key for value

    Returns:
        Dictionary representing a DynamoDB write item
    """
    if ref_key is None:
        ref_key = f"write_key_{task_id}_{idx}"

    return {
        "PK": {"S": f"WRITES_{thread_id}#{checkpoint_ns}#{checkpoint_id}"},
        "SK": {"S": f"{task_id}#{idx}"},
        "task_id": {"S": task_id},
        "idx": {"N": str(idx)},
        "channel": {"S": channel},
        "ref_key": {"S": ref_key},
        "ref_loc": {"S": ref_loc},
        "value_type": {"S": "msgpack"},
    }


def build_checkpoint_data(
    thread_id: str = TEST_THREAD_ID,
    checkpoint_id: str = TEST_CHECKPOINT_ID,
    checkpoint_ns: str = TEST_CHECKPOINT_NS,
    parent_checkpoint_id: str | None = None,
) -> dict:
    """Build checkpoint data dictionary for testing.

    Args:
        thread_id: Thread identifier
        checkpoint_id: Checkpoint identifier
        checkpoint_ns: Checkpoint namespace
        parent_checkpoint_id: Parent checkpoint ID (optional)

    Returns:
        Dictionary with checkpoint data
    """
    return {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint": {"id": checkpoint_id, "data": "test"},
        "metadata": {"step": 1},
        "parent_checkpoint_id": parent_checkpoint_id,
    }
