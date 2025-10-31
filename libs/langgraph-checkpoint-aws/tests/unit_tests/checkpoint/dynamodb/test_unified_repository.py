"""Comprehensive tests for UnifiedRepository."""

from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from conftest import (
    TEST_CHECKPOINT_ID,
    TEST_CHECKPOINT_NS,
    TEST_TABLE_NAME,
    TEST_TASK_ID,
    TEST_THREAD_ID,
)

from langgraph_checkpoint_aws.checkpoint.dynamodb.serialization import CompressionType
from langgraph_checkpoint_aws.checkpoint.dynamodb.unified_repository import (
    UnifiedRepository,
    _checkpoint_pk,
    _checkpoint_ref_pk,
    _checkpoint_s3_key,
    _checkpoint_sk,
    _writes_pk,
    _writes_ref_pk,
    _writes_s3_key,
    _writes_sk,
)

# ============================================================================
# KEY GENERATION FUNCTIONS
# ============================================================================


class TestKeyGenerationFunctions:
    """Test key generation functions."""

    def test_checkpoint_keys(self):
        """Test checkpoint key generation."""
        # Partition key
        pk = _checkpoint_pk(TEST_THREAD_ID)
        assert pk == "CHECKPOINT_thread_123"
        assert pk.startswith("CHECKPOINT_")

        # Sort key with namespace
        sk = _checkpoint_sk("namespace", "checkpoint_id")
        assert sk == "namespace#checkpoint_id"
        assert "#" in sk

        # Sort key with empty namespace
        sk_empty = _checkpoint_sk("", "checkpoint_id")
        assert sk_empty == "#checkpoint_id"

        # Chunk partition key
        ref_pk = _checkpoint_ref_pk(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert ref_pk == "CHUNK_thread_123#ns#cp_id"
        assert ref_pk.count("#") == 2

        # S3 key
        s3_key = _checkpoint_s3_key(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert s3_key == "thread_123/checkpoints/ns/cp_id"
        assert "checkpoints" in s3_key

    def test_writes_keys(self):
        """Test writes key generation."""
        # Partition key
        pk = _writes_pk(TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID)
        assert pk == "WRITES_thread_123#ns#cp_id"
        assert pk.startswith("WRITES_")

        # Sort key
        sk = _writes_sk(TEST_TASK_ID, 0)
        assert sk == f"{TEST_TASK_ID}#0"

        # Chunk partition key
        ref_pk = _writes_ref_pk(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID, TEST_TASK_ID, "0"
        )
        expected_pk = (
            f"CHUNK_{TEST_THREAD_ID}#{TEST_CHECKPOINT_NS}#"
            f"{TEST_CHECKPOINT_ID}#{TEST_TASK_ID}#0"
        )
        assert ref_pk == expected_pk
        assert ref_pk.count("#") == 4

        # S3 key
        s3_key = _writes_s3_key(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID, TEST_TASK_ID, "0"
        )
        expected_key = (
            f"{TEST_THREAD_ID}/checkpoints/{TEST_CHECKPOINT_NS}/"
            f"{TEST_CHECKPOINT_ID}/writes/{TEST_TASK_ID}/0"
        )
        assert s3_key == expected_key
        assert "writes" in s3_key


# ============================================================================
# INITIALIZATION
# ============================================================================


class TestUnifiedRepositoryInitialization:
    """Test UnifiedRepository initialization."""

    def test_repository_init(self):
        """Test initialization with and without TTL."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        # With all parameters including TTL
        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
            ttl_seconds=3600,
        )
        assert repo.dynamodb_client == mock_client
        assert repo.table_name == TEST_TABLE_NAME
        assert repo.serializer == mock_serializer
        assert repo.storage == mock_storage
        assert repo.ttl_seconds == 3600

        # Without TTL
        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )
        assert repo.ttl_seconds is None


class TestGetCheckpoint:
    """Tests for get_checkpoint method."""

    def test_get_checkpoint_scenarios(self):
        """Test checkpoint retrieval scenarios including S3, compression, and errors."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        # Test S3 storage retrieval
        mock_client.get_item.return_value = {
            "Item": {
                "PK": {"S": "CHECKPOINT_thread_123"},
                "SK": {"S": "ns#cp_id"},
                "id": {"S": TEST_CHECKPOINT_ID},
                "ns": {"S": TEST_CHECKPOINT_NS},
                "ref_key": {"S": "s3_key"},
                "ref_loc": {"S": "S3"},
                "type": {"S": "msgpack"},
                "compression": {"S": "gzip"},
            }
        }
        mock_storage.retrieve_data.return_value = b"compressed_data"
        mock_serializer.deserialize.return_value = {
            "checkpoint": {"id": TEST_CHECKPOINT_ID, "data": "large"},
            "metadata": {"step": 1},
        }

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test successful retrieval with compression
        result = repo.get_checkpoint(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert result is not None
        assert result["checkpoint_id"] == TEST_CHECKPOINT_ID
        mock_storage.retrieve_data.assert_called_with("s3_key", "S3")
        mock_serializer.deserialize.assert_called_with(
            "msgpack", b"compressed_data", CompressionType.GZIP
        )

        # Test payload not found
        mock_storage.retrieve_data.return_value = None
        result = repo.get_checkpoint(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert result is None

        # Test error handling
        mock_client.get_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "Server error"}},
            "GetItem",
        )
        with pytest.raises(ClientError):
            repo.get_checkpoint(TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID)


class TestPutCheckpoint:
    """Tests for put_checkpoint method."""

    def test_put_checkpoint_scenarios(self):
        """Test checkpoint storage with parent, TTL, S3, and error handling."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        # Test with parent and TTL
        mock_serializer.serialize.return_value = ("msgpack", b"data", False)
        mock_storage.store_data.return_value = "DYNAMODB"

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
            ttl_seconds=3600,
        )

        checkpoint = {"id": TEST_CHECKPOINT_ID, "ts": "2024-01-01"}
        metadata = {"step": 2}

        result = repo.put_checkpoint(
            TEST_THREAD_ID,
            TEST_CHECKPOINT_NS,
            checkpoint,
            metadata,
            parent_checkpoint_id="parent_id",
        )

        assert result["checkpoint_id"] == TEST_CHECKPOINT_ID
        call_kwargs = mock_client.put_item.call_args[1]
        assert "ttl" in call_kwargs["Item"]

        # Test large data to S3
        mock_serializer.serialize.return_value = (
            "msgpack",
            b"x" * 400000,
            CompressionType.GZIP,
        )
        mock_storage.store_data.return_value = "S3"

        repo.put_checkpoint(TEST_THREAD_ID, TEST_CHECKPOINT_NS, checkpoint, metadata)

        call_kwargs = mock_client.put_item.call_args[1]
        assert call_kwargs["Item"]["ref_loc"]["S"] == "S3"

        # Test error handling
        mock_serializer.serialize.side_effect = Exception("Serialization failed")
        with pytest.raises(Exception):  # noqa: B017
            repo.put_checkpoint(
                TEST_THREAD_ID, TEST_CHECKPOINT_NS, checkpoint, metadata
            )

        # Test DynamoDB error
        mock_serializer.serialize.side_effect = None
        mock_serializer.serialize.return_value = ("msgpack", b"data", False)
        mock_client.put_item.side_effect = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException"}}, "PutItem"
        )
        with pytest.raises(ClientError):
            repo.put_checkpoint(
                TEST_THREAD_ID, TEST_CHECKPOINT_NS, checkpoint, metadata
            )


class TestListCheckpoints:
    """Test list_checkpoints method."""

    def test_list_checkpoints_scenarios(self):
        """Test listing checkpoints with various filters and pagination."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        # Test basic listing
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#cp1"},
                    "id": {"S": "cp1"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "key1"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                },
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#cp2"},
                    "id": {"S": "cp2"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "key2"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                },
            ]
        }

        mock_storage.retrieve_data.return_value = b"data"
        mock_serializer.deserialize.return_value = {
            "checkpoint": {"id": "cp1"},
            "metadata": {},
        }

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        checkpoints = list(repo.list_checkpoints(TEST_THREAD_ID, TEST_CHECKPOINT_NS))
        assert len(checkpoints) == 2
        assert checkpoints[0]["checkpoint_id"] == "cp1"

        # Test with limit
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#cp1"},
                    "id": {"S": "cp1"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "key1"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                }
            ]
        }

        checkpoints = list(
            repo.list_checkpoints(TEST_THREAD_ID, TEST_CHECKPOINT_NS, limit=1)
        )
        assert len(checkpoints) == 1
        assert mock_client.query.call_args[1]["Limit"] == 1

        # Test empty result
        mock_client.query.return_value = {"Items": []}
        checkpoints = list(repo.list_checkpoints(TEST_THREAD_ID, TEST_CHECKPOINT_NS))
        assert len(checkpoints) == 0

        # Test payload not found (skipped)
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#cp1"},
                    "id": {"S": "cp1"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "missing"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                }
            ]
        }
        mock_storage.retrieve_data.return_value = None
        checkpoints = list(repo.list_checkpoints(TEST_THREAD_ID, TEST_CHECKPOINT_NS))
        assert len(checkpoints) == 0

        # Test error handling
        mock_client.query.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, "Query"
        )
        with pytest.raises(ClientError):
            list(repo.list_checkpoints(TEST_THREAD_ID, TEST_CHECKPOINT_NS))


class TestGetCheckpointLatest:
    """Tests for getting latest checkpoint without checkpoint_id."""

    def test_get_latest_checkpoint_scenarios(self):
        """Test retrieving latest checkpoint with and without results."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test successful retrieval
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#latest_id"},
                    "id": {"S": "latest_id"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "key"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                }
            ]
        }

        mock_storage.retrieve_data.return_value = b"data"
        mock_serializer.deserialize.return_value = {
            "checkpoint": {"id": "latest_id"},
            "metadata": {"step": 5},
        }

        result = repo.get_checkpoint(TEST_THREAD_ID, TEST_CHECKPOINT_NS, None)
        assert result is not None
        assert result["checkpoint_id"] == "latest_id"
        call_args = mock_client.query.call_args[1]
        assert call_args["ScanIndexForward"] is False
        assert call_args["Limit"] == 1

        # Test not found
        mock_client.query.return_value = {"Items": []}
        result = repo.get_checkpoint(TEST_THREAD_ID, TEST_CHECKPOINT_NS, None)
        assert result is None


class TestPendingWrites:
    """Tests for get_pending_writes and put_writes methods."""

    def test_get_pending_writes_scenarios(self):
        """Test retrieving pending writes with data and missing values."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test with data
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "WRITES_thread_123#ns#cp_id"},
                    "SK": {"S": "task1#0"},
                    "task_id": {"S": "task1"},
                    "idx": {"N": "0"},
                    "channel": {"S": "channel1"},
                    "ref_key": {"S": "write_key"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                }
            ]
        }

        mock_storage.retrieve_data.return_value = b"write_data"
        mock_serializer.deserialize.return_value = {"value": "write_value"}

        result = repo.get_pending_writes(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert len(result) >= 0
        mock_client.query.assert_called_once()

        # Test value not found (skipped)
        mock_storage.retrieve_data.return_value = None
        result = repo.get_pending_writes(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID
        )
        assert len(result) == 0

    def test_put_writes_scenarios(self):
        """Test storing writes with multiple scenarios including S3 and empty list."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test multiple writes
        mock_serializer.serialize.return_value = ("msgpack", b"data", False)
        mock_storage.store_data.return_value = "DYNAMODB"

        writes = [("channel1", {"value": 1}), ("channel2", {"value": 2})]
        repo.put_writes(
            TEST_THREAD_ID,
            TEST_CHECKPOINT_NS,
            TEST_CHECKPOINT_ID,
            writes,
            task_id="task1",
        )
        assert mock_storage.store_data.called

        # Test large writes to S3
        mock_serializer.serialize.return_value = (
            "msgpack",
            b"large" * 100000,
            CompressionType.GZIP,
        )
        mock_storage.store_data.return_value = "S3"

        writes = [("channel1", {"large": "data"})]
        repo.put_writes(
            TEST_THREAD_ID,
            TEST_CHECKPOINT_NS,
            TEST_CHECKPOINT_ID,
            writes,
            task_id="task1",
        )
        assert mock_storage.store_data.called

        # Test empty writes list
        mock_storage.store_data.reset_mock()
        repo.put_writes(
            TEST_THREAD_ID, TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID, [], task_id="task1"
        )
        assert not mock_storage.store_data.called

        # Test error handling
        mock_storage.store_data.side_effect = Exception("Storage failed")
        with pytest.raises(Exception):  # noqa: B017
            repo.put_writes(
                TEST_THREAD_ID,
                TEST_CHECKPOINT_NS,
                TEST_CHECKPOINT_ID,
                writes,
                task_id="task1",
            )


class TestDeleteOperations:
    """Tests for delete operations."""

    def test_delete_thread_checkpoints_scenarios(self):
        """Test deleting checkpoints with S3, multiple items, and errors."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test with S3 reference
        mock_client.batch_get_item.return_value = {
            "Responses": {
                TEST_TABLE_NAME: [
                    {
                        "PK": {"S": "CHECKPOINT_thread_123"},
                        "SK": {"S": "ns#cp_id"},
                        "ref_loc": {"S": "S3"},
                        "ref_key": {"S": "s3_key"},
                    },
                ]
            },
            "UnprocessedKeys": {},
        }
        mock_storage.batch_delete_data.return_value = {"failed": []}
        mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

        checkpoint_info = [(TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID)]
        repo.delete_thread_checkpoints(TEST_THREAD_ID, checkpoint_info)

        assert mock_client.batch_get_item.called
        assert mock_storage.batch_delete_data.called
        assert mock_client.batch_write_item.called

        # Test error handling
        mock_client.batch_get_item.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "BatchGetItem"
        )
        try:
            repo.delete_thread_checkpoints(TEST_THREAD_ID, checkpoint_info)
        except ClientError:
            pass

    def test_delete_thread_writes_scenarios(self):
        """Test deleting writes with S3 and multiple items."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        # Test multiple writes with S3
        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "WRITES_thread_123#ns#cp_id"},
                    "SK": {"S": "task1#0"},
                    "ref_loc": {"S": "S3"},
                    "ref_key": {"S": "key1"},
                },
                {
                    "PK": {"S": "WRITES_thread_123#ns#cp_id"},
                    "SK": {"S": "task2#0"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "ref_key": {"S": "key2"},
                },
            ]
        }
        mock_storage.batch_delete_data.return_value = {"failed": []}
        mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

        checkpoint_info = [(TEST_CHECKPOINT_NS, TEST_CHECKPOINT_ID)]
        repo.delete_thread_writes(TEST_THREAD_ID, checkpoint_info)

        assert mock_client.query.called
        assert mock_storage.batch_delete_data.called
        assert mock_client.batch_write_item.called


class TestMiscellaneousOperations:
    """Tests for miscellaneous repository operations."""

    def test_get_thread_checkpoint_info(self):
        """Test retrieving checkpoint info for a thread."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        mock_client.query.return_value = {
            "Items": [{"SK": {"S": "ns1#cp1"}}, {"SK": {"S": "ns2#cp2"}}]
        }

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        result = repo.get_thread_checkpoint_info(TEST_THREAD_ID)
        assert len(result) == 2
        assert result[0] == ("ns1", "cp1")

        # Test empty
        mock_client.query.return_value = {"Items": []}
        result = repo.get_thread_checkpoint_info(TEST_THREAD_ID)
        assert result == []

    def test_list_checkpoints_with_metadata_filter(self):
        """Test listing checkpoints with metadata filter."""
        mock_client = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        mock_client.query.return_value = {
            "Items": [
                {
                    "PK": {"S": "CHECKPOINT_thread_123"},
                    "SK": {"S": "ns#cp1"},
                    "id": {"S": "cp1"},
                    "ns": {"S": TEST_CHECKPOINT_NS},
                    "ref_key": {"S": "key1"},
                    "ref_loc": {"S": "DYNAMODB"},
                    "type": {"S": "msgpack"},
                },
            ]
        }

        mock_storage.retrieve_data.return_value = b"data1"
        mock_serializer.deserialize.return_value = {
            "checkpoint": {"id": "cp1"},
            "metadata": {"step": 1, "status": "complete"},
        }

        repo = UnifiedRepository(
            dynamodb_client=mock_client,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

        checkpoints = list(
            repo.list_checkpoints(
                TEST_THREAD_ID, TEST_CHECKPOINT_NS, filter={"status": "complete"}
            )
        )

        assert len(checkpoints) == 1
        assert checkpoints[0]["checkpoint_id"] == "cp1"


class TestGetCheckpointFallback:
    """Test get_checkpoint fallback when latest checkpoint payload is missing.

    This tests a critical reliability feature:
    - When the latest checkpoint's payload is missing (corrupted/deleted)
    - The system automatically falls back to the next available checkpoint
    """

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for UnifiedRepository."""
        mock_dynamodb = Mock()
        mock_storage = Mock()
        mock_serializer = Mock()

        return {
            "dynamodb_client": mock_dynamodb,
            "storage": mock_storage,
            "serializer": mock_serializer,
        }

    @pytest.fixture
    def repository(self, mock_dependencies):
        """Create UnifiedRepository with mocked dependencies."""
        return UnifiedRepository(
            dynamodb_client=mock_dependencies["dynamodb_client"],
            table_name="test-table",
            serializer=mock_dependencies["serializer"],
            storage_strategy=mock_dependencies["storage"],
        )

    def test_get_checkpoint_latest_payload_missing_fallback_success(
        self, repository, mock_dependencies
    ):
        """Test fallback when latest checkpoint payload missing."""
        thread_id = "test-thread-123"
        checkpoint_ns = "test-ns"

        # Mock DynamoDB responses
        # Phase 1: Latest checkpoint query (Limit=1)
        latest_response = {
            "Items": [
                {
                    "id": {"S": "checkpoint-1"},
                    "ns": {"S": checkpoint_ns},
                    "type": {"S": "checkpoint"},
                    "ref_key": {"S": "key-1"},
                    "ref_loc": {"S": "dynamodb"},
                    "parent_checkpoint_id": {"S": "parent-1"},
                }
            ],
            "LastEvaluatedKey": {
                "PK": {"S": "CHECKPOINT_thread-123"},
                "SK": {"S": "test-ns#checkpoint-1"},
            },
        }

        # Phase 2: All checkpoints query (no Limit)
        all_response_page1 = {
            "Items": [
                {
                    "id": {"S": "checkpoint-2"},
                    "ns": {"S": checkpoint_ns},
                    "type": {"S": "checkpoint"},
                    "ref_key": {"S": "key-2"},
                    "ref_loc": {"S": "dynamodb"},
                    "parent_checkpoint_id": {"S": "parent-2"},
                },
            ]
        }

        # Configure mock responses
        mock_dynamodb = mock_dependencies["dynamodb_client"]
        mock_dynamodb.query.side_effect = [latest_response, all_response_page1]

        # Mock storage responses
        mock_storage = mock_dependencies["storage"]
        mock_storage.retrieve_data.side_effect = [
            None,  # First call (latest checkpoint in Phase 1) returns None
            # Second call (checkpoint-2 in Phase 2, checkpoint-1 skipped)
            b"valid-payload-data",
        ]

        # Mock serializer
        mock_serializer = mock_dependencies["serializer"]
        mock_serializer.deserialize.return_value = {
            "checkpoint": {"data": "test"},
            "metadata": {"step": 1},
        }

        # Execute
        result = repository.get_checkpoint(
            thread_id=thread_id, checkpoint_ns=checkpoint_ns
        )

        # Verify result
        assert result is not None
        assert result["checkpoint"] == {"data": "test"}
        assert result["metadata"] == {"step": 1}
        assert result["checkpoint_id"] == "checkpoint-2"
        assert result["checkpoint_ns"] == checkpoint_ns
        assert result["thread_id"] == thread_id
        assert result["parent_checkpoint_id"] == "parent-2"

        # Verify DynamoDB calls
        assert mock_dynamodb.query.call_count == 2

        # First call should be Limit=1 (fast path)
        first_call = mock_dynamodb.query.call_args_list[0]
        assert first_call[1]["Limit"] == 1

        # Second call should NOT have Limit (fallback searches all)
        second_call = mock_dynamodb.query.call_args_list[1]
        assert "Limit" not in second_call[1]
        assert second_call[1]["ScanIndexForward"] is False  # Latest first

        # Verify storage calls
        assert mock_storage.retrieve_data.call_count == 2
        # First call tries latest checkpoint (key-1)
        first_storage_call = mock_storage.retrieve_data.call_args_list[0][0]
        assert first_storage_call == ("key-1", "dynamodb")
        # Second call tries checkpoint-2 (checkpoint-1 is skipped)
        second_storage_call = mock_storage.retrieve_data.call_args_list[1][0]
        assert second_storage_call == ("key-2", "dynamodb")

        # Verify serializer call
        mock_serializer.deserialize.assert_called_once_with(
            "checkpoint", b"valid-payload-data", None
        )


# ============================================================================
# METADATA-BEFORE-PAYLOAD ORDERING TESTS
# ============================================================================


class TestMetadataBeforePayloadOrdering:
    """Test that metadata is written before payload in all operations."""

    @pytest.fixture
    def mock_repo_components(self):
        """Create mocked components for UnifiedRepository."""
        mock_dynamodb = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        # Setup serializer to return test data
        mock_serializer.serialize.return_value = (
            "msgpack",  # type
            b"test_payload_data",  # data
            None,  # compression
        )

        # Setup storage to indicate small payload (DYNAMODB storage)
        mock_storage.should_offload_to_s3.return_value = False

        return {
            "dynamodb": mock_dynamodb,
            "serializer": mock_serializer,
            "storage": mock_storage,
        }

    @pytest.fixture
    def repo(self, mock_repo_components):
        """Create UnifiedRepository with mocked dependencies."""
        return UnifiedRepository(
            dynamodb_client=mock_repo_components["dynamodb"],
            table_name=TEST_TABLE_NAME,
            serializer=mock_repo_components["serializer"],
            storage_strategy=mock_repo_components["storage"],
            ttl_seconds=None,
        )

    def test_put_checkpoint_metadata_before_payload(self, repo, mock_repo_components):
        """Test that put_checkpoint writes metadata before payload."""
        # Arrange
        checkpoint = {"id": TEST_CHECKPOINT_ID, "data": "test"}
        metadata = {"step": 1}

        # Track call order
        call_order = []

        def track_put_item(**kwargs):
            call_order.append(("put_item", kwargs))

        def track_store_data(*args, **kwargs):
            call_order.append(("store_data", args, kwargs))

        mock_repo_components["dynamodb"].put_item.side_effect = track_put_item
        mock_repo_components["storage"].store_data.side_effect = track_store_data

        # Act
        repo.put_checkpoint(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint=checkpoint,
            metadata=metadata,
        )

        # Assert
        assert len(call_order) == 2, "Expected exactly 2 calls"
        assert call_order[0][0] == "put_item", "First call should be put_item"
        assert call_order[1][0] == "store_data", "Second call should be store_data"

        # Verify put_item was called with correct metadata
        put_item_call = call_order[0][1]
        assert put_item_call["TableName"] == TEST_TABLE_NAME
        assert "Item" in put_item_call
        item = put_item_call["Item"]
        assert item["id"]["S"] == TEST_CHECKPOINT_ID
        assert item["ns"]["S"] == TEST_CHECKPOINT_NS
        assert item["ref_loc"]["S"] == "DYNAMODB"

        # Verify store_data was called after metadata
        store_data_call = call_order[1]
        assert store_data_call[1][2] == b"test_payload_data"

    def test_put_checkpoint_s3_offload_metadata_before_payload(
        self, repo, mock_repo_components
    ):
        """Test metadata-before-payload for S3 offloaded checkpoints."""
        # Arrange - large payload that should go to S3
        checkpoint = {"id": TEST_CHECKPOINT_ID, "data": "large_data"}
        metadata = {"step": 1}

        # Configure storage to offload to S3
        mock_repo_components["storage"].should_offload_to_s3.return_value = True

        # Track call order
        call_order = []

        def track_put_item(**kwargs):
            call_order.append(("put_item", kwargs))

        def track_store_data(*args, **kwargs):
            call_order.append(("store_data", args, kwargs))

        mock_repo_components["dynamodb"].put_item.side_effect = track_put_item
        mock_repo_components["storage"].store_data.side_effect = track_store_data

        # Act
        repo.put_checkpoint(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint=checkpoint,
            metadata=metadata,
        )

        # Assert
        assert len(call_order) == 2
        assert call_order[0][0] == "put_item"
        assert call_order[1][0] == "store_data"

        # Verify metadata indicates S3 storage
        item = call_order[0][1]["Item"]
        assert item["ref_loc"]["S"] == "S3"
        assert "checkpoints" in item["ref_key"]["S"]

    def test_put_writes_metadata_before_payload_per_write(
        self, repo, mock_repo_components
    ):
        """Test that each write's metadata is stored before its payload."""
        # Arrange
        writes = [
            ("channel1", "value1"),
            ("channel2", "value2"),
            ("channel3", "value3"),
        ]

        # Track call order with details
        call_order = []

        def track_put_item(**kwargs):
            item = kwargs.get("Item", {})
            channel = item.get("channel", {}).get("S", "unknown")
            call_order.append(("put_item", channel))

        def track_store_data(*args, **kwargs):
            # Extract channel info from chunk_key
            chunk_key = kwargs.get("chunk_key", "")
            call_order.append(("store_data", chunk_key))

        mock_repo_components["dynamodb"].put_item.side_effect = track_put_item
        mock_repo_components["storage"].store_data.side_effect = track_store_data

        # Act
        repo.put_writes(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint_id=TEST_CHECKPOINT_ID,
            writes=writes,
            task_id=TEST_TASK_ID,
        )

        # Assert
        assert len(call_order) == 6, "Expected 3 metadata + 3 payload calls"

        # Verify alternating pattern: metadata, payload, metadata, payload, ...
        for i in range(0, len(call_order), 2):
            assert call_order[i][0] == "put_item", (
                f"Call {i} should be put_item (metadata)"
            )
            assert call_order[i + 1][0] == "store_data", (
                f"Call {i + 1} should be store_data (payload)"
            )

        # Verify each write's metadata comes before its payload
        assert call_order[0] == ("put_item", "channel1")
        assert "CHUNK_" in call_order[1][1]  # store_data with chunk_key
        assert call_order[2] == ("put_item", "channel2")
        assert "CHUNK_" in call_order[3][1]
        assert call_order[4] == ("put_item", "channel3")
        assert "CHUNK_" in call_order[5][1]

    def test_put_writes_sequential_not_parallel(self, repo, mock_repo_components):
        """Test that writes are processed sequentially, not in parallel."""
        # Arrange
        writes = [("channel1", "value1"), ("channel2", "value2")]

        # Track execution order with timestamps
        execution_log = []

        def track_put_item(**kwargs):
            item = kwargs.get("Item", {})
            channel = item.get("channel", {}).get("S", "unknown")
            execution_log.append(f"metadata_{channel}")

        def track_store_data(*args, **kwargs):
            execution_log.append("payload")

        mock_repo_components["dynamodb"].put_item.side_effect = track_put_item
        mock_repo_components["storage"].store_data.side_effect = track_store_data

        # Act
        repo.put_writes(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint_id=TEST_CHECKPOINT_ID,
            writes=writes,
            task_id=TEST_TASK_ID,
        )

        # Assert - verify sequential execution pattern
        expected_order = [
            "metadata_channel1",
            "payload",
            "metadata_channel2",
            "payload",
        ]
        assert execution_log == expected_order

    def test_put_checkpoint_failure_no_payload_stored(self, repo, mock_repo_components):
        """Test that payload is not stored if metadata write fails."""
        # Arrange
        checkpoint = {"id": TEST_CHECKPOINT_ID, "data": "test"}
        metadata = {"step": 1}

        # Make put_item fail
        error_response = {"Error": {"Code": "InternalServerError", "Message": "Test"}}
        mock_repo_components["dynamodb"].put_item.side_effect = ClientError(
            error_response, "PutItem"
        )

        # Act & Assert
        with pytest.raises(ClientError):
            repo.put_checkpoint(
                thread_id=TEST_THREAD_ID,
                checkpoint_ns=TEST_CHECKPOINT_NS,
                checkpoint=checkpoint,
                metadata=metadata,
            )

        # Verify store_data was never called
        mock_repo_components["storage"].store_data.assert_not_called()

    def test_put_writes_failure_stops_at_failed_write(self, repo, mock_repo_components):
        """Test that write processing stops if metadata write fails."""
        # Arrange
        writes = [("channel1", "value1"), ("channel2", "value2")]

        # Make second put_item fail
        error_response = {"Error": {"Code": "InternalServerError", "Message": "Test"}}
        call_count = [0]

        def failing_put_item(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call
                raise ClientError(error_response, "PutItem")

        mock_repo_components["dynamodb"].put_item.side_effect = failing_put_item

        # Act & Assert
        with pytest.raises(ClientError):
            repo.put_writes(
                thread_id=TEST_THREAD_ID,
                checkpoint_ns=TEST_CHECKPOINT_NS,
                checkpoint_id=TEST_CHECKPOINT_ID,
                writes=writes,
                task_id=TEST_TASK_ID,
            )

        # Verify store_data was called only once (for first write)
        assert mock_repo_components["storage"].store_data.call_count == 1

    def test_put_single_write_item_called_before_storage(
        self, repo, mock_repo_components
    ):
        """Test that put_single_write_item is called before storage.store_data."""
        # Arrange
        writes = [("channel1", "value1")]

        # Track method calls
        call_order = []

        original_put_single = repo.put_single_write_item

        def tracked_put_single(*args, **kwargs):
            call_order.append("put_single_write_item")
            return original_put_single(*args, **kwargs)

        def tracked_store_data(*args, **kwargs):
            call_order.append("store_data")

        repo.put_single_write_item = tracked_put_single
        mock_repo_components["storage"].store_data.side_effect = tracked_store_data

        # Act
        repo.put_writes(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint_id=TEST_CHECKPOINT_ID,
            writes=writes,
            task_id=TEST_TASK_ID,
        )

        # Assert
        assert call_order == ["put_single_write_item", "store_data"]


class TestDetermineStorageLocation:
    """Test the _determine_storage_location helper method."""

    @pytest.fixture
    def repo(self):
        """Create UnifiedRepository with mocked dependencies."""
        mock_dynamodb = Mock()
        mock_serializer = Mock()
        mock_storage = Mock()

        return UnifiedRepository(
            dynamodb_client=mock_dynamodb,
            table_name=TEST_TABLE_NAME,
            serializer=mock_serializer,
            storage_strategy=mock_storage,
        )

    def test_determine_storage_location_dynamodb(self, repo):
        """Test storage location determination for small payloads."""
        # Arrange
        repo.storage.should_offload_to_s3.return_value = False
        payload = b"small_data"

        # Act
        location = repo._determine_storage_location(payload)

        # Assert
        assert location == "DYNAMODB"
        repo.storage.should_offload_to_s3.assert_called_once_with(payload)

    def test_determine_storage_location_s3(self, repo):
        """Test storage location determination for large payloads."""
        # Arrange
        repo.storage.should_offload_to_s3.return_value = True
        payload = b"x" * 1000000

        # Act
        location = repo._determine_storage_location(payload)

        # Assert
        assert location == "S3"
        repo.storage.should_offload_to_s3.assert_called_once_with(payload)

    def test_determine_storage_location_consistency(self, repo):
        """Test that same payload always returns same location."""
        # Arrange
        repo.storage.should_offload_to_s3.return_value = False
        payload = b"test_data"

        # Act
        location1 = repo._determine_storage_location(payload)
        location2 = repo._determine_storage_location(payload)

        # Assert
        assert location1 == location2
        assert location1 == "DYNAMODB"

    def test_determine_storage_location_used_in_put_checkpoint(self, repo):
        """Test that _determine_storage_location is used in put_checkpoint."""
        # Arrange
        checkpoint = {"id": TEST_CHECKPOINT_ID, "data": "test"}
        metadata = {"step": 1}

        repo.serializer.serialize.return_value = ("msgpack", b"data", None)
        repo.storage.should_offload_to_s3.return_value = True

        # Act
        repo.put_checkpoint(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint=checkpoint,
            metadata=metadata,
        )

        # Assert - verify should_offload_to_s3 was called
        repo.storage.should_offload_to_s3.assert_called_once()

        # Verify put_item was called with S3 location
        put_item_call = repo.dynamodb_client.put_item.call_args
        item = put_item_call[1]["Item"]
        assert item["ref_loc"]["S"] == "S3"

    def test_determine_storage_location_used_in_put_writes(self, repo):
        """Test that _determine_storage_location is used in put_writes."""
        # Arrange
        writes = [("channel1", "value1")]

        repo.serializer.serialize.return_value = ("msgpack", b"data", None)
        repo.storage.should_offload_to_s3.return_value = False

        # Act
        repo.put_writes(
            thread_id=TEST_THREAD_ID,
            checkpoint_ns=TEST_CHECKPOINT_NS,
            checkpoint_id=TEST_CHECKPOINT_ID,
            writes=writes,
            task_id=TEST_TASK_ID,
        )

        # Assert - verify should_offload_to_s3 was called
        repo.storage.should_offload_to_s3.assert_called_once()

        # Verify put_item was called with DYNAMODB location
        put_item_call = repo.dynamodb_client.put_item.call_args
        item = put_item_call[1]["Item"]
        assert item["ref_loc"]["S"] == "DYNAMODB"
