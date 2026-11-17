"""Comprehensive unit tests for DynamoDBSaver."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from botocore.config import Config
from conftest import (
    TEST_CHANNEL,
    TEST_CHECKPOINT_ID,
    TEST_CHECKPOINT_NS,
    TEST_S3_BUCKET,
    TEST_TABLE_NAME,
    TEST_TASK_ID,
    TEST_THREAD_ID,
    TEST_TTL_SECONDS,
    build_checkpoint_item,
    build_write_item,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_aws import DynamoDBSaver
from langgraph_checkpoint_aws.checkpoint.dynamodb.serialization import (
    CheckpointSerializer,
)


class TestDynamoDBSaverInitialization:
    """Test DynamoDBSaver initialization."""

    def test_init_basic_and_compression(self, mock_saver_dependencies):
        """Test initialization with minimal parameters and compression."""
        mock_ddb = mock_saver_dependencies["ddb_instance"]

        # Minimal parameters - compression disabled by default
        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)
        assert saver.client == mock_ddb
        call_args = mock_saver_dependencies["serializer"].call_args[0]
        assert call_args[1] is False  # compression disabled

        # With compression enabled
        DynamoDBSaver(table_name=TEST_TABLE_NAME, enable_checkpoint_compression=True)
        call_args = mock_saver_dependencies["serializer"].call_args[0]
        assert call_args[1] is True  # compression enabled

    def test_init_with_ttl_and_s3(self, mock_saver_dependencies):
        """Test initialization with TTL and S3 offload configuration."""
        mock_s3 = Mock()
        mock_saver_dependencies["create_s3_client"].return_value = mock_s3

        # With TTL and S3
        s3_config = {"bucket_name": TEST_S3_BUCKET, "region_name": "us-west-2"}
        DynamoDBSaver(
            table_name=TEST_TABLE_NAME,
            ttl_seconds=TEST_TTL_SECONDS,
            s3_offload_config=s3_config,
        )

        # Verify TTL
        call_kwargs = mock_saver_dependencies["storage"].call_args[1]
        assert call_kwargs["ttl_seconds"] == TEST_TTL_SECONDS

        # Verify S3
        assert call_kwargs["s3_client"] == mock_s3
        assert call_kwargs["s3_bucket"] == TEST_S3_BUCKET

    def test_init_with_s3_key_prefix(self, mock_saver_dependencies):
        """Test initialization passes key_prefix to StorageStrategy."""
        mock_s3 = Mock()
        mock_saver_dependencies["create_s3_client"].return_value = mock_s3

        # With key_prefix
        DynamoDBSaver(
            table_name=TEST_TABLE_NAME,
            s3_offload_config={
                "bucket_name": TEST_S3_BUCKET,
                "key_prefix": "my-app/langgraph",
            },
        )

        call_kwargs = mock_saver_dependencies["storage"].call_args[1]
        assert call_kwargs["s3_key_prefix"] == "my-app/langgraph"

        # Without key_prefix
        DynamoDBSaver(
            table_name=TEST_TABLE_NAME,
            s3_offload_config={"bucket_name": TEST_S3_BUCKET},
        )

        call_kwargs = mock_saver_dependencies["storage"].call_args[1]
        assert call_kwargs["s3_key_prefix"] is None

    def test_init_with_all_aws_parameters(self, mock_saver_dependencies):
        """Test initialization with all AWS configuration parameters."""
        mock_s3 = Mock()
        mock_saver_dependencies["create_s3_client"].return_value = mock_s3
        boto_config = Config(region_name="us-west-2")

        # Test with all parameters including different endpoints for DynamoDB and S3
        DynamoDBSaver(
            table_name=TEST_TABLE_NAME,
            region_name="us-west-2",
            endpoint_url="http://dynamodb.local:8000",
            boto_config=boto_config,
            s3_offload_config={
                "bucket_name": TEST_S3_BUCKET,
                "endpoint_url": "http://s3.local:9000",
            },
        )

        # Verify DynamoDB parameters
        dynamodb_kwargs = mock_saver_dependencies["create_dynamodb_client"].call_args[1]
        assert dynamodb_kwargs["region_name"] == "us-west-2"
        assert dynamodb_kwargs["endpoint_url"] == "http://dynamodb.local:8000"
        assert dynamodb_kwargs["boto_config"] == boto_config

        # Verify S3 has different endpoint
        s3_kwargs = mock_saver_dependencies["create_s3_client"].call_args[1]
        assert s3_kwargs["endpoint_url"] == "http://s3.local:9000"

    def test_init_consistent_read(
        self, mock_saver_dependencies: dict[str, Mock]
    ) -> None:
        """Test consistent_read defaults to False and reaches storage and repo."""
        storage_cls = mock_saver_dependencies["storage"]
        repo_cls = mock_saver_dependencies["repo"]

        DynamoDBSaver(table_name=TEST_TABLE_NAME)
        assert storage_cls.call_args[1]["consistent_read"] is False
        assert repo_cls.call_args[1]["consistent_read"] is False

        DynamoDBSaver(table_name=TEST_TABLE_NAME, consistent_read=True)
        assert storage_cls.call_args[1]["consistent_read"] is True
        assert repo_cls.call_args[1]["consistent_read"] is True


class TestDynamoDBSaverGetTuple:
    """Tests for get_tuple method."""

    def test_get_tuple_found_and_not_found(self, mock_saver_dependencies):
        """Test get_tuple when checkpoint is found and not found."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo
        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        # Test found scenario
        mock_repo.get_checkpoint.return_value = {
            "checkpoint_id": "cp_id",
            "checkpoint_ns": "ns",
            "checkpoint": {"id": "cp_id", "data": "test"},
            "metadata": {"step": 1},
            "parent_checkpoint_id": "parent_id",
        }
        mock_repo.get_pending_writes.return_value = [
            ("task1", "channel1", {"value": 1})
        ]

        config = {
            "configurable": {
                "thread_id": "thread_123",
                "checkpoint_ns": "ns",
                "checkpoint_id": "cp_id",
            }
        }
        result = saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "cp_id"
        assert result.metadata["step"] == 1
        assert result.parent_config is not None
        assert len(result.pending_writes) == 1

        # Test not found scenario
        mock_repo.get_checkpoint.return_value = None
        result = saver.get_tuple(config)
        assert result is None


class TestDynamoDBSaverPut:
    """Tests for put method."""

    def test_put_checkpoint(self, mock_saver_dependencies):
        """Test putting a checkpoint."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo

        mock_repo.put_checkpoint.return_value = {
            "thread_id": TEST_THREAD_ID,
            "checkpoint_ns": TEST_CHECKPOINT_NS,
            "checkpoint_id": "new_cp_id",
        }

        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
                "checkpoint_id": "parent_id",
            },
            "metadata": {"extra": "data"},
        }

        checkpoint = {"id": "new_cp_id", "data": "test"}
        metadata = {"step": 2}
        new_versions = {}

        result = saver.put(config, checkpoint, metadata, new_versions)

        assert result["configurable"]["checkpoint_id"] == "new_cp_id"
        assert result["configurable"]["thread_id"] == TEST_THREAD_ID

        # Verify metadata was merged
        call_kwargs = mock_repo.put_checkpoint.call_args[1]
        assert call_kwargs["metadata"]["step"] == 2
        assert call_kwargs["metadata"]["extra"] == "data"


class TestDynamoDBSaverPutWrites:
    """Tests for put_writes method."""

    def test_put_writes(self, mock_saver_dependencies):
        """Test putting writes."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo

        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
                "checkpoint_id": "cp_id",
            }
        }

        writes = [(TEST_CHANNEL, {"value": 1}), ("channel2", {"value": 2})]
        task_id = TEST_TASK_ID

        saver.put_writes(config, writes, task_id)

        # Verify repo.put_writes was called
        mock_repo.put_writes.assert_called_once()
        call_kwargs = mock_repo.put_writes.call_args[1]
        assert call_kwargs["thread_id"] == TEST_THREAD_ID
        assert call_kwargs["checkpoint_ns"] == TEST_CHECKPOINT_NS
        assert call_kwargs["checkpoint_id"] == "cp_id"
        assert call_kwargs["task_id"] == TEST_TASK_ID


class TestDynamoDBSaverList:
    """Tests for list method."""

    def test_list_checkpoints(self, mock_saver_dependencies):
        """Test listing checkpoints."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo

        # Mock list_checkpoints to return iterator
        mock_repo.list_checkpoints.return_value = iter(
            [
                {
                    "thread_id": TEST_THREAD_ID,
                    "checkpoint_id": "cp1",
                    "checkpoint_ns": TEST_CHECKPOINT_NS,
                    "checkpoint": {"id": "cp1"},
                    "metadata": {"step": 1},
                    "parent_checkpoint_id": None,
                },
                {
                    "thread_id": TEST_THREAD_ID,
                    "checkpoint_id": "cp2",
                    "checkpoint_ns": TEST_CHECKPOINT_NS,
                    "checkpoint": {"id": "cp2"},
                    "metadata": {"step": 2},
                    "parent_checkpoint_id": "cp1",
                },
            ]
        )

        # Mock get_pending_writes
        mock_repo.get_pending_writes.return_value = []

        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
            }
        }

        result = list(saver.list(config))

        assert len(result) == 2
        assert result[0].checkpoint["id"] == "cp1"
        assert result[1].checkpoint["id"] == "cp2"
        assert result[1].parent_config is not None

    def test_list_raises_for_config_without_thread_id(self, mock_saver_dependencies):
        """Test listing rejects malformed configs without thread_id."""
        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        with pytest.raises(ValueError, match="Runnable config must contain thread_id"):
            list(saver.list({"configurable": {}}))


class TestDynamoDBSaverAsyncMethods:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_async_operations(self, mock_saver_dependencies):
        """Test async get_tuple, put, and put_writes."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo
        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        # Test aget_tuple
        mock_repo.get_checkpoint.return_value = {
            "checkpoint_id": TEST_CHECKPOINT_ID,
            "checkpoint_ns": TEST_CHECKPOINT_NS,
            "checkpoint": {"id": TEST_CHECKPOINT_ID},
            "metadata": {},
            "parent_checkpoint_id": None,
        }
        mock_repo.get_pending_writes.return_value = []
        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
                "checkpoint_id": TEST_CHECKPOINT_ID,
            }
        }
        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == TEST_CHECKPOINT_ID

        # Test aput
        mock_repo.put_checkpoint.return_value = {
            "thread_id": TEST_THREAD_ID,
            "checkpoint_ns": TEST_CHECKPOINT_NS,
            "checkpoint_id": TEST_CHECKPOINT_ID,
        }
        result = await saver.aput(config, {"id": TEST_CHECKPOINT_ID}, {}, {})
        assert result["configurable"]["checkpoint_id"] == TEST_CHECKPOINT_ID

        # Test aput_writes
        await saver.aput_writes(config, [(TEST_CHANNEL, {"value": 1})], TEST_TASK_ID)
        assert mock_repo.put_writes.called


class TestDynamoDBSaverDeleteThread:
    """Tests for delete_thread method."""

    @pytest.mark.asyncio
    async def test_delete_thread_sync_and_async(self, mock_saver_dependencies):
        """Test deleting a thread with sync and async methods."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo
        mock_repo.get_thread_checkpoint_info.return_value = [
            (TEST_CHECKPOINT_NS, "cp1"),
            (TEST_CHECKPOINT_NS, "cp2"),
        ]
        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        # Test sync delete
        saver.delete_thread(TEST_THREAD_ID)
        mock_repo.delete_thread_writes.assert_called_with(
            TEST_THREAD_ID, [(TEST_CHECKPOINT_NS, "cp1"), (TEST_CHECKPOINT_NS, "cp2")]
        )
        mock_repo.delete_thread_checkpoints.assert_called_with(
            TEST_THREAD_ID, [(TEST_CHECKPOINT_NS, "cp1"), (TEST_CHECKPOINT_NS, "cp2")]
        )

        # Test async delete
        await saver.adelete_thread(TEST_THREAD_ID)
        assert mock_repo.delete_thread_writes.call_count == 2
        assert mock_repo.delete_thread_checkpoints.call_count == 2


class TestDynamoDBSaverAsyncList:
    """Tests for async list method."""

    @pytest.mark.asyncio
    async def test_alist_checkpoints(self, mock_saver_dependencies):
        """Test async list checkpoints."""
        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo

        mock_repo.list_checkpoints.return_value = iter(
            [
                {
                    "thread_id": TEST_THREAD_ID,
                    "checkpoint_id": "cp1",
                    "checkpoint_ns": TEST_CHECKPOINT_NS,
                    "checkpoint": {"id": "cp1"},
                    "metadata": {},
                    "parent_checkpoint_id": None,
                }
            ]
        )
        mock_repo.get_pending_writes.return_value = []

        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)

        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
            }
        }

        result = []
        async for item in saver.alist(config):
            result.append(item)

        assert len(result) == 1
        assert result[0].checkpoint["id"] == "cp1"

    @pytest.mark.asyncio
    async def test_alist_runs_blocking_work_off_event_loop(
        self, mock_saver_dependencies
    ):
        """`alist` must not run blocking boto3 calls on the event loop thread."""
        import threading

        loop_thread_id = threading.get_ident()
        observed_thread_ids = []

        def list_checkpoints(*args, **kwargs):
            # Record the thread each item is produced on. With a blocking
            # implementation this would run on the event loop thread.
            for checkpoint_id in ("cp1", "cp2"):
                observed_thread_ids.append(threading.get_ident())
                yield {
                    "thread_id": TEST_THREAD_ID,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": TEST_CHECKPOINT_NS,
                    "checkpoint": {"id": checkpoint_id},
                    "metadata": {},
                    "parent_checkpoint_id": None,
                }

        mock_repo = Mock()
        mock_saver_dependencies["repo"].return_value = mock_repo
        mock_repo.list_checkpoints.side_effect = list_checkpoints
        mock_repo.get_pending_writes.return_value = []

        saver = DynamoDBSaver(table_name=TEST_TABLE_NAME)
        config = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
            }
        }

        result = [item async for item in saver.alist(config)]

        assert [item.checkpoint["id"] for item in result] == ["cp1", "cp2"]
        assert observed_thread_ids, "list_checkpoints was never driven"
        assert all(tid != loop_thread_id for tid in observed_thread_ids)


def _build_fake_dynamodb_client() -> Mock:
    """Build a mock DynamoDB client serving one checkpoint with one pending write.

    Payloads are stored as DynamoDB chunks so every read path issues real
    ``get_item``/``query``/``batch_get_item`` calls through the saver's
    repository and storage strategy.
    """
    serializer = CheckpointSerializer(JsonPlusSerializer())
    checkpoint_type, checkpoint_payload, _ = serializer.serialize(
        {"checkpoint": empty_checkpoint(), "metadata": {"step": 1}}
    )
    write_type, write_payload, _ = serializer.serialize({"value": 1})

    checkpoint_item = {**build_checkpoint_item(), "type": {"S": checkpoint_type}}
    write_item = {**build_write_item(), "type": {"S": write_type}}
    write_ref_key = write_item["ref_key"]["S"]

    def get_item(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key["SK"]["S"] != "CHUNK":
            return {"Item": checkpoint_item}
        is_write = key["PK"]["S"] == write_ref_key
        payload = write_payload if is_write else checkpoint_payload
        return {"Item": {"payload": {"B": payload}}}

    def query(**kwargs: Any) -> dict[str, Any]:
        pk = kwargs["ExpressionAttributeValues"][":pk"]["S"]
        return {
            "Items": [write_item] if pk.startswith("WRITES_") else [checkpoint_item]
        }

    client = Mock()
    client.get_item.side_effect = get_item
    client.query.side_effect = query
    client.batch_get_item.return_value = {
        "Responses": {TEST_TABLE_NAME: [checkpoint_item]},
        "UnprocessedKeys": {},
    }
    client.batch_write_item.return_value = {"UnprocessedItems": {}}
    return client


class TestDynamoDBSaverConsistentRead:
    """Tests that every DynamoDB read honors the consistent_read setting."""

    @pytest.mark.parametrize("consistent_read", [False, True])
    def test_every_read_uses_configured_consistency(
        self, consistent_read: bool
    ) -> None:
        """Test get_tuple, list, and delete_thread pass ConsistentRead on reads."""
        mock_client = _build_fake_dynamodb_client()
        with patch(
            "langgraph_checkpoint_aws.checkpoint.dynamodb.saver.create_dynamodb_client",
            return_value=mock_client,
        ):
            saver = DynamoDBSaver(
                table_name=TEST_TABLE_NAME, consistent_read=consistent_read
            )

        latest_config: RunnableConfig = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
            }
        }
        by_id_config: RunnableConfig = {
            "configurable": {
                "thread_id": TEST_THREAD_ID,
                "checkpoint_ns": TEST_CHECKPOINT_NS,
                "checkpoint_id": TEST_CHECKPOINT_ID,
            }
        }

        latest = saver.get_tuple(latest_config)
        by_id = saver.get_tuple(by_id_config)
        listed = list(saver.list(latest_config))
        saver.delete_thread(TEST_THREAD_ID)

        # Confirm the payload chunk and pending-write read paths actually ran
        for checkpoint_tuple in [latest, by_id, *listed]:
            assert checkpoint_tuple is not None
            assert checkpoint_tuple.pending_writes == [
                (TEST_TASK_ID, TEST_CHANNEL, {"value": 1})
            ]
        assert mock_client.batch_get_item.called

        single_reads = (
            mock_client.get_item.call_args_list + mock_client.query.call_args_list
        )
        assert single_reads
        for read_call in single_reads:
            assert read_call.kwargs["ConsistentRead"] is consistent_read

        for batch_call in mock_client.batch_get_item.call_args_list:
            for table_request in batch_call.kwargs["RequestItems"].values():
                assert table_request["ConsistentRead"] is consistent_read
