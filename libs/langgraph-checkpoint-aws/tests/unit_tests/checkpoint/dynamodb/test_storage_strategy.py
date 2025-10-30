"""Comprehensive tests for StorageStrategy.

Tests cover:
- Size-based routing logic
- DynamoDB storage operations
- S3 storage operations
- TTL configuration
- Error handling
- Edge cases
"""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from langgraph_checkpoint_aws.checkpoint.dynamodb.storage_strategy import (
    S3_OFFLOAD_THRESHOLD,
    StorageStrategy,
)


class TestStorageStrategyInitialization:
    """Test StorageStrategy initialization."""

    def test_init_configurations(self):
        """Test initialization with various configurations."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # DynamoDB only (no S3)
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb, table_name="test_chunks"
        )
        assert strategy.dynamodb_client == mock_dynamodb
        assert strategy.table_name == "test_chunks"
        assert strategy.s3_client is None
        assert strategy.s3_enabled is False
        assert strategy.ttl_seconds is None

        # With S3 enabled
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )
        assert strategy.s3_client == mock_s3
        assert strategy.s3_bucket == "test-bucket"
        assert strategy.s3_enabled is True

        # With TTL
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            ttl_seconds=3600,
        )
        assert strategy.ttl_seconds == 3600

        # S3 requires both client and bucket
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,  # Only client, no bucket
        )
        assert strategy.s3_enabled is False


class TestShouldOffloadToS3:
    """Test should_offload_to_s3 logic."""

    def test_offload_logic_comprehensive(self):
        """Test S3 offload logic with various data sizes and configurations."""
        mock_client = Mock()
        mock_s3 = Mock()

        # With S3 enabled
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        # Small data - no offload
        assert strategy.should_offload_to_s3(b"x" * 1000) is False

        # Empty data - no offload
        assert strategy.should_offload_to_s3(b"") is False

        # At threshold - no offload
        assert strategy.should_offload_to_s3(b"x" * S3_OFFLOAD_THRESHOLD) is False

        # Over threshold - offload
        assert strategy.should_offload_to_s3(b"x" * (S3_OFFLOAD_THRESHOLD + 1)) is True

        # Without S3 - never offload even for large data
        strategy_no_s3 = StorageStrategy(
            dynamodb_client=mock_client, table_name="test_chunks"
        )
        assert (
            strategy_no_s3.should_offload_to_s3(b"x" * (S3_OFFLOAD_THRESHOLD + 1))
            is False
        )


class TestStoreData:
    """Test store_data method."""

    def test_store_small_data_to_dynamodb(self):
        """Test storing small data to DynamoDB."""
        mock_client = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        small_data = b"test_data"
        location = strategy.store_data("chunk_key", "s3_key", small_data)

        assert location == "DYNAMODB"
        mock_client.put_item.assert_called_once()
        call_args = mock_client.put_item.call_args[1]
        assert call_args["TableName"] == "test_chunks"
        assert call_args["Item"]["PK"]["S"] == "chunk_key"
        assert call_args["Item"]["payload"]["B"] == small_data

    def test_store_large_data_to_s3(self):
        """Test storing large data to S3."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        large_data = b"x" * (S3_OFFLOAD_THRESHOLD + 1)
        location = strategy.store_data(
            "chunk_key", "s3_key", large_data, allow_overwrite=True
        )

        assert location == "S3"
        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args[1]
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == "s3_key"
        assert call_args["Body"] == large_data

    def test_store_with_ttl(self):
        """Test storing data with TTL."""
        mock_client = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
            ttl_seconds=3600,
        )

        data = b"test_data"
        with patch("time.time", return_value=1000):
            strategy.store_data("chunk_key", "s3_key", data)

        call_args = mock_client.put_item.call_args[1]
        assert "ttl" in call_args["Item"]
        assert call_args["Item"]["ttl"]["N"] == str(1000 + 3600)

    def test_s3_lifecycle_policy_management(self):
        """Test S3 lifecycle policy creation, skipping, and TTL scenarios."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # Scenario 1: Create policy when TTL configured and no existing policy
        mock_s3.get_bucket_lifecycle_configuration.side_effect = ClientError(
            {"Error": {"Code": "NoSuchLifecycleConfiguration"}},
            "GetBucketLifecycleConfiguration",
        )
        StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            ttl_seconds=7200,  # 2 hours = 1 day when rounded up
        )
        mock_s3.put_bucket_lifecycle_configuration.assert_called_once()
        call_args = mock_s3.put_bucket_lifecycle_configuration.call_args[1]
        rule = call_args["LifecycleConfiguration"]["Rules"][0]
        assert rule["ID"] == "ttl-expiration-1d"
        assert rule["Expiration"]["Days"] == 1
        assert rule["Filter"] == {"Tag": {"Key": "ttl-days", "Value": "1"}}

        # Scenario 2: Don't create policy without TTL
        mock_s3.reset_mock()
        StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            ttl_seconds=None,
        )
        mock_s3.get_bucket_lifecycle_configuration.assert_not_called()
        mock_s3.put_bucket_lifecycle_configuration.assert_not_called()

        # Scenario 3: Skip if policy already exists
        mock_s3.reset_mock()
        mock_s3.get_bucket_lifecycle_configuration.side_effect = None
        mock_s3.get_bucket_lifecycle_configuration.return_value = {
            "Rules": [
                {
                    "ID": "ttl-expiration-1d",
                    "Status": "Enabled",
                    "Filter": {"Tag": {"Key": "ttl-days", "Value": "1"}},
                    "Expiration": {"Days": 1},
                }
            ]
        }
        StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            ttl_seconds=7200,
        )
        mock_s3.get_bucket_lifecycle_configuration.assert_called_once()
        mock_s3.put_bucket_lifecycle_configuration.assert_not_called()

    def test_s3_objects_tagged_with_ttl(self):
        """Test S3 objects tagged with TTL for lifecycle filtering."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # Mock lifecycle configuration check
        from botocore.exceptions import ClientError

        mock_s3.get_bucket_lifecycle_configuration.side_effect = ClientError(
            {"Error": {"Code": "NoSuchLifecycleConfiguration"}},
            "GetBucketLifecycleConfiguration",
        )

        # Create strategy with TTL (2 hours = 1 day)
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            ttl_seconds=7200,
        )

        # Store large data to S3
        large_data = b"x" * (S3_OFFLOAD_THRESHOLD + 1)
        strategy.store_data("chunk_key", "s3_key", large_data, allow_overwrite=True)

        # Verify object was tagged with TTL
        call_args = mock_s3.put_object.call_args[1]
        assert "Tagging" in call_args
        assert call_args["Tagging"] == "ttl-days=1"

    @pytest.mark.parametrize(
        "ttl_seconds,expected_days",
        [
            (1, 1),  # 1 second = 1 day (minimum)
            (3600, 1),  # 1 hour = 1 day (round up)
            (86400, 1),  # 1 day = 1 day
            (86401, 2),  # 1 day + 1 second = 2 days (round up)
            (604800, 7),  # 7 days = 7 days
        ],
    )
    def test_ttl_to_days_conversion_accuracy(self, ttl_seconds, expected_days):
        """Test accurate conversion of TTL seconds to days (always round up)."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # Mock no existing lifecycle configuration
        from botocore.exceptions import ClientError

        mock_s3.get_bucket_lifecycle_configuration.side_effect = ClientError(
            {"Error": {"Code": "NoSuchLifecycleConfiguration"}},
            "GetBucketLifecycleConfiguration",
        )

        # Create storage strategy
        StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test-table",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
            ttl_seconds=ttl_seconds,
        )

        # Verify correct days calculation in lifecycle rule
        call_args = mock_s3.put_bucket_lifecycle_configuration.call_args
        lifecycle_config = call_args[1]["LifecycleConfiguration"]
        actual_days = lifecycle_config["Rules"][0]["Expiration"]["Days"]

        assert actual_days == expected_days, (
            f"TTL {ttl_seconds}s should convert to {expected_days} days, "
            f"but got {actual_days} days"
        )


class TestRetrieveData:
    """Test retrieve_data method."""

    def test_retrieve_comprehensive(self):
        """Test retrieving data from DynamoDB and S3 with various scenarios."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # DynamoDB - successful retrieval
        mock_dynamodb.get_item.return_value = {
            "Item": {
                "PK": {"S": "chunk_key"},
                "SK": {"S": "CHUNK"},
                "payload": {"B": b"test_data"},
            }
        }
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb, table_name="test_chunks"
        )
        data = strategy.retrieve_data("chunk_key", "DYNAMODB")
        assert data == b"test_data"

        # DynamoDB - not found
        mock_dynamodb.get_item.return_value = {}
        data = strategy.retrieve_data("chunk_key", "DYNAMODB")
        assert data is None

        # S3 - successful retrieval
        mock_body = Mock()
        mock_body.read.return_value = b"s3_data"
        mock_s3.get_object.return_value = {"Body": mock_body}
        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )
        data = strategy.retrieve_data("s3_key", "S3")
        assert data == b"s3_data"

        # S3 - not found
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        data = strategy.retrieve_data("s3_key", "S3")
        assert data is None

        # Invalid location
        data = strategy.retrieve_data("key", "INVALID")
        assert data is None

    def test_retrieve_error_propagation(self):
        """Test that non-NotFound errors are propagated."""
        mock_client = Mock()
        mock_client.get_item.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "GetItem",
        )
        strategy = StorageStrategy(
            dynamodb_client=mock_client, table_name="test_chunks"
        )

        with pytest.raises(ClientError) as exc_info:
            strategy.retrieve_data("chunk_key", "DYNAMODB")
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"


class TestBatchDeleteData:
    """Test batch_delete_data method."""

    def test_batch_delete_empty_list(self):
        """Test batch delete with empty list."""
        mock_client = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        result = strategy.batch_delete_data([])

        assert result == {"failed": []}
        mock_client.batch_write_item.assert_not_called()

    def test_batch_delete_dynamodb_only(self):
        """Test batch deleting items from DynamoDB only."""
        mock_client = Mock()
        mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        items = [
            ("chunk_key_1", "DYNAMODB"),
            ("chunk_key_2", "DYNAMODB"),
            ("chunk_key_3", "DYNAMODB"),
        ]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        mock_client.batch_write_item.assert_called_once()

        call_args = mock_client.batch_write_item.call_args[1]
        request_items = call_args["RequestItems"]["test_chunks"]
        assert len(request_items) == 3

        # Verify delete requests structure
        for i, req in enumerate(request_items):
            assert "DeleteRequest" in req
            assert req["DeleteRequest"]["Key"]["PK"]["S"] == f"chunk_key_{i + 1}"
            assert req["DeleteRequest"]["Key"]["SK"]["S"] == "CHUNK"

    def test_batch_delete_s3_only(self):
        """Test batch deleting items from S3 only."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        mock_s3.delete_objects.return_value = {"Deleted": [{"Key": "s3_key_1"}]}

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        items = [("s3_key_1", "S3")]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        mock_s3.delete_objects.assert_called_once()

        call_args = mock_s3.delete_objects.call_args[1]
        assert call_args["Bucket"] == "test-bucket"
        assert len(call_args["Delete"]["Objects"]) == 1
        assert call_args["Delete"]["Objects"][0]["Key"] == "s3_key_1"

    def test_batch_delete_mixed_locations(self):
        """Test batch deleting items from both DynamoDB and S3."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        mock_dynamodb.batch_write_item.return_value = {"UnprocessedItems": {}}
        mock_s3.delete_objects.return_value = {
            "Deleted": [{"Key": "s3_key_1"}, {"Key": "s3_key_2"}]
        }

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        items = [
            ("chunk_key_1", "DYNAMODB"),
            ("s3_key_1", "S3"),
            ("chunk_key_2", "DYNAMODB"),
            ("s3_key_2", "S3"),
        ]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        mock_dynamodb.batch_write_item.assert_called_once()
        mock_s3.delete_objects.assert_called_once()

    def test_batch_delete_dynamodb_batching(self):
        """Test DynamoDB batch delete respects 25 item limit."""
        mock_client = Mock()
        mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        # Create 30 items (should be split into 2 batches: 25 + 5)
        items = [(f"chunk_key_{i}", "DYNAMODB") for i in range(30)]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        assert mock_client.batch_write_item.call_count == 2

        # Verify first batch has 25 items
        first_call = mock_client.batch_write_item.call_args_list[0][1]
        assert len(first_call["RequestItems"]["test_chunks"]) == 25

        # Verify second batch has 5 items
        second_call = mock_client.batch_write_item.call_args_list[1][1]
        assert len(second_call["RequestItems"]["test_chunks"]) == 5

    def test_batch_delete_s3_batching(self):
        """Test S3 batch delete respects 1000 item limit."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        mock_s3.delete_objects.return_value = {"Deleted": []}

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        # Create 1500 items (should be split into 2 batches: 1000 + 500)
        items = [(f"s3_key_{i}", "S3") for i in range(1500)]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        assert mock_s3.delete_objects.call_count == 2

        # Verify first batch has 1000 items
        first_call = mock_s3.delete_objects.call_args_list[0][1]
        assert len(first_call["Delete"]["Objects"]) == 1000

        # Verify second batch has 500 items
        second_call = mock_s3.delete_objects.call_args_list[1][1]
        assert len(second_call["Delete"]["Objects"]) == 500

    def test_batch_delete_dynamodb_unprocessed_items(self):
        """Test handling of unprocessed items in DynamoDB batch delete."""
        mock_client = Mock()

        # First call returns unprocessed items, second call succeeds
        mock_client.batch_write_item.side_effect = [
            {
                "UnprocessedItems": {
                    "test_chunks": [
                        {
                            "DeleteRequest": {
                                "Key": {
                                    "PK": {"S": "chunk_key_2"},
                                    "SK": {"S": "CHUNK"},
                                }
                            }
                        }
                    ]
                }
            },
            {"UnprocessedItems": {}},
        ]

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        items = [("chunk_key_1", "DYNAMODB"), ("chunk_key_2", "DYNAMODB")]

        result = strategy.batch_delete_data(items)

        assert result == {"failed": []}
        # Should be called twice: initial + retry
        assert mock_client.batch_write_item.call_count == 2

    def test_batch_delete_dynamodb_error(self):
        """Test error handling in DynamoDB batch delete."""
        mock_client = Mock()
        mock_client.batch_write_item.side_effect = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException"}},
            "BatchWriteItem",
        )

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        items = [("chunk_key_1", "DYNAMODB"), ("chunk_key_2", "DYNAMODB")]

        result = strategy.batch_delete_data(items)

        # All items should be marked as failed
        assert len(result["failed"]) == 2
        assert "chunk_key_1" in result["failed"]
        assert "chunk_key_2" in result["failed"]

    def test_batch_delete_s3_errors(self):
        """Test handling of S3 delete errors."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        mock_s3.delete_objects.return_value = {
            "Deleted": [{"Key": "s3_key_1"}],
            "Errors": [
                {
                    "Key": "s3_key_2",
                    "Code": "AccessDenied",
                    "Message": "Access denied",
                },
                {
                    "Key": "s3_key_3",
                    "Code": "InternalError",
                    "Message": "Internal error",
                },
            ],
        }

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        items = [("s3_key_1", "S3"), ("s3_key_2", "S3"), ("s3_key_3", "S3")]

        result = strategy.batch_delete_data(items)

        # Two items should be marked as failed
        assert len(result["failed"]) == 2
        assert "s3_key_2" in result["failed"]
        assert "s3_key_3" in result["failed"]

    def test_batch_delete_s3_client_error(self):
        """Test handling of S3 client errors."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        mock_s3.delete_objects.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable"}}, "DeleteObjects"
        )

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        items = [("s3_key_1", "S3"), ("s3_key_2", "S3")]

        result = strategy.batch_delete_data(items)

        # All items should be marked as failed
        assert len(result["failed"]) == 2
        assert "s3_key_1" in result["failed"]
        assert "s3_key_2" in result["failed"]

    def test_batch_delete_s3_when_disabled(self):
        """Test batch deleting S3 items when S3 is not configured."""
        mock_client = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        items = [("s3_key_1", "S3"), ("s3_key_2", "S3")]

        result = strategy.batch_delete_data(items)

        # All S3 items should be marked as failed
        assert len(result["failed"]) == 2
        assert "s3_key_1" in result["failed"]
        assert "s3_key_2" in result["failed"]

    def test_batch_delete_invalid_location(self):
        """Test batch delete with invalid storage location."""
        mock_client = Mock()
        mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        items = [("key_1", "INVALID"), ("chunk_key_1", "DYNAMODB")]

        strategy.batch_delete_data(items)

        # Only valid DynamoDB item should be processed
        mock_client.batch_write_item.assert_called_once()
        call_args = mock_client.batch_write_item.call_args[1]
        assert len(call_args["RequestItems"]["test_chunks"]) == 1

    def test_batch_delete_mixed_with_errors(self):
        """Test batch delete with mixed locations and partial failures."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()

        # DynamoDB succeeds
        mock_dynamodb.batch_write_item.return_value = {"UnprocessedItems": {}}

        # S3 has partial failure
        mock_s3.delete_objects.return_value = {
            "Deleted": [{"Key": "s3_key_1"}],
            "Errors": [{"Key": "s3_key_2", "Code": "AccessDenied"}],
        }

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        items = [
            ("chunk_key_1", "DYNAMODB"),
            ("s3_key_1", "S3"),
            ("s3_key_2", "S3"),
        ]

        result = strategy.batch_delete_data(items)

        # Only s3_key_2 should fail
        assert len(result["failed"]) == 1
        assert "s3_key_2" in result["failed"]


class TestStorageStrategyEdgeCases:
    """Test edge cases in StorageStrategy."""

    def test_store_at_threshold_boundary(self):
        """Test storing data at threshold boundaries."""
        mock_client = Mock()
        mock_s3 = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        # Just below threshold - should use DynamoDB
        data_below = b"x" * (S3_OFFLOAD_THRESHOLD - 1)
        location = strategy.store_data("chunk_key_1", "s3_key_1", data_below)
        assert location == "DYNAMODB"

        # Just above threshold - should use S3
        data_above = b"x" * (S3_OFFLOAD_THRESHOLD + 1)
        location = strategy.store_data(
            "chunk_key_2", "s3_key_2", data_above, allow_overwrite=True
        )
        assert location == "S3"

    def test_ttl_zero_not_configured(self):
        """Test that TTL=0 is not configured."""
        mock_client = Mock()
        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
            ttl_seconds=0,
        )

        data = b"test_data"
        strategy.store_data("chunk_key", "s3_key", data)

        call_args = mock_client.put_item.call_args[1]
        assert "ttl" not in call_args["Item"]


class TestStorageStrategyIntegration:
    """Test integration scenarios."""

    def test_store_and_retrieve_roundtrip_dynamodb(self):
        """Test complete store and retrieve cycle for DynamoDB."""
        mock_client = Mock()
        stored_items = {}

        def mock_put(TableName, Item, **kwargs):
            key = Item["PK"]["S"]
            stored_items[key] = Item
            return {}

        def mock_get(TableName, Key, **kwargs):
            key = Key["PK"]["S"]
            item = stored_items.get(key)
            return {"Item": item} if item else {}

        mock_client.put_item.side_effect = mock_put
        mock_client.get_item.side_effect = mock_get

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        # Store data
        test_data = b"test_data_content"
        location = strategy.store_data("test_key", "s3_key", test_data)
        assert location == "DYNAMODB"

        # Retrieve data
        retrieved = strategy.retrieve_data("test_key", "DYNAMODB")
        assert retrieved == test_data

    def test_store_and_retrieve_roundtrip_s3(self):
        """Test complete store and retrieve cycle for S3."""
        mock_dynamodb = Mock()
        mock_s3 = Mock()
        stored_objects = {}

        def mock_put(Bucket, Key, Body, **kwargs):
            stored_objects[(Bucket, Key)] = Body
            return {}

        def mock_get(Bucket, Key, **kwargs):
            body_data = stored_objects.get((Bucket, Key))
            if body_data is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                    "GetObject",
                )
            mock_body = Mock()
            mock_body.read.return_value = body_data
            return {"Body": mock_body}

        mock_s3.put_object.side_effect = mock_put
        mock_s3.get_object.side_effect = mock_get

        strategy = StorageStrategy(
            dynamodb_client=mock_dynamodb,
            table_name="test_chunks",
            s3_client=mock_s3,
            s3_bucket="test-bucket",
        )

        # Store large data
        large_data = b"x" * (S3_OFFLOAD_THRESHOLD + 1)
        location = strategy.store_data(
            "chunk_key", "test_s3_key", large_data, allow_overwrite=True
        )
        assert location == "S3"

        # Retrieve data
        retrieved = strategy.retrieve_data("test_s3_key", "S3")
        assert retrieved == large_data

    def test_store_batch_delete_retrieve(self):
        """Test store, batch delete, and retrieve cycle."""
        mock_client = Mock()
        stored_items = {}

        def mock_put(TableName, Item, **kwargs):
            key = Item["PK"]["S"]
            stored_items[key] = Item
            return {}

        def mock_get(TableName, Key, **kwargs):
            key = Key["PK"]["S"]
            item = stored_items.get(key)
            return {"Item": item} if item else {}

        def mock_batch_write(RequestItems, **kwargs):
            for _table_name, requests in RequestItems.items():
                for request in requests:
                    if "DeleteRequest" in request:
                        key = request["DeleteRequest"]["Key"]["PK"]["S"]
                        stored_items.pop(key, None)
            return {"UnprocessedItems": {}}

        mock_client.put_item.side_effect = mock_put
        mock_client.get_item.side_effect = mock_get
        mock_client.batch_write_item.side_effect = mock_batch_write

        strategy = StorageStrategy(
            dynamodb_client=mock_client,
            table_name="test_chunks",
        )

        # Store multiple items
        test_data_1 = b"test_data_1"
        test_data_2 = b"test_data_2"
        strategy.store_data("test_key_1", "s3_key_1", test_data_1)
        strategy.store_data("test_key_2", "s3_key_2", test_data_2)

        # Verify stored
        retrieved_1 = strategy.retrieve_data("test_key_1", "DYNAMODB")
        assert retrieved_1 == test_data_1
        retrieved_2 = strategy.retrieve_data("test_key_2", "DYNAMODB")
        assert retrieved_2 == test_data_2

        # Batch delete data
        items_to_delete = [("test_key_1", "DYNAMODB"), ("test_key_2", "DYNAMODB")]
        result = strategy.batch_delete_data(items_to_delete)
        assert result == {"failed": []}

        # Verify deleted
        retrieved_after_delete_1 = strategy.retrieve_data("test_key_1", "DYNAMODB")
        assert retrieved_after_delete_1 is None
        retrieved_after_delete_2 = strategy.retrieve_data("test_key_2", "DYNAMODB")
        assert retrieved_after_delete_2 is None
