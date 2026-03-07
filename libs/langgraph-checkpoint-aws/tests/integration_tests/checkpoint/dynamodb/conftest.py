import pytest

from langgraph_checkpoint_aws import DynamoDBSaver

TTL_SECONDS: int = 600  # 10 minutes


@pytest.fixture(scope="function")
def dynamodb_saver(
    dynamodb_table: str,
    s3_bucket: str,
    aws_region: str,
) -> DynamoDBSaver:
    """Create DynamoDBSaver with S3 offloading and TTL."""
    return DynamoDBSaver(
        table_name=dynamodb_table,
        region_name=aws_region,
        s3_offload_config={"bucket_name": s3_bucket},
        ttl_seconds=TTL_SECONDS,
        enable_checkpoint_compression=True,
    )
