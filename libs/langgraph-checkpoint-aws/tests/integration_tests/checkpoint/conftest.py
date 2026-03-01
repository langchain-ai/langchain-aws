import logging
import os

import pytest
from botocore.exceptions import ClientError

from tests.integration_tests.conftest import skip_on_aws_403


logger = logging.getLogger(__name__)


# Configuration constants
DYNAMODB_TABLE = os.getenv(
    "DYNAMODB_TABLE_NAME", "langgraph-checkpoints-dynamodb-integ"
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "langgraph-checkpoints-bucket-integ")


@pytest.fixture(scope="session")
def dynamodb_table_name() -> str:
    """Get DynamoDB table name from environment or use default."""
    return DYNAMODB_TABLE


@pytest.fixture(scope="session")
def s3_bucket_name() -> str:
    """Get S3 bucket name from environment or use default."""
    return S3_BUCKET


@pytest.fixture(scope="session")
def dynamodb_table(dynamodb_client, dynamodb_table_name: str) -> str:
    """Ensure DynamoDB table exists and return its name."""
    try:
        skip_on_aws_403(
            lambda: dynamodb_client.describe_table(TableName=dynamodb_table_name),
            f"DynamoDB DescribeTable on {dynamodb_table_name}",
        )
        logger.info(f"DynamoDB table '{dynamodb_table_name}' already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info(f"Creating DynamoDB table '{dynamodb_table_name}'...")
            skip_on_aws_403(
                lambda: dynamodb_client.create_table(
                    TableName=dynamodb_table_name,
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
            waiter = dynamodb_client.get_waiter("table_exists")
            skip_on_aws_403(
                lambda: waiter.wait(TableName=dynamodb_table_name),
                "DynamoDB GetWaiter",
            )
            logger.info(f"DynamoDB table '{dynamodb_table_name}' created successfully")
        else:
            raise

    return dynamodb_table_name


@pytest.fixture(scope="session")
def s3_bucket(s3_client, s3_bucket_name: str, aws_region: str) -> str:
    """Ensure S3 bucket exists and return its name."""
    try:
        skip_on_aws_403(
            lambda: s3_client.head_bucket(Bucket=s3_bucket_name),
            "S3 HeadBucket",
        )
        logger.info(f"S3 bucket '{s3_bucket_name}' already exists")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.info(f"Creating S3 bucket '{s3_bucket_name}'...")

            def create_bucket():
                if aws_region == "us-east-1":
                    s3_client.create_bucket(Bucket=s3_bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=s3_bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": aws_region},
                    )

            skip_on_aws_403(create_bucket, "S3 CreateBucket")
            logger.info(f"S3 bucket '{s3_bucket_name}' created successfully")
        else:
            raise

    return s3_bucket_name
