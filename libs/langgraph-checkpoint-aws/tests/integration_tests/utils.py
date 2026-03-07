import random
import string
from typing import TYPE_CHECKING, Literal

import pytest
from botocore.exceptions import ClientError
from langchain_core.tools import tool

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

##########################################################
# Pytest
##########################################################


def skip_on_aws_403(call_fn, action_description: str):
    """Skip test if AWS returns 403/AccessDenied error."""
    try:
        return call_fn()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("AccessDenied", "AccessDeniedException", "403"):
            pytest.skip(
                f"Insufficient permissions to execute "
                f"{action_description}, skipping test."
            )
        else:
            raise


##########################################################
# S3
##########################################################


def verify_s3_checkpoint_exists(
    bucket: str,
    thread_id: str,
    s3_client: "S3Client",
) -> tuple[bool, int]:
    """Check if checkpoint data exists in S3 for given thread.

    Returns:
        tuple[bool, int]: Whether the checkpoint exists, and its total size in bytes.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{thread_id}/")
        objects = response.get("Contents", [])
        total_size = sum(obj["Size"] for obj in objects)
        return len(objects) > 0, total_size
    except ClientError:
        return False, 0


##########################################################
# Utils
##########################################################


def generate_large_data(size_kb: int) -> str:
    """Generate random data of specified size to prevent compression."""
    return "".join(
        random.choices(string.ascii_letters + string.digits, k=size_kb * 1024)
    )


##########################################################
# Tools
##########################################################


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")
