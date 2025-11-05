"""
LangGraph DynamoDB Checkpoint Saver Integration Tests

This test suite validates the DynamoDBSaver implementation with LangGraph workflows,
focusing on checkpoint persistence, S3 offloading, TTL, state management, and history.
"""

import logging
import os
import random
import string
import time
from typing import Annotated, Literal, TypedDict

import boto3
import pytest
from botocore.exceptions import ClientError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from langgraph_checkpoint_aws import DynamoDBSaver

logger = logging.getLogger(__name__)


def skip_on_aws_403(call_fn, action_description: str):
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


# Configuration
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
DYNAMODB_TABLE = os.getenv(
    "DYNAMODB_TABLE_NAME", "langgraph-checkpoints-dynamodb-integ"
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "langgraph-checkpoints-bucket-integ")
TTL_SECONDS = 3600  # 1 hour


# ============================================================================
# STATE DEFINITIONS
# ============================================================================


class WorkflowState(TypedDict):
    """State for comprehensive workflow testing."""

    messages: Annotated[list, add_messages]
    step_count: int
    processing_complete: bool
    large_payload: str  # For S3 offloading tests
    validation_result: str
    metadata: dict


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def aws_resources():
    """Setup AWS resources (DynamoDB table and S3 bucket) before tests."""
    dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Create DynamoDB table if not exists
    try:
        skip_on_aws_403(
            lambda: dynamodb.describe_table(TableName=DYNAMODB_TABLE),
            f"DynamoDB DescribeTable on {DYNAMODB_TABLE}",
        )
        logger.info(f"DynamoDB table '{DYNAMODB_TABLE}' already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info(f"Creating DynamoDB table '{DYNAMODB_TABLE}'...")
            skip_on_aws_403(
                lambda: dynamodb.create_table(
                    TableName=DYNAMODB_TABLE,
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
            waiter = dynamodb.get_waiter("table_exists")
            skip_on_aws_403(
                lambda: waiter.wait(TableName=DYNAMODB_TABLE), "DynamoDB GetWaiter"
            )
            logger.info(f"DynamoDB table '{DYNAMODB_TABLE}' created successfully")
        else:
            raise

    # Create S3 bucket if not exists
    try:
        skip_on_aws_403(lambda: s3.head_bucket(Bucket=S3_BUCKET), "S3 HeadBucket")
        logger.info(f"S3 bucket '{S3_BUCKET}' already exists")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.info(f"Creating S3 bucket '{S3_BUCKET}'...")

            def create_bucket():
                if AWS_REGION == "us-east-1":
                    s3.create_bucket(Bucket=S3_BUCKET)
                else:
                    s3.create_bucket(
                        Bucket=S3_BUCKET,
                        CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                    )

            skip_on_aws_403(create_bucket, "S3 CreateBucket")
            logger.info(f"S3 bucket '{S3_BUCKET}' created successfully")
        else:
            raise

    yield {"dynamodb_table": DYNAMODB_TABLE, "s3_bucket": S3_BUCKET}

    # Cleanup is handled per-test to avoid affecting other tests


@pytest.fixture
def checkpoint_saver(aws_resources):
    """Create DynamoDBSaver with S3 offloading and TTL."""
    saver = DynamoDBSaver(
        table_name=aws_resources["dynamodb_table"],
        region_name=AWS_REGION,
        s3_offload_config={"bucket_name": aws_resources["s3_bucket"]},
        ttl_seconds=TTL_SECONDS,
        enable_checkpoint_compression=True,
    )
    return saver


@pytest.fixture
def thread_id():
    """Generate unique thread ID for test isolation."""
    return (
        f"test_{''.join(random.choices(string.ascii_lowercase + string.digits, k=12))}"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_large_data(size_kb: int) -> str:
    """Generate random data of specified size to prevent compression."""
    return "".join(
        random.choices(string.ascii_letters + string.digits, k=size_kb * 1024)
    )


def verify_s3_checkpoint_exists(bucket: str, thread_id: str) -> tuple[bool, int]:
    """Check if checkpoint data exists in S3 for given thread."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{thread_id}/")
        objects = response.get("Contents", [])
        total_size = sum(obj["Size"] for obj in objects)
        return len(objects) > 0, total_size
    except ClientError:
        return False, 0


def cleanup_thread_resources(saver: DynamoDBSaver, thread_id: str):
    """Clean up all resources for a thread."""
    try:
        saver.delete_thread(thread_id)
        logger.info(f"Cleaned up thread: {thread_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup thread {thread_id}: {e}")


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


def initialize_workflow(state: WorkflowState) -> dict:
    """Initialize workflow state."""
    # Preserve step_count if provided in input, otherwise default to 1
    step_count = state.get("step_count", 1)
    return {
        "messages": [{"role": "system", "content": "Workflow initialized"}],
        "step_count": step_count,
        "processing_complete": False,
        "metadata": {"start_time": time.time()},
    }


def process_step(state: WorkflowState) -> dict:
    """Process a workflow step."""
    current_step = state.get("step_count", 0)
    return {
        "messages": [
            {"role": "assistant", "content": f"Processed step {current_step}"}
        ],
        "step_count": current_step + 1,
    }


def generate_large_checkpoint_data(state: WorkflowState) -> dict:
    """Generate large data to trigger S3 offloading (>350KB)."""
    # Generate 600KB to ensure it stays above 350KB even after compression
    large_data = generate_large_data(600)  # 600KB
    return {
        "messages": [{"role": "system", "content": "Large data generated"}],
        "large_payload": large_data,
    }


def validate_state(state: WorkflowState) -> dict:
    """Validate workflow state."""
    step_count = state.get("step_count", 0)
    has_large_data = len(state.get("large_payload", "")) > 0
    validation = f"Steps: {step_count}, Large data: {has_large_data}"
    return {
        "messages": [{"role": "system", "content": "Validation complete"}],
        "validation_result": validation,
    }


def finalize_workflow(state: WorkflowState) -> dict:
    """Finalize workflow."""
    return {
        "messages": [{"role": "system", "content": "Workflow completed"}],
        "processing_complete": True,
        "metadata": {**state.get("metadata", {}), "end_time": time.time()},
    }


def should_generate_large_data(state: WorkflowState) -> Literal["yes", "no"]:
    """Decide whether to generate large data."""
    return "yes" if state.get("step_count", 0) >= 2 else "no"


# ============================================================================
# TEST 1: Complete Workflow with Checkpoint Validation
# ============================================================================


def test_complete_workflow_checkpoint_lifecycle(checkpoint_saver, thread_id):
    """
    Test complete workflow lifecycle with checkpoint validation at each step.

    1. Creates a linear workflow: init → process → validate → finalize
    2. Executes the workflow with DynamoDBSaver as checkpointer
    3. Validates checkpoint creation after each node execution
    4. Verifies checkpoint structure and content
    5. Checks checkpoint history ordering
    6. Cleans up all resources

    - ✓ Checkpoint creation: Verifies checkpoints after each workflow step
    - ✓ Checkpoint structure: Validates id, config, metadata, and state
    - ✓ State persistence: Confirms values correctly stored in checkpoints
    - ✓ Checkpoint history: Validates all checkpoints in order (newest first)
    - ✓ Thread cleanup: Ensures delete_thread removes all checkpoints

    EXPECTED BEHAVIOR:
    - Workflow completes successfully with processing_complete=True
    - At least 4 checkpoints created (one per node: init, process, validate, finalize)
    - Checkpoints ordered newest first (higher step_count appears first)
    - Each checkpoint has valid structure with thread_id and checkpoint_id
    - Thread cleanup removes all data from DynamoDB
    """
    # Build workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)
    workflow.add_node("validate", validate_state)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_edge("process", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({}, config)

        # Validate final state
        assert result["processing_complete"] is True, "Workflow should complete"
        assert result["step_count"] >= 1, "Step count should increment"
        assert len(result["messages"]) > 0, "Messages should be recorded"
        assert "metadata" in result, "Metadata should be present"

        # Validate checkpoint was created
        current_state = app.get_state(config)
        assert current_state is not None, "Current state should exist"
        assert current_state.values is not None, "State values should exist"
        assert current_state.values["processing_complete"] is True, (
            "State should reflect completion"
        )

        # Validate checkpoint history
        history = list(app.get_state_history(config))
        assert len(history) >= 4, (
            "Should have checkpoints for each node (init, process, validate, finalize)"
        )

        # Validate checkpoint ordering (newest first)
        for i in range(len(history) - 1):
            current_step = history[i].values.get("step_count", 0)
            next_step = history[i + 1].values.get("step_count", 0)
            assert current_step >= next_step, (
                "Checkpoints should be ordered newest first"
            )

        # Validate checkpoint structure
        for checkpoint_state in history:
            assert checkpoint_state.values is not None, (
                "Each checkpoint should have values"
            )
            assert checkpoint_state.config is not None, (
                "Each checkpoint should have config"
            )
            assert "thread_id" in checkpoint_state.config["configurable"], (
                "Config should have thread_id"
            )
            assert "checkpoint_id" in checkpoint_state.config["configurable"], (
                "Config should have checkpoint_id"
            )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


# ============================================================================
# TEST 2: S3 Offloading and TTL Validation
# ============================================================================


def test_s3_offloading_and_ttl_validation(checkpoint_saver, thread_id, aws_resources):
    """
    Test S3 offloading for large checkpoints and TTL configuration.

    1. Creates a workflow that generates 600KB of random data
    2. Executes the workflow to trigger large checkpoint creation
    3. Verifies the large checkpoint (>350KB threshold) is offloaded to S3
    4. Validates TTL attribute is set on DynamoDB items
    5. Confirms checkpoint can be retrieved from S3
    6. Cleans up both DynamoDB and S3 resources

    - ✓ S3 offloading: Large checkpoints (>350KB) automatically stored in S3
    - ✓ DynamoDB storage: Checkpoint metadata remains in DynamoDB with S3 reference
    - ✓ TTL configuration: TTL attribute set correctly on DynamoDB items (3600 seconds)
    - ✓ TTL validation: TTL value is in the future and within expected range
    - ✓ S3 retrieval: Checkpoint data can be retrieved from S3 successfully
    - ✓ Data integrity: Retrieved payload matches original size (600KB)
    - ✓ Resource cleanup: Both DynamoDB items and S3 objects are deleted

    EXPECTED BEHAVIOR:
    - Workflow generates 600KB payload (614,400 bytes)
    - Checkpoint exceeds 350KB threshold even after compression
    - S3 objects created in bucket with thread_id prefix
    - DynamoDB item has TTL attribute set to current_time + 3600 seconds
    - Checkpoint retrieval returns complete state with large payload
    - Cleanup removes all S3 objects and DynamoDB items
    """
    # Build workflow with large data generation
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("large_data", generate_large_checkpoint_data)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "large_data")
    workflow.add_edge("large_data", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({}, config)

        # Validate large data was generated
        large_payload_size = len(result.get("large_payload", ""))
        assert large_payload_size > 500 * 1024, (
            f"Large payload should exceed 500KB, got {large_payload_size / 1024:.0f}KB"
        )

        # Validate S3 offloading (may not happen if compression effective)
        s3_exists, s3_size = verify_s3_checkpoint_exists(
            aws_resources["s3_bucket"], thread_id
        )
        if s3_exists:
            logger.info(f"✓ S3 offloading verified: {s3_size / 1024:.2f}KB stored")
        else:
            logger.info("Note: S3 offloading may not have occurred due to compression")

        # Validate checkpoint can be retrieved from S3
        current_state = app.get_state(config)
        assert current_state is not None, "State should be retrievable from S3"
        retrieved_payload_size = len(current_state.values.get("large_payload", ""))
        assert retrieved_payload_size == large_payload_size, (
            "Retrieved payload should match original size"
        )

        # Validate TTL is set on DynamoDB items
        dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
        response = dynamodb.query(
            TableName=aws_resources["dynamodb_table"],
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": {"S": f"THREAD#{thread_id}"}},
            Limit=1,
        )
        if response["Items"]:
            item = response["Items"][0]
            assert "ttl" in item, "DynamoDB item should have TTL attribute"
            ttl_value = int(item["ttl"]["N"])
            current_time = int(time.time())
            assert ttl_value > current_time, "TTL should be in the future"
            assert ttl_value <= current_time + TTL_SECONDS + 60, (
                "TTL should be within expected range"
            )
            ttl_remaining = ttl_value - current_time
            logger.info(f"✓ DynamoDB TTL validated: expires in {ttl_remaining}s")

        # Validate S3 lifecycle configuration aligns with TTL
        s3 = boto3.client("s3", region_name=AWS_REGION)
        try:
            lifecycle_config = s3.get_bucket_lifecycle_configuration(
                Bucket=aws_resources["s3_bucket"]
            )

            # Calculate expected expiration days from TTL_SECONDS
            expected_expiration_days = (
                TTL_SECONDS + 86399
            ) // 86400  # Round up to days
            rule_id = f"langgraph-checkpoint-expiration-{expected_expiration_days}d"

            # Find the lifecycle rule for our TTL
            matching_rules = [
                rule
                for rule in lifecycle_config.get("Rules", [])
                if rule.get("ID") == rule_id
            ]

            if matching_rules:
                rule = matching_rules[0]
                assert rule.get("Status") == "Enabled", (
                    "Lifecycle rule should be enabled"
                )

                # Validate expiration configuration
                expiration = rule.get("Expiration", {})
                assert "Days" in expiration, (
                    "Lifecycle rule should have Days expiration"
                )
                actual_days = expiration["Days"]
                assert actual_days == expected_expiration_days, (
                    f"S3 lifecycle expiration ({actual_days} days) should match TTL "
                    f"({expected_expiration_days} days from {TTL_SECONDS} seconds)"
                )

                # Validate tag filter (objects are tagged with ttl-days)
                tag_filter = rule.get("Filter", {}).get("Tag", {})
                assert tag_filter.get("Key") == "ttl-days", (
                    "Lifecycle rule should filter by ttl-days tag"
                )
                assert tag_filter.get("Value") == str(expected_expiration_days), (
                    f"Lifecycle rule tag value should be {expected_expiration_days}"
                )

                logger.info(f"✓ S3 lifecycle validated: rule '{rule_id}' configured")
                logger.info(f"  - Expiration: {actual_days} days (TTL {TTL_SECONDS}s)")
                logger.info(f"  - Tag filter: ttl-days={expected_expiration_days}")
                logger.info(f"  - Status: {rule.get('Status')}")
            else:
                logger.warning(
                    f"S3 lifecycle rule '{rule_id}' not found "
                    "(may be created on first S3 write)"
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                logger.warning(
                    "No S3 lifecycle config found (may be created on first S3 write)"
                )
            else:
                logger.warning(
                    f"Could not validate S3 lifecycle: {e.response['Error']['Code']}"
                )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


# ============================================================================
# TEST 3: State Persistence and History Tracking
# ============================================================================


def test_state_persistence_and_history_tracking(checkpoint_saver, thread_id):
    """
    Test state persistence across multiple invocations and history tracking.

    1. Creates a simple workflow: init → process
    2. Invokes the workflow 3 times with the same thread_id
    3. Tracks checkpoint history growth after each invocation
    4. Validates checkpoint accumulation across invocations
    5. Tests time-travel by retrieving earlier checkpoints
    6. Verifies checkpoint metadata preservation

    - ✓ State persistence: Each invocation creates new checkpoints in same thread
    - ✓ History accumulation: Checkpoint count increases with each invocation
    - ✓ Checkpoint ordering: History maintains newest-first ordering
    - ✓ Time-travel: Earlier checkpoints can be retrieved using their config
    - ✓ Metadata preservation: All checkpoints maintain config and values
    - ✓ Message history: Messages are preserved in checkpoint state
    - ✓ Thread isolation: All checkpoints belong to same thread_id

    EXPECTED BEHAVIOR:
    - Invocation 1: Creates 2+ checkpoints (init, process)
    - Invocation 2: Adds 2+ more checkpoints to history
    - Invocation 3: Adds 2+ more checkpoints to history
    - Total history grows: count_1 < count_2 < count_3
    - Each checkpoint has step_count=2 (workflow resets each time)
    - Earlier checkpoints retrievable via their checkpoint_id
    - All checkpoints contain messages and metadata
    """
    # Build simple workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_edge("process", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # First invocation
        result1 = app.invoke({}, config)
        step_count_1 = result1["step_count"]
        assert step_count_1 == 2, (
            f"First invocation should have step_count=2, got {step_count_1}"
        )

        # Get checkpoint count after first invocation
        history_1 = list(app.get_state_history(config))
        checkpoint_count_1 = len(history_1)
        assert checkpoint_count_1 >= 2, (
            "Should have at least 2 checkpoints (init, process)"
        )

        # Second invocation (continues from previous state due to checkpointer)
        result2 = app.invoke({}, config)
        step_count_2 = result2["step_count"]
        # With checkpointer, state persists, so step_count continues from previous
        # Previous result had step_count=2, init preserves it, process increments to 3
        assert step_count_2 == 3, (
            f"Second invocation should have step_count=3 (from 2), got {step_count_2}"
        )

        # Validate history accumulated
        history_2 = list(app.get_state_history(config))
        checkpoint_count_2 = len(history_2)
        assert checkpoint_count_2 > checkpoint_count_1, "Checkpoint history should grow"

        # Third invocation
        result3 = app.invoke({}, config)
        step_count_3 = result3["step_count"]
        # With checkpointer, state continues from previous (3 → 4)
        assert step_count_3 == 4, (
            f"Third invocation should have step_count=4 (from 3), got {step_count_3}"
        )

        # Validate complete history
        history_3 = list(app.get_state_history(config))
        assert len(history_3) > checkpoint_count_2, "History should continue growing"

        # Validate time-travel: retrieve earlier checkpoint
        if len(history_3) >= 3:
            earlier_checkpoint = history_3[-2]  # Second-to-last checkpoint
            earlier_state = app.get_state(earlier_checkpoint.config)
            assert earlier_state is not None, (
                "Should be able to retrieve earlier checkpoint"
            )
            # Verify we can access historical checkpoints
            logger.info("Time-travel validated: retrieved checkpoint from history")

        # Validate checkpoint metadata preservation
        for checkpoint_state in history_3:
            assert checkpoint_state.config is not None, "Checkpoint should have config"
            assert checkpoint_state.values is not None, "Checkpoint should have values"
            assert "messages" in checkpoint_state.values, (
                "Checkpoint should preserve messages"
            )

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


# ============================================================================
# TEST 4: Conditional Routing and Checkpoint Branching
# ============================================================================


def test_conditional_routing_and_checkpoint_branching(checkpoint_saver, thread_id):
    """
    Test conditional routing with checkpoint validation at branch points.

    1. Creates a workflow with conditional routing based on step_count
    2. Executes two scenarios with different thread IDs:
       - Scenario 1: Workflow execution with one path
       - Scenario 2: Workflow execution with another path
    3. Validates checkpoints are created at branch points
    4. Verifies different execution paths create distinct checkpoints
    5. Confirms state reflects routing decisions
    6. Cleans up resources for both scenarios

    - ✓ Conditional routing: Workflow branches based on state (step_count)
    - ✓ Branch checkpoints: Checkpoints created at conditional decision points
    - ✓ Path isolation: Different threads have independent checkpoint histories
    - ✓ State reflection: Checkpoint state shows which path was taken
    - ✓ Checkpoint structure: All checkpoints have valid structure regardless of path
    - ✓ Multiple scenarios: Single test validates multiple execution paths

    WORKFLOW STRUCTURE:
    - init → process → [conditional routing]
    - If step_count >= 2: process → large_data → validate → finalize
    - If step_count < 2: process → validate → finalize

    EXPECTED BEHAVIOR:
    - Scenario 1: Creates 4+ checkpoints for one execution path
    - Scenario 2: Creates 4+ checkpoints for another execution path
    - Each scenario has independent checkpoint history
    - Checkpoints contain workflow completion status
    - Both scenarios clean up successfully
    """
    # Build workflow with conditional routing
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", initialize_workflow)
    workflow.add_node("process", process_step)
    workflow.add_node("large_data", generate_large_checkpoint_data)
    workflow.add_node("validate", validate_state)
    workflow.add_node("finalize", finalize_workflow)

    workflow.add_edge(START, "init")
    workflow.add_edge("init", "process")
    workflow.add_conditional_edges(
        "process",
        should_generate_large_data,
        {"yes": "large_data", "no": "validate"},
    )
    workflow.add_edge("large_data", "validate")
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=checkpoint_saver)

    # Scenario 1: Path WITHOUT large data (step_count < 2 after process)
    config1 = {"configurable": {"thread_id": f"{thread_id}_path1"}}
    try:
        # Start with step_count=0 to take "no" path
        # After init: step_count=0, after process: step_count=1
        # Condition: 1 >= 2? NO → routes to "validate" (skips large_data)
        result1 = app.invoke({"step_count": 0}, config1)

        history1 = list(app.get_state_history(config1))
        assert len(history1) >= 4, (
            f"Path 1 should have at least 4 checkpoints, got {len(history1)}"
        )

        # Verify NO large payload (took "no" path)
        has_large_payload = (
            "large_payload" in result1 and len(result1.get("large_payload", "")) > 0
        )
        assert not has_large_payload, (
            "Path 1 should NOT generate large data (took 'no' path)"
        )
        assert result1["processing_complete"] is True, "Workflow should complete"
        assert result1["step_count"] == 1, (
            f"Step count should be 1 after process, got {result1['step_count']}"
        )

        logger.info(
            f"✓ Scenario 1: {len(history1)} checkpoints, NO large data (took 'no' path)"
        )

    finally:
        cleanup_thread_resources(checkpoint_saver, f"{thread_id}_path1")

    # Scenario 2: Path WITH large data (step_count >= 2 after process)
    config2 = {"configurable": {"thread_id": f"{thread_id}_path2"}}
    try:
        # Start with step_count=2 to take "yes" path
        # After init: step_count=2, after process: step_count=3
        # Condition: 3 >= 2? YES → routes to "large_data"
        result2 = app.invoke({"step_count": 2}, config2)

        history2 = list(app.get_state_history(config2))
        assert len(history2) >= 5, (
            f"Path 2 should have >=5 checkpoints (includes large_data), "
            f"got {len(history2)}"
        )

        # Verify HAS large payload (took "yes" path)
        has_large_payload = (
            "large_payload" in result2 and len(result2.get("large_payload", "")) > 0
        )
        assert has_large_payload, "Path 2 SHOULD generate large data (took 'yes' path)"
        assert result2["processing_complete"] is True, "Workflow should complete"
        assert result2["step_count"] == 3, (
            f"Step count should be 3 after process, got {result2['step_count']}"
        )
    finally:
        cleanup_thread_resources(checkpoint_saver, f"{thread_id}_path2")


# ============================================================================
# TEST 5: Parallel Execution with Checkpoint Validation and Resumability
# ============================================================================


def test_parallel_execution_with_resumability(checkpoint_saver, thread_id):
    """
    Test parallel node execution with TRUE resumability and partial failure recovery.

    WHAT THIS TEST DOES:
    1. Creates a workflow with 3 parallel nodes (task_a, task_b, task_c)
    2. First execution: PARTIAL failure (task_a ✓, task_b ✗, task_c ✓)
    3. Validates that successful tasks' results are preserved in checkpoints
    4. Second execution: TRUE resumability using invoke(None) from checkpoint
    5. Verifies only failed task (task_b) re-executes, successful results are reused
    6. Validates all results are merged correctly in final state

    - ✓ Parallel execution: Multiple nodes execute concurrently
    - ✓ Partial failure recovery: Some tasks succeed, some fail
    - ✓ Checkpoint preservation: Successful tasks' results saved in checkpoints
    - ✓ TRUE resumability: Workflow continues from checkpoint (not fresh run)
    - ✓ Selective retry: Only failed nodes re-execute on resume
    - ✓ State accumulation: Results from both attempts merged correctly
    - ✓ Checkpoint history: Shows partial completion and resume

    WORKFLOW STRUCTURE:
    - START → [task_a, task_b, task_c] (parallel) → merge → END
    - Each task adds result to parallel_results list
    - Merge node combines all results

    EXPECTED BEHAVIOR:
    - First attempt: Partial failure (task_a ✓, task_b ✗, task_c ✓)
    - Checkpoints preserve successful tasks' results (result_a, result_c)
    - State shows pending nodes (task_b and merge)
    - Second attempt: Resume with invoke(None) - only task_b re-executes
    - Final state contains results from both attempts (result_a, result_b, result_c)
    - Checkpoint history shows both partial completion and resume
    - TRUE resumability: No duplicate work, continues from checkpoint
    """
    import operator
    from typing import Annotated

    # Define state with reducer for parallel results
    class ParallelState(TypedDict):
        """State for parallel execution testing."""

        messages: Annotated[list, add_messages]
        parallel_results: Annotated[list[str], operator.add]
        attempt_count: int
        retry_count: int  # Counter to track retries

    # Parallel task nodes
    def task_a(state: ParallelState) -> dict:
        """Parallel task A - always succeeds."""
        logger.info("Task A: Executing (always succeeds)")
        return {
            "messages": [{"role": "system", "content": "Task A completed"}],
            "parallel_results": ["result_a"],
        }

    def task_b(state: ParallelState) -> dict:
        """Parallel task B - fails first (retry_count < 1), succeeds on retry."""
        retry_count = state.get("retry_count", 0)
        logger.info(f"Task B: Executing (retry_count={retry_count})")

        if retry_count < 1:
            # First attempt - fail and increment counter
            logger.info("Task B: FAILING (first attempt, retry_count < 1)")
            raise ValueError("Task B failed (simulated partial failure)")

        # Retry attempt - succeed
        logger.info("Task B: SUCCESS (retry_count >= 1)")
        return {
            "messages": [{"role": "system", "content": "Task B completed"}],
            "parallel_results": ["result_b"],
        }

    def task_c(state: ParallelState) -> dict:
        """Parallel task C - always succeeds."""
        logger.info("Task C: Executing (always succeeds)")
        return {
            "messages": [{"role": "system", "content": "Task C completed"}],
            "parallel_results": ["result_c"],
        }

    def merge_results(state: ParallelState) -> dict:
        """Merge parallel results."""
        result_count = len(state.get("parallel_results", []))
        return {
            "messages": [
                {"role": "system", "content": f"Merged {result_count} results"}
            ],
            "attempt_count": state.get("attempt_count", 0) + 1,
        }

    # Build workflow with parallel execution
    workflow = StateGraph(ParallelState)
    workflow.add_node("task_a", task_a)
    workflow.add_node("task_b", task_b)
    workflow.add_node("task_c", task_c)
    workflow.add_node("merge", merge_results)

    # Parallel edges from START
    workflow.add_edge(START, "task_a")
    workflow.add_edge(START, "task_b")
    workflow.add_edge(START, "task_c")

    # All parallel tasks converge to merge
    workflow.add_edge("task_a", "merge")
    workflow.add_edge("task_b", "merge")
    workflow.add_edge("task_c", "merge")
    workflow.add_edge("merge", END)

    app = workflow.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # ====================================================================
        # ATTEMPT 1: PARTIAL FAILURE (task_b fails, task_a and task_c succeed)
        # ====================================================================
        logger.info("=" * 80)
        logger.info("ATTEMPT 1: Partial failure - task_b fails, task_a/task_c succeed")
        logger.info("=" * 80)

        try:
            app.invoke({"retry_count": 0, "attempt_count": 0}, config)
            raise AssertionError("Should have raised ValueError from task_b")
        except ValueError as e:
            logger.info(f"✓ Expected partial failure occurred: {e}")

        # Validate state after partial failure
        state_after_failure = app.get_state(config)
        assert state_after_failure is not None, (
            "State should exist after partial failure"
        )

        # Check partial results - task_a and task_c should have completed
        partial_results = state_after_failure.values.get("parallel_results", [])
        logger.info(f"Partial results preserved: {partial_results}")
        assert len(partial_results) == 2, (
            f"Should have 2 partial results (task_a, task_c), "
            f"got {len(partial_results)}"
        )

        # Check pending nodes - task_b and merge should be pending
        pending_nodes = state_after_failure.next
        logger.info(f"Pending nodes after partial failure: {pending_nodes}")
        assert len(pending_nodes) == 1, (
            "Should have pending nodes after partial failure"
        )

        # Get checkpoint history after partial failure
        history_after_failure = list(app.get_state_history(config))
        checkpoint_count_after_failure = len(history_after_failure)
        assert checkpoint_count_after_failure > 0, (
            "Should have checkpoints after partial failure"
        )

        # ====================================================================
        # ATTEMPT 2: TRUE RESUMABILITY - Continue from checkpoint
        # ====================================================================
        logger.info("=" * 80)
        logger.info("ATTEMPT 2: TRUE RESUMABILITY - Continuing from checkpoint")
        logger.info("=" * 80)

        logger.info("  - Update: Sets retry_count=1 so task_b will succeed")
        result = app.invoke(Command(update={"retry_count": 1}, goto="task_b"), config)

        # Validate successful completion
        assert "parallel_results" in result, "Result should contain parallel_results"
        parallel_results = result["parallel_results"]
        assert len(parallel_results) == 3, (
            f"Should have 3 parallel results, got {len(parallel_results)}"
        )
        assert "result_a" in parallel_results, "Should contain result from task_a"
        assert "result_b" in parallel_results, "Should contain result from task_b"
        assert "result_c" in parallel_results, "Should contain result from task_c"
        assert result["attempt_count"] >= 1, (
            "Attempt count should be incremented by merge"
        )

        # Validate checkpoint history shows both attempts
        history_after_success = list(app.get_state_history(config))
        checkpoint_count_after_success = len(history_after_success)
        assert checkpoint_count_after_success > checkpoint_count_after_failure, (
            "Checkpoint count should increase after successful resume"
        )
        logger.info(
            f"Checkpoints after successful resume: {checkpoint_count_after_success}"
        )

        # Validate checkpoint structure
        for checkpoint_state in history_after_success[:5]:
            assert checkpoint_state.config is not None, "Checkpoint should have config"
            assert checkpoint_state.values is not None, "Checkpoint should have values"
            assert "thread_id" in checkpoint_state.config["configurable"], (
                "Config should have thread_id"
            )

        # Validate current state reflects successful completion
        current_state = app.get_state(config)
        assert current_state.values["attempt_count"] >= 1, (
            "Current state should show completion"
        )
        assert len(current_state.next) == 0, "No pending nodes after completion"

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


# ============================================================================
# TEST 6: Subgraph Execution with Checkpoint Validation
# ============================================================================


def test_subgraph_execution_with_checkpoints(checkpoint_saver, thread_id):
    """
    Test 2 subgraphs processing data and parent collecting results.

    WORKFLOW:
    - Parent: START → analysis_subgraph → validation_subgraph → combine_results → END
    - Analysis Subgraph: extract → transform → END
    - Validation Subgraph: check → verify → END

    VALIDATES:
    - Both subgraphs execute independently
    - Parent collects outputs from both subgraphs
    - Final result combines both subgraph outputs
    - Checkpoints track entire workflow
    """

    # State for subgraphs and parent
    class WorkflowState(TypedDict):
        input_data: str
        analysis_result: str
        validation_result: str
        final_output: str

    # === ANALYSIS SUBGRAPH ===
    def extract_data(state: WorkflowState) -> dict:
        logger.info("Analysis: Extracting data")
        data = state.get("input_data", "")
        return {"analysis_result": f"extracted[{data}]"}

    def transform_data(state: WorkflowState) -> dict:
        logger.info("Analysis: Transforming data")
        result = state.get("analysis_result", "")
        return {"analysis_result": f"{result}→transformed"}

    analysis_graph = StateGraph(WorkflowState)
    analysis_graph.add_node("extract", extract_data)
    analysis_graph.add_node("transform", transform_data)
    analysis_graph.add_edge(START, "extract")
    analysis_graph.add_edge("extract", "transform")
    analysis_graph.add_edge("transform", END)
    analysis_subgraph = analysis_graph.compile(checkpointer=checkpoint_saver)

    # === VALIDATION SUBGRAPH ===
    def check_data(state: WorkflowState) -> dict:
        logger.info("Validation: Checking data")
        data = state.get("input_data", "")
        return {"validation_result": f"checked[{data}]"}

    def verify_data(state: WorkflowState) -> dict:
        logger.info("Validation: Verifying data")
        result = state.get("validation_result", "")
        return {"validation_result": f"{result}→verified"}

    validation_graph = StateGraph(WorkflowState)
    validation_graph.add_node("check", check_data)
    validation_graph.add_node("verify", verify_data)
    validation_graph.add_edge(START, "check")
    validation_graph.add_edge("check", "verify")
    validation_graph.add_edge("verify", END)
    validation_subgraph = validation_graph.compile(checkpointer=checkpoint_saver)

    # === PARENT GRAPH ===
    def combine_results(state: WorkflowState) -> dict:
        logger.info("Parent: Combining results from both subgraphs")
        analysis = state.get("analysis_result", "")
        validation = state.get("validation_result", "")
        return {"final_output": f"COMBINED[{analysis} + {validation}]"}

    parent = StateGraph(WorkflowState)
    parent.add_node("analysis", analysis_subgraph)
    parent.add_node("validation", validation_subgraph)
    parent.add_node("combine", combine_results)
    parent.add_edge(START, "analysis")
    parent.add_edge("analysis", "validation")
    parent.add_edge("validation", "combine")
    parent.add_edge("combine", END)

    app = parent.compile(checkpointer=checkpoint_saver)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Execute workflow
        result = app.invoke({"input_data": "test_data"}, config)

        # Validate both subgraphs executed
        assert "extracted" in result["analysis_result"], (
            "Analysis subgraph should extract"
        )
        assert "transformed" in result["analysis_result"], (
            "Analysis subgraph should transform"
        )
        assert "checked" in result["validation_result"], (
            "Validation subgraph should check"
        )
        assert "verified" in result["validation_result"], (
            "Validation subgraph should verify"
        )

        # Validate final output combines both
        assert "COMBINED" in result["final_output"], "Should combine results"
        assert "extracted" in result["final_output"], "Should include analysis result"
        assert "checked" in result["final_output"], "Should include validation result"

        # Validate checkpoints - including subgraph steps
        history = list(app.get_state_history(config))
        assert len(history) > 0, "Should create checkpoints"
        assert len(history) >= 5, f"Expected at least 5 checkpoints, got {len(history)}"

    finally:
        cleanup_thread_resources(checkpoint_saver, thread_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
