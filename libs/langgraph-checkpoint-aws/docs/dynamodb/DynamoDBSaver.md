# LangGraph DynamoDB

A LangGraph checkpointer that persists agent state to Amazon DynamoDB with automatic S3 offloading for large payloads.

## Features

- **Unified Storage** - Single DynamoDB table for all checkpoint data
- **Automatic S3 Offloading** - Seamlessly stores large checkpoints (>350KB) in S3
- **TTL Support** - Automatic cleanup via DynamoDB TTL and S3 lifecycle policies
- **Smart Compression** - Optional gzip compression with intelligent thresholds
- **Flexible Configuration** - Custom AWS clients, sessions, and endpoints

## Understanding Automatic Cleanup

### DynamoDB Time To Live (TTL)

[DynamoDB TTL](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html) is a cost-effective mechanism for automatically deleting items that are no longer needed. When you enable TTL on the checkpoint table:

- DynamoDB automatically deletes expired items **within a few days** of expiration (not immediately)
- **Zero write throughput is consumed** for TTL deletions

When you set `ttl_seconds` in DynamoDBSaver, each checkpoint is automatically tagged with an expiration time. For example, `ttl_seconds=86400 * 7` means checkpoints expire 7 days after creation (86400 seconds = 1 day).

### S3 Lifecycle Policies

[S3 Lifecycle](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html) manages object storage throughout their lifecycle. When using S3 offloading with TTL:

- **Expiration actions** queue objects for deletion and remove them asynchronously
- The library automatically configures lifecycle rules when you set `ttl_seconds`
- Lifecycle rules apply to both existing objects and new objects added later
- Billing stops as soon as objects become eligible for expiration (even if not yet deleted)

**Important:** S3 Lifecycle uses day-level precision for expiration rules. When DynamoDBSaver configures S3 lifecycle policies, it converts `ttl_seconds` to days. Objects expire after the specified number of days from creation.

DynamoDBSaver synchronizes TTL settings between DynamoDB and S3, ensuring consistent cleanup across both storage layers. You don't need to manually configure S3 lifecycle policies - just set `ttl_seconds` and both services handle cleanup automatically.

## Installation

```bash
pip install langgraph-checkpoint-aws
```

## Quick Start

```python
from langgraph_checkpoint_aws.checkpoint.dynamodb import DynamoDBSaver
from langgraph.graph import StateGraph

# Initialize checkpointer
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-east-1"
)

# Use with LangGraph
graph = StateGraph(state_schema)
# ... define your graph ...
app = graph.compile(checkpointer=checkpointer)

# Run with persistence
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(input_data, config)
```

## Setup AWS Resources

### CloudFormation (Recommended)

The provided [CloudFormation template](../../../../samples/memory/cfn/langgraph-ddb-cfn-template.yaml) is a sample - customize it for your requirements (encryption, backups, tags, etc.).

```bash
# Deploy with S3 offloading and TTL (auto-generated bucket name)
aws cloudformation create-stack \
  --stack-name langgraph-checkpoints \
  --template-body file://langgraph-ddb-cfn-template.yaml \
  --parameters \
    ParameterKey=CheckpointTableName,ParameterValue=my-checkpoints \
    ParameterKey=CreateS3Bucket,ParameterValue=true \
    ParameterKey=EnableTTL,ParameterValue=true

# Get created resource names
aws cloudformation describe-stacks \
  --stack-name langgraph-checkpoints \
  --query 'Stacks[0].Outputs[?OutputKey==`CheckpointTableName`].OutputValue' \
  --output text

aws cloudformation describe-stacks \
  --stack-name langgraph-checkpoints \
  --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
  --output text
```

**Parameters:**

| Parameter             | Default                          | Description                                         |
| --------------------- | -------------------------------- | --------------------------------------------------- |
| `CheckpointTableName` | `langgraph-checkpoints-dynamodb` | DynamoDB table name                                 |
| `EnableTTL`           | `false`                          | Enable automatic TTL-based expiration               |
| `CreateS3Bucket`      | `false`                          | Create S3 bucket for offloading                     |
| `S3BucketName`        | `` (empty)                       | S3 bucket name (empty = auto-generated random name) |

**Note:** S3 lifecycle policies are automatically managed by the library when you set `ttl_seconds` - no manual configuration needed.

### IAM Permissions

**DynamoDB Only:**

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Query",
      "dynamodb:BatchGetItem",
      "dynamodb:BatchWriteItem"
    ],
    "Resource": "arn:aws:dynamodb:<REGION>:<ACCOUNT>:table/<TABLE_NAME>"
  }]
}
```

**With S3 Offloading:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query",
        "dynamodb:BatchGetItem",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": "arn:aws:dynamodb:<REGION>:<ACCOUNT>:table/<TABLE_NAME>"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:PutObjectTagging"
      ],
      "Resource": "arn:aws:s3:::<BUCKET_NAME>/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLifecycleConfiguration",
        "s3:PutBucketLifecycleConfiguration"
      ],
      "Resource": "arn:aws:s3:::<BUCKET_NAME>"
    }
  ]
}
```

**Note:** S3 lifecycle permissions are only needed when using TTL with S3 offloading.

## Configuration

### Common Configurations

**With S3 Offloading:**

```python
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-east-1",
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)
```

**With AWS Profile (Recommended for Security):**

```python
import boto3

# Use specific AWS profile from ~/.aws/credentials
session = boto3.Session(
    profile_name="production",
    region_name="us-east-1"
)

checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    session=session,
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)
```

**With Assumed Role:**

```python
import boto3

# Assume role for cross-account or elevated permissions
sts = boto3.client('sts')
assumed_role = sts.assume_role(
    RoleArn='arn:aws:iam::123456789012:role/CheckpointRole',
    RoleSessionName='langgraph-session'
)

session = boto3.Session(
    aws_access_key_id=assumed_role['Credentials']['AccessKeyId'],
    aws_secret_access_key=assumed_role['Credentials']['SecretAccessKey'],
    aws_session_token=assumed_role['Credentials']['SessionToken'],
    region_name='us-east-1'
)

checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    session=session,
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)
```

**With TTL (7 days):**

```python
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-east-1",
    ttl_seconds=86400 * 7,
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)
```

**With Compression:**

```python
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-east-1",
    enable_checkpoint_compression=True,
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)
```

**Production Recommended (All Features):**

```python
import boto3
from botocore.config import Config

# Use dedicated AWS profile for production
session = boto3.Session(
    profile_name="production",
    region_name="us-east-1"
)

checkpointer = DynamoDBSaver(
    table_name="prod-checkpoints",
    session=session,
    ttl_seconds=86400 * 30,  # 30 days
    boto_config=Config(
        retries={"mode": "adaptive", "max_attempts": 6},
        max_pool_connections=50
    ),
    s3_offload_config={"bucket_name": "prod-checkpoint-bucket"}
)
```

### All Parameters

**Required:**

- `table_name` (str) - DynamoDB checkpoint table name

**AWS Connection:**

- `session` (boto3.Session) - Custom boto3 session
- `region_name` (str) - AWS region (e.g., "us-east-1")
- `endpoint_url` (str) - Custom endpoint (for Dynamodb Local)
- `boto_config` (Config) - Botocore config for retries, timeouts, etc.

**Storage:**

- `ttl_seconds` (int) - Auto-cleanup after N seconds
- `enable_checkpoint_compression` (bool) - Enable gzip compression
- `s3_offload_config` (dict) - S3 configuration:
    - `bucket_name` (str, required) - S3 bucket name
    - `endpoint_url` (str, optional) - Custom S3 endpoint

## Usage

### Thread Management

```python
# Delete all checkpoints for a thread
checkpointer.delete_thread("user-123")

# Async version
await checkpointer.adelete_thread("user-123")
```

### List Checkpoints

```python
config = {"configurable": {"thread_id": "user-123"}}

# List recent checkpoints
for checkpoint_tuple in checkpointer.list(config, limit=10):
    print(f"ID: {checkpoint_tuple.checkpoint['id']}")
    print(f"Metadata: {checkpoint_tuple.metadata}")

# List with filter
for checkpoint_tuple in checkpointer.list(config, filter={"source": "user"}):
    print(checkpoint_tuple)
```

### Get Specific Checkpoint

```python
config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": "1ef4f797-8335-6ace-8001-1b7f24e6d7fa"
    }
}
checkpoint_tuple = checkpointer.get_tuple(config)
```

### Use Namespaces

```python
config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_ns": "conversation_1"
    }
}
result = app.invoke(input_data, config)
```

## Storage Architecture

### Unified Table Design

Single DynamoDB table with intelligent key patterns:

- **Checkpoint Metadata**: `PK=CHECKPOINT_{thread_id}`,
  `SK={checkpoint_ns}#{checkpoint_id}`
- **Checkpoint Payloads**: `PK=CHUNK_{thread_id}#{checkpoint_ns}#{checkpoint_id}`,
  `SK=CHUNK`
- **Write Metadata**: `PK=WRITES_{thread_id}#{checkpoint_ns}#{checkpoint_id}`,
  `SK={task_id}#{idx}`
- **Write Payloads**:
  `PK=CHUNK_{thread_id}#{checkpoint_ns}#{checkpoint_id}#{task_id}#{idx}`,
  `SK=CHUNK`

### Storage Decision

Checkpoints <350KB → DynamoDB  
Checkpoints ≥350KB → S3 (if configured) or DynamoDB

### Compression

When enabled, compression only applies if:

- Data size >1KB
- Compression saves ≥10% space
- Uses gzip level 6

## Development

```bash
# Install dependencies
uv sync

# Run tests
make test
make integration_test  # requires AWS credentials

# Code quality
make format
make lint
```
