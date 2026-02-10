# DynamoDB Store for LangGraph

A DynamoDB-backed store implementation for LangGraph that provides persistent key-value storage with hierarchical namespaces.

## Features

- **Persistent Storage**: Durable storage using AWS DynamoDB
- **Hierarchical Namespaces**: Organize data with multi-level namespaces
- **TTL Support**: Automatic item expiration with configurable time-to-live
- **Filtering**: Basic filtering capabilities for search operations (equality checks)
- **Batch Operations**: Efficient batch processing of multiple operations
- **Async Support**: Full async/sync parity via `abatch`, `aput`, `aget`, `asearch`, `alist_namespaces`
- **Cost-Effective**: Pay-per-request billing for unpredictable workloads

## Installation

```bash
pip install langgraph-checkpoint-aws
```

## Quick Start

```python
from langgraph_checkpoint_aws import DynamoDBStore

# Create a store instance
store = DynamoDBStore(table_name="my-store-table", region_name="us-east-1")

# Setup the table (creates it if it doesn't exist)
store.setup()

# Store and retrieve data
store.put(("users", "123"), "prefs", {"theme": "dark"})
item = store.get(("users", "123"), "prefs")
print(item.value)  # {"theme": "dark"}
```

## Basic Usage

### Storing Documents

```python
# Store a document with hierarchical namespace
store.put(
    ("documents", "user123"),
    "report_1",
    {
        "text": "Machine learning report on customer behavior analysis...",
        "tags": ["ml", "analytics", "report"],
        "author": "data_scientist"
    }
)
```

### Retrieving Documents

```python
# Get a specific document
item = store.get(("documents", "user123"), "report_1")
print(f"Text: {item.value['text']}")
print(f"Created: {item.created_at}")
print(f"Updated: {item.updated_at}")
```

### Searching

```python
# Search all documents in a namespace
results = store.search(("documents", "user123"))

# Search with filter (equality match on value fields)
results = store.search(
    ("documents", "user123"),
    filter={"author": "data_scientist"}
)
```

### Deleting Items

```python
store.delete(("documents", "user123"), "report_1")
```

## Advanced Features

### Time-To-Live (TTL)

Configure automatic item expiration:

```python
store = DynamoDBStore(
    table_name="my-store-table",
    region_name="us-east-1",
    ttl={
        "default_ttl": 60,  # 60 minutes default TTL
        "refresh_on_read": True,  # Refresh TTL on reads
    }
)
store.setup()

# Item will expire after 60 minutes
store.put(("temp", "session_123"), "data", {"value": "temporary data"})

# Custom TTL for specific item (30 minutes)
store.put(
    ("temp", "session_123"),
    "short_lived",
    {"value": "expires soon"},
    ttl=30
)
```

### Async Operations

All operations have async counterparts with full sync/async parity:

```python
import asyncio

async def main():
    store = DynamoDBStore(table_name="my-store-table", region_name="us-east-1")
    store.setup()

    # Async put and get
    await store.aput(("users", "123"), "prefs", {"theme": "dark"})
    item = await store.aget(("users", "123"), "prefs")

    # Async search
    results = await store.asearch(("users", "123"), limit=10)

    # Async list namespaces
    namespaces = await store.alist_namespaces(prefix=("users",))

asyncio.run(main())
```

### Listing Namespaces

```python
# List all namespaces
namespaces = store.list_namespaces()

# List with prefix filter
user_namespaces = store.list_namespaces(prefix=("users",))

# Limit depth
shallow_namespaces = store.list_namespaces(max_depth=2)
```

### Batch Operations

```python
from langgraph.store.base import PutOp, GetOp, SearchOp

# Batch put operations
ops = [
    PutOp(namespace=("batch",), key="item1", value={"value": 1}),
    PutOp(namespace=("batch",), key="item2", value={"value": 2}),
    PutOp(namespace=("batch",), key="item3", value={"value": 3}),
]
results = store.batch(ops)

# Batch get operations
get_ops = [
    GetOp(namespace=("batch",), key="item1"),
    GetOp(namespace=("batch",), key="item2"),
]
items = store.batch(get_ops)

# Async batch (operations execute in parallel via thread pool)
items = await store.abatch(get_ops)
```

### Context Manager

```python
with DynamoDBStore.from_conn_string(
    "my-store-table",
    region_name="us-east-1",
) as store:
    store.setup()
    store.put(("test",), "example", {"data": "value"})
    item = store.get(("test",), "example")
```

### Specifying AWS Region

```python
# Option 1: Explicit region_name
store = DynamoDBStore(table_name="my-store", region_name="us-east-1")
store.setup()

# Option 2: Using boto3 session
import boto3
session = boto3.Session(region_name="us-west-2")
store = DynamoDBStore(table_name="my-store", boto3_session=session)
store.setup()

# Option 3: Environment variable (set before running)
# export AWS_DEFAULT_REGION=us-east-1
store = DynamoDBStore(table_name="my-store")
store.setup()

# Option 4: Custom endpoint URL (e.g., for DynamoDB Local)
store = DynamoDBStore(
    table_name="my-store",
    region_name="us-east-1",
    endpoint_url="http://localhost:8000",
)
store.setup()
```

## Configuration Options

### Constructor Parameters

| Parameter                  | Type                   | Required | Default | Description                                                                                          |
| -------------------------- | ---------------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `table_name`               | str                    | Yes      | -       | Name of the DynamoDB table                                                                           |
| `region_name`              | str                    | No       | None    | AWS region name. Either this, `boto3_session`, or AWS region environment variables must be provided. |
| `boto3_session`            | boto3.Session          | No       | None    | Custom boto3 session                                                                                 |
| `endpoint_url`             | str                    | No       | None    | Custom endpoint URL for the DynamoDB service (e.g., `http://localhost:8000` for DynamoDB Local)      |
| `boto_config`              | botocore.config.Config | No       | None    | Botocore config object for advanced configuration (timeouts, retries, etc.)                          |
| `ttl`                      | TTLConfig              | No       | None    | TTL configuration for automatic item expiration                                                      |
| `max_read_capacity_units`  | int                    | No       | 10      | Max read capacity units for on-demand mode. Only used when creating a new table.                     |
| `max_write_capacity_units` | int                    | No       | 10      | Max write capacity units for on-demand mode. Only used when creating a new table.                    |

### TTL Configuration

```python
ttl = {
    "default_ttl": 60,  # Default TTL in minutes
    "refresh_on_read": True,  # Refresh TTL when items are read
}
```

## Error Handling

The store defines custom exception classes for specific error conditions:

| Exception                 | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| `DynamoDBStoreError`      | Base exception for all DynamoDB store errors                              |
| `DynamoDBConnectionError` | Raised when DynamoDB client initialization fails                          |
| `ValidationError`         | Raised when constructor parameter validation fails (e.g., missing region) |
| `TableCreationError`      | Raised when table creation or setup fails                                 |

```python
from langgraph_checkpoint_aws.store.dynamodb.exceptions import (
    DynamoDBStoreError,
    DynamoDBConnectionError,
    ValidationError,
    TableCreationError,
)

try:
    store = DynamoDBStore(table_name="my-table")
except ValidationError:
    print("Region configuration is missing")

try:
    store.setup()
except TableCreationError:
    print("Failed to create DynamoDB table")
```

## DynamoDB Table Schema

The store uses a single DynamoDB table with the following structure:

- **PK** (Partition Key, String): Namespace joined with `:`
- **SK** (Sort Key, String): Item key
- **value** (Map): The stored dictionary (serialized via boto3 TypeSerializer)
- **created_at** (String): ISO format timestamp (UTC, preserved on updates)
- **updated_at** (String): ISO format timestamp (UTC, updated on each write)
- **expires_at** (Number, optional): Unix timestamp in seconds for TTL

Billing mode: `PAY_PER_REQUEST` (on-demand).

## AWS Configuration

**Required**: You must provide AWS region configuration through one of:

1. **Explicit parameter**: Pass `region_name` or `boto3_session` to constructor
2. **Environment variables**: Set `AWS_DEFAULT_REGION` or `AWS_REGION`
3. **AWS config file**: Configure region in `~/.aws/config`

Additionally, ensure you have proper AWS credentials configured through:

- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM role when running on AWS services

Required IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:CreateTable",
        "dynamodb:DescribeTable",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:DeleteItem",
        "dynamodb:UpdateItem",
        "dynamodb:UpdateTimeToLive"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/your-table-name"
    }
  ]
}
```

## Comparison with Other Stores

### DynamoDB Store vs Valkey Store

| Feature       | DynamoDB Store                    | Valkey Store                          |
| ------------- | --------------------------------- | ------------------------------------- |
| Vector Search | No                                | Yes                                   |
| Performance   | Good                              | Excellent                             |
| Async Support | Yes (thread pool)                 | Yes                                   |
| Best For      | Cost-effective persistent storage | Vector search, ultra-high performance |

Use **DynamoDB Store** when:

- You don't require vector search capabilities
- You want cost-effective, fully managed storage

Use **Valkey Store** when:

- You need vector search capabilities
- You require ultra-low latency

## Limitations

- **No Vector Search**: This store does not support semantic/vector search
- **Scan Cost**: Listing namespaces uses DynamoDB Scan which can be expensive for large tables
- **Filter Limitations**: Client-side filtering only (equality checks on value fields)
- **No Transactions**: Operations are not transactional across multiple items
- **Item Size Limit**: DynamoDB has a [400KB limit per item](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-use-s3-too.html). For larger items, consider using S3 or splitting data.

## Examples

See the [example notebook](/samples/memory/dynamodb_store.ipynb) for comprehensive usage examples.

## Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [BaseStore Interface](https://langchain-ai.github.io/langgraph/reference/store/)
