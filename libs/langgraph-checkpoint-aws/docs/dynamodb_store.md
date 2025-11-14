# DynamoDB Store for LangGraph

A DynamoDB-backed store implementation for LangGraph that provides persistent key-value storage with hierarchical namespaces.

## Features

- ✅ **Persistent Storage**: Durable storage using AWS DynamoDB
- ✅ **Hierarchical Namespaces**: Organize data with multi-level namespaces
- ✅ **TTL Support**: Automatic item expiration with configurable time-to-live
- ✅ **Filtering**: Basic filtering capabilities for search operations
- ✅ **Batch Operations**: Efficient batch processing of multiple operations
- ✅ **Cost-Effective**: Pay-per-request billing for unpredictable workloads

## Installation

```bash
pip install langgraph-checkpoint-aws
```

## Quick Start

```python
from langgraph_checkpoint_aws import DynamoDBStore

# Create a store instance
store = DynamoDBStore(table_name="my-store-table")

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

# Search with filter
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
from langgraph.store.base import PutOp, GetOp

# Batch put operations
ops = [
    PutOp(("batch",), "item1", {"value": 1}, None, None),
    PutOp(("batch",), "item2", {"value": 2}, None, None),
    PutOp(("batch",), "item3", {"value": 3}, None, None),
]
results = store.batch(ops)

# Batch get operations
get_ops = [
    GetOp(("batch",), "item1", False),
    GetOp(("batch",), "item2", False),
]
items = store.batch(get_ops)
```

### Context Manager

```python
with DynamoDBStore.from_conn_string("my-store-table") as store:
    store.setup()
    store.put(("test",), "example", {"data": "value"})
    item = store.get(("test",), "example")
```

## Configuration Options

### Constructor Parameters

- `table_name` (str): Name of the DynamoDB table
- `region_name` (str, optional): AWS region name
- `boto3_session` (boto3.Session, optional): Custom boto3 session
- `ttl` (TTLConfig, optional): TTL configuration
- `max_read_capacity_units` (int, optional): Max read capacity (default: 10)
- `max_write_capacity_units` (int, optional): Max write capacity (default: 10)

### TTL Configuration

```python
ttl = {
    "default_ttl": 60,  # Default TTL in minutes
    "refresh_on_read": True,  # Refresh TTL when items are read
}
```

## DynamoDB Table Schema

The store uses a single DynamoDB table with the following structure:

- **PK** (Partition Key, String): Namespace joined with ':'
- **SK** (Sort Key, String): Item key
- **value** (Map): The stored dictionary
- **created_at** (String): ISO format timestamp
- **updated_at** (String): ISO format timestamp
- **expires_at** (Number, optional): Unix timestamp for TTL

## AWS Configuration

Ensure you have proper AWS credentials configured through:

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

| Feature | DynamoDB Store | Valkey Store |
|---------|---------------|--------------|
| Vector Search | ❌ No | ✅ Yes |
| High Performance | ✅ Good | ✅ Excellent |
| TTL Support | ✅ Yes | ✅ Yes |
| Cost | Pay-per-request | Infrastructure cost |
| Best For | Simple storage, managed infra | Vector search, high performance |

Use **DynamoDB Store** when:
- You need a fully managed solution
- You don't require vector search capabilities
- You want pay-per-request pricing
- Your workload is unpredictable

Use **Valkey Store** when:
- You need vector search capabilities
- You require ultra-low latency
- You can manage your own infrastructure
- You have consistent, predictable workloads

## Limitations

- **No Vector Search**: This store does not support semantic/vector search
- **Scan Cost**: Listing namespaces uses DynamoDB Scan which can be expensive
- **Filter Limitations**: Basic filtering only (equality checks)
- **No Transactions**: Operations are not transactional across multiple items

## Examples

See the [example notebook](../../samples/memory/dynamodb_store.ipynb) for comprehensive usage examples.

## Contributing

Contributions are welcome! Please see the main [CONTRIBUTING.md](../../libs/langgraph-checkpoint-aws/CONTRIBUTING.md) for guidelines.

## License

This package is part of the `langgraph-checkpoint-aws` project. See [LICENSE](../../LICENSE) for details.

## Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [BaseStore Interface](https://langchain-ai.github.io/langgraph/reference/store/)
