# LangGraph Checkpoint AWS
A custom AWS-based persistence solution for LangGraph agents that provides multiple storage backends including Bedrock AgentCore Memory, DynamoDB with S3 offloading, and high-performance Valkey (Redis-compatible) storage.

## Overview
This package provides multiple persistence solutions for LangGraph agents:

### AWS Bedrock AgentCore Memory Service
1. Stateful conversations and interactions
2. Resumable agent sessions
3. Efficient state persistence and retrieval
4. Seamless integration with AWS Bedrock

### DynamoDB Storage
1. **Checkpoint storage** with DynamoDB and automatic S3 offloading
2. Unified table design with TTL support
3. Intelligent compression for optimal storage

### Valkey Storage Solutions
1. **Checkpoint storage** with Valkey (Redis-compatible)
2. **Intelligent caching** for LLM responses and computation results
3. **Document storage** with vector search capabilities

## Installation

You can install the package using pip:

```bash
# Base package (includes Bedrock AgentCore Memory components)
pip install langgraph-checkpoint-aws

# Optional Valkey support
pip install 'langgraph-checkpoint-aws[valkey]'

```

## Components

This package provides following main components:

1. **AgentCoreMemorySaver** - AWS Bedrock-based checkpoint storage
2. **AgentCoreValkeySaver** - AgentCore-compatible Valkey checkpoint storage
3. **DynamoDBSaver** - DynamoDB-based checkpoint storage with S3 offloading
4. **ValkeySaver** - Valkey checkpoint storage
5. **AgentCoreMemoryStore** - AWS Bedrock-based document store
6. **ValkeyStore** - Valkey document store
7. **ValkeyCache** - Valkey LLM response cache

## Usage

### 1. Bedrock Session Management

```python
# Import LangGraph and LangChain components
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# Import the AgentCoreMemory integrations
from langgraph_checkpoint_aws import AgentCoreMemorySaver

REGION = "us-west-2"
MEMORY_ID = "YOUR_MEMORY_ID"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Initialize checkpointer for state persistence. No additional setup required.
# Sessions will be saved and persisted for actor_id/session_id combinations
checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)

# Initialize chat model
model = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer, # AgentCoreMemorySaver we created above
)

# Specify config at runtime for ACTOR and SESSION
config = {
    "configurable": {
        "thread_id": "session-1", # REQUIRED: This maps to Bedrock AgentCore session_id under the hood
        "actor_id": "react-agent-1", # REQUIRED: This maps to Bedrock AgentCore actor_id under the hood
    }
}

# Invoke the agent
response = graph.invoke(
    {"messages": [("human", "I like sushi with tuna. In general seafood is great.")]},
    config=config
)
```

### 2. Bedrock Memory Store

```python
# Import LangGraph and LangChain components
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langgraph_checkpoint_aws import (
    AgentCoreMemoryStore
)

REGION = "us-west-2"
MEMORY_ID = "YOUR_MEMORY_ID"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Initialize store for saving and searching over long term memories
# such as preferences and facts across sessions
store = AgentCoreMemoryStore(MEMORY_ID, region_name=REGION)

# Pre-model hook runs and saves messages of your choosing to AgentCore Memory
# for async processing and extraction
def pre_model_hook(state, config: RunnableConfig, *, store: BaseStore):
    """Hook that runs pre-model invocation to save the latest human message"""
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    
    # Saving the message to the actor and session combination that we get at runtime
    namespace = (actor_id, thread_id)
    
    messages = state.get("messages", [])
    # Save the last human message we see before model invocation
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            store.put(namespace, str(uuid.uuid4()), {"message": msg})
            break
            
    # OPTIONAL: Retrieve user preferences based on the last message and append to state
    # user_preferences_namespace = ("preferences", actor_id)
    # preferences = store.search(user_preferences_namespace, query=msg.content, limit=5)
    # # Add to input messages as needed
    
    return {"model_input_messages": messages}

# Initialize chat model
model = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=model,
    tools=[],
    pre_model_hook=pre_model_hook,
)

# Specify config at runtime for ACTOR and SESSION
config = {
    "configurable": {
        "thread_id": "session-1", # REQUIRED: This maps to Bedrock AgentCore session_id under the hood
        "actor_id": "react-agent-1", # REQUIRED: This maps to Bedrock AgentCore actor_id under the hood
    }
}

# Invoke the agent
response = graph.invoke(
    {"messages": [("human", "I like sushi with tuna. In general seafood is great.")]},
    config=config
)
```
### 3. Valkey Cache - LLM Response caching
Intelligent caching to improve performance and reduce costs:

```python
from langgraph_checkpoint_aws import ValkeyCache
from langchain_aws import ChatBedrockConverse

# Initialize cache
with ValkeyCache.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as cache:
    
    # Use cache with your LLM calls
    model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )   
    
    # Cache expensive prompts/computations
    cache_key = "expensive_computation_key"
    result = cache.get([cache_key])
    
    if not result:
        # Compute and cache result
        prompt: str = "Your expensive prompt"
        response = model.invoke([HumanMessage(content=prompt)])
        cache.set({cache_key: (response.content, 3600)})  # Cache for 1 hour
```
### 4. DynamoDB Checkpoint Storage

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws import DynamoDBSaver

# Basic usage with DynamoDB only
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-west-2"
)

# With S3 offloading for large checkpoints (>350KB)
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-west-2",
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)

# Production configuration with TTL and compression
checkpointer = DynamoDBSaver(
    table_name="my-checkpoints",
    region_name="us-west-2",
    ttl_seconds=86400 * 7,  # 7 days
    enable_checkpoint_compression=True,
    s3_offload_config={"bucket_name": "my-checkpoint-bucket"}
)

# Create your graph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(1, config)
```

### 5. Valkey Checkpoint Storage

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws import ValkeySaver

# Using connection string
with ValkeySaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as checkpointer:
    # Create your graph
    builder = StateGraph(int)
    builder.add_node("add_one", lambda x: x + 1)
    builder.set_entry_point("add_one")
    builder.set_finish_point("add_one")
    
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-1"}}
    result = graph.invoke(1, config)
```

### 6. AgentCore Valkey Storage

For AWS Bedrock AgentCore-compatible applications that want to use Valkey instead of managed Memory:

```python
from langgraph.prebuilt import create_react_agent
from langgraph_checkpoint_aws import AgentCoreValkeySaver

# Using connection string
with AgentCoreValkeySaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as checkpointer:
    # Create your agent
    graph = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer
    )

    # AgentCore-style configuration (requires actor_id)
    config = {
        "configurable": {
            "thread_id": "session-123",
            "actor_id": "agent-456",  # Required for AgentCore compatibility
            "checkpoint_ns": "production"
        }
    }

    result = graph.invoke({"messages": [...]}, config)
```

**Key Differences from ValkeySaver:**
- ✅ Requires `actor_id` in configuration (AgentCore requirement)
- ✅ Uses AgentCore-compatible key structure (`agentcore:*` prefix)
- ✅ Built-in retry logic with exponential backoff
- ✅ Pydantic validation for data integrity
- ⚠️ **Data is NOT compatible with ValkeySaver** - choose one at project start

### 7. Valkey Store for Document Storage

Document storage with vector search capabilities using ValkeyIndexConfig:

```python
from langchain_aws import BedrockEmbeddings
from langgraph_checkpoint_aws import ValkeyStore
# Initialize Bedrock embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=AWS_REGION
)
# Basic usage with ValkeyIndexConfig
with ValkeyStore.from_conn_string(
    "valkey://localhost:6379",
    index={
        "collection_name": "my_documents",
        "dims": 1536,
        "embed": embeddings,
        "fields": ["text", "author"],
        "timezone": "UTC",
        "index_type": "hnsw"
    },
    ttl={"default_ttl": 60.0}  # 1 hour TTL
) as store:
    
    # Setup vector search index
    store.setup()
    
    # Store documents
    store.put(
        ("documents", "user123"),
        "report_1",
        {
            "text": "Machine learning report on customer behavior analysis...",
            "tags": ["ml", "analytics", "report"],
            "author": "data_scientist"
        }
    )
    
    # Search documents
    results = store.search(
        ("documents",),
        query="machine learning customer analysis",
        filter={"author": "data_scientist"},
        limit=10
    )

# Advanced HNSW configuration for performance tuning
with ValkeyStore.from_conn_string(
    "valkey://localhost:6379",
    index={
        "collection_name": "high_performance_docs",
        "dims": 768,
        "embed": embeddings,
        "fields": ["text", "title", "summary"],
        "timezone": "America/New_York",
        "index_type": "hnsw",
        "hnsw_m": 32,  # More connections for better recall
        "hnsw_ef_construction": 400,  # Higher construction quality
        "hnsw_ef_runtime": 20,  # Better search accuracy
    }
) as store:
    # Optimized for high-accuracy vector search
    pass

# FLAT index for exact search (smaller datasets)
with ValkeyStore.from_conn_string(
    "valkey://localhost:6379",
    index={
        "collection_name": "exact_search_docs",
        "dims": 384,
        "embed": embeddings,
        "fields": ["text"],
        "index_type": "flat"  # Exact search, no approximation
    }
) as store:
    # Exact vector search for smaller datasets
    pass
```

## Async Usage

All components support async operations:

```python
from langgraph_checkpoint_aws.async_saver import AsyncBedrockSessionSaver
from langgraph_checkpoint_aws import AsyncValkeySaver
from langgraph_checkpoint_aws import DynamoDBSaver

# Async Bedrock usage
session_saver = AsyncBedrockSessionSaver(region_name="us-west-2")
session_id = (await session_saver.session_client.create_session()).session_id

# Async DynamoDB usage
checkpointer = DynamoDBSaver(table_name="my-checkpoints", region_name="us-west-2")
result = await graph.ainvoke(1, {"configurable": {"thread_id": "session-1"}})

# Async ValkeySaver usage
async with AsyncValkeySaver.from_conn_string("valkey://localhost:6379") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(1, {"configurable": {"thread_id": "session-1"}})

# Async ValkeyStore usage
async with AsyncValkeyStore.from_conn_string("valkey://localhost:6379") as store:
    namespace = ("example",)
    key = "key"
    data = {
        "message": "Sample message",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    await store.setup()
    await store.aput(namespace, key, data)
    result = await store.aget(namespace, key)
```

## Configuration Options

### Bedrock Session Saver

`BedrockSessionSaver` and `AsyncBedrockSessionSaver` accept the following parameters:

```python
def __init__(
    client: Optional[Any] = None,
    session: Optional[boto3.Session] = None,
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[SecretStr] = None,
    aws_secret_access_key: Optional[SecretStr] = None,
    aws_session_token: Optional[SecretStr] = None,
    endpoint_url: Optional[str] = None,
    config: Optional[Config] = None,
)
```

### DynamoDB Saver

`DynamoDBSaver` provides persistent checkpoint storage with these options:

```python
DynamoDBSaver(
    table_name: str,  # Required: DynamoDB table name
    session: Optional[boto3.Session] = None,  # Custom boto3 session
    region_name: Optional[str] = None,  # AWS region
    endpoint_url: Optional[str] = None,  # Custom dynamodb endpoint url
    boto_config: Optional[Config] = None,  # Botocore config
    ttl_seconds: Optional[int] = None,  # Auto-cleanup after N seconds
    enable_checkpoint_compression: bool = False,  # Enable gzip compression
    s3_offload_config: Optional[dict] = None  # S3 config for large checkpoints
)
```

**Key Features:**
- Unified table design for checkpoints and writes
- Automatic S3 offloading for payloads >350KB (when configured)
- Optional gzip compression with intelligent thresholds
- TTL support with automatic DynamoDB and S3 lifecycle management

**S3 Offload Configuration:**
```python
s3_offload_config = {
    "bucket_name": "my-checkpoint-bucket",  # Required
    "endpoint_url": "http://localhost:4566"  # Optional: Custom s3 endpoint url
}
```

### Valkey Components

Valkey components support these common configuration options:

#### Connection Options
- **Connection String**: `valkey://localhost:6379` or `valkeys://localhost:6380` (SSL). Refer [connection examples](https://valkey-py.readthedocs.io/en/latest/examples/connection_examples.html).
- **Connection Pool**: Reusable connection pools for better performance
- **Pool Size**: Maximum number of connections (default: 10)
- **SSL Support**: Secure connections with certificate validation

#### Performance Options
- **TTL (Time-to-Live)**: Automatic expiration of stored data
- **Batch Operations**: Efficient bulk operations for better throughput
- **Async Support**: Non-blocking operations for high concurrency

#### ValkeyCache Options
```python
ValkeyCache(
    client: Valkey,
    prefix: str = "langgraph:cache:",  # Key prefix
    ttl: float | None = None,  # Default TTL in seconds
    serde: SerializerProtocol | None = None
)
```

#### ValkeySaver Options
```python
valkey_client = Valkey.from_url("valkey://localhost:6379")
ValkeySaver(
    client: valkey_client,
    ttl: float | None = None,  # TTL in seconds
    serde: SerializerProtocol | None = None  # Custom serialization
)
```
#### ValkeyStore Options
```python
ValkeyStore(
    client: Valkey,
    index: ValkeyIndexConfig | None = None,  # Valkey-specific vector search configuration
    ttl: TTLConfig | None = None  # TTL configuration
)

# ValkeyIndexConfig - Enhanced vector search configuration
from langgraph_checkpoint_aws.store.valkey import ValkeyIndexConfig

index_config = {
    # Basic configuration
    "collection_name": "my_documents",  # Index collection name
    "dims": 1536,  # Vector dimensions
    "embed": embeddings,  # Embedding model
    "fields": ["text", "content"],  # Fields to index
    
    # Valkey-specific configuration
    "timezone": "UTC",  # Timezone for operations (default: "UTC")
    "index_type": "hnsw",  # Algorithm: "hnsw" or "flat" (default: "hnsw")
    
    # HNSW performance tuning parameters
    "hnsw_m": 16,  # Connections per layer (default: 16)
    "hnsw_ef_construction": 200,  # Construction search width (default: 200)
    "hnsw_ef_runtime": 10,  # Runtime search width (default: 10)
}

# TTL Configuration
ttl_config = {
    "default_ttl": 60.0  # Default TTL in minutes
}
```
##### Algorithm Selection Guide

**HNSW (Hierarchical Navigable Small World)**
- **Best for**: Large datasets (>10K vectors), fast approximate search
- **Trade-off**: Speed vs accuracy - configurable via parameters
- **Use cases**: Real-time search, large document collections, production systems

**FLAT (Brute Force)**
- **Best for**: Small datasets (<10K vectors), exact search requirements
- **Trade-off**: Perfect accuracy but slower on large datasets
- **Use cases**: High-precision requirements, smaller collections, research

#### Performance Tuning Parameters

**hnsw_m (Connections per layer)**
- **Range**: 4-64 (default: 16)
- **Higher values**: Better recall, more memory usage
- **Lower values**: Faster search, less memory, lower recall
- **Recommendation**: 16-32 for most use cases

**hnsw_ef_construction (Construction search width)**
- **Range**: 100-800 (default: 200)
- **Higher values**: Better index quality, slower construction
- **Lower values**: Faster construction, potentially lower quality
- **Recommendation**: 200-400 for production systems

**hnsw_ef_runtime (Query search width)**
- **Range**: 10-500 (default: 10)
- **Higher values**: Better recall, slower queries
- **Lower values**: Faster queries, potentially lower recall
- **Recommendation**: 10-50 depending on speed/accuracy requirements

##### Configuration Examples

```python
# High-speed configuration (prioritize speed)
speed_config = {
    "collection_name": "fast_search",
    "index_type": "hnsw",
    "hnsw_m": 8,  # Fewer connections
    "hnsw_ef_construction": 100,  # Faster construction
    "hnsw_ef_runtime": 10,  # Fast queries
}

# High-accuracy configuration (prioritize recall)
accuracy_config = {
    "collection_name": "precise_search", 
    "index_type": "hnsw",
    "hnsw_m": 32,  # More connections
    "hnsw_ef_construction": 400,  # Better construction
    "hnsw_ef_runtime": 50,  # More thorough search
}

# Balanced configuration (good speed/accuracy trade-off)
balanced_config = {
    "collection_name": "balanced_search",
    "index_type": "hnsw", 
    "hnsw_m": 16,  # Default connections
    "hnsw_ef_construction": 200,  # Default construction
    "hnsw_ef_runtime": 20,  # Moderate search width
}

# Exact search configuration (perfect accuracy)
exact_config = {
    "collection_name": "exact_search",
    "index_type": "flat",  # No HNSW parameters needed
}
```

## Development

Setting Up Development Environment

* Clone the repository:

```bash
git clone <repository-url>
cd libs/aws/langgraph-checkpoint-aws
```

* Install development dependencies:

```bash
make install_all
```

* Or install specific components:

```bash
make install_dev        # Basic development tools
make install_test       # Testing tools
make install_lint       # Linting tools
make install_typing     # Type checking tools
make install_codespell  # Spell checking tools
```

## Running Tests

```bash
make tests         # Run all tests
make test_watch   # Run tests in watch mode

```

## Code Quality

```bash
make lint           # Run linter
make format         # Format code
make spell_check    # Check spelling
```

## Clean Up

```bash
make clean          # Remove all generated files
```

## Infrastructure Setup

### AWS Configuration (for Bedrock components)

Ensure you have AWS credentials configured using one of these methods:

1. Environment variables
2. AWS credentials file (~/.aws/credentials)
3. IAM roles
4. Direct credential injection via constructor parameters

Required AWS permissions for Bedrock Session Management:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockSessionManagement",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:CreateEvent",
                "bedrock-agentcore:ListEvents",
                "bedrock-agentcore:GetEvent",
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}
```

### DynamoDB Setup

#### CloudFormation Template
A sample CloudFormation template is available at [`langgraph-ddb-cfn-template.yaml`](../../samples/memory/cfn/langgraph-ddb-cfn-template.yaml) for quick setup:

```bash
aws cloudformation create-stack \
  --stack-name langgraph-checkpoints \
  --template-body file://langgraph-ddb-cfn-template.yaml \
  --parameters \
    ParameterKey=CheckpointTableName,ParameterValue=my-checkpoints \
    ParameterKey=CreateS3Bucket,ParameterValue=true \
    ParameterKey=EnableTTL,ParameterValue=true
```

#### Required IAM Permissions

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
    "Resource": "arn:aws:dynamodb:REGION:ACCOUNT:table/TABLE_NAME"
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
      "Resource": "arn:aws:dynamodb:REGION:ACCOUNT:table/TABLE_NAME"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:PutObjectTagging"
      ],
      "Resource": "arn:aws:s3:::BUCKET_NAME/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLifecycleConfiguration",
        "s3:PutBucketLifecycleConfiguration"
      ],
      "Resource": "arn:aws:s3:::BUCKET_NAME"
    }
  ]
}
```

## Bedrock Session Saver (Alternative Implementation)

This package also provides an alternative checkpointing solution using AWS Bedrock Session Management Service:

### Usage

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws.saver import BedrockSessionSaver

# Initialize the saver
session_saver = BedrockSessionSaver(
    region_name="us-west-2",  # Your AWS region
    credentials_profile_name="default",  # Optional: AWS credentials profile
)

# Create a session
session_id = session_saver.session_client.create_session().session_id

# Use with LangGraph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=session_saver)
config = {"configurable": {"thread_id": session_id}}
graph.invoke(1, config)
```

You can also invoke the graph asynchronously:

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws.async_saver import AsyncBedrockSessionSaver

# Initialize the saver
session_saver = AsyncBedrockSessionSaver(
    region_name="us-west-2",  # Your AWS region
    credentials_profile_name="default",  # Optional: AWS credentials profile
)

# Create a session
session_create_response = await session_saver.session_client.create_session()
session_id = session_create_response.session_id

# Use with LangGraph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=session_saver)
config = {"configurable": {"thread_id": session_id}}
await graph.ainvoke(1, config)
```

### Configuration Options

`BedrockSessionSaver` and `AsyncBedrockSessionSaver` accepts the following parameters:

```python
def __init__(
    client: Optional[Any] = None,
    session: Optional[boto3.Session] = None,
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[SecretStr] = None,
    aws_secret_access_key: Optional[SecretStr] = None,
    aws_session_token: Optional[SecretStr] = None,
    endpoint_url: Optional[str] = None,
    config: Optional[Config] = None,
)
```

* `client`: boto3 Bedrock runtime client (e.g. boto3.client("bedrock-agent-runtime"))
* `session`: boto3.Session for custom credentials
* `region_name`: AWS region where Bedrock is available
* `credentials_profile_name`: Name of AWS credentials profile to use
* `aws_access_key_id`: AWS access key ID for authentication
* `aws_secret_access_key`: AWS secret access key for authentication
* `aws_session_token`: AWS session token for temporary credentials
* `endpoint_url`: Custom endpoint URL for the Bedrock service
* `config`: Botocore configuration object

### Additional AWS permissions for Session Saver

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Statement1",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateSession",
                "bedrock:GetSession",
                "bedrock:UpdateSession",
                "bedrock:DeleteSession",
                "bedrock:EndSession",
                "bedrock:ListSessions",
                "bedrock:CreateInvocation",
                "bedrock:ListInvocations",
                "bedrock:PutInvocationStep",
                "bedrock:GetInvocationStep",
                "bedrock:ListInvocationSteps"
            ],
            "Resource": ["*"]
        },
        {
            "Sid": "KMSAccess",
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey",
                "kms:DescribeKey"
            ],
            "Resource": "arn:aws:kms:{region}:{account}:key/{kms-key-id}"
        }
    ]
}
```

### Valkey Setup

#### Using AWS ElastiCache for Valkey (Recommended)
```python
# Connect to AWS ElastiCache from host running inside VPC with access to cache
from langgraph_checkpoint_aws.checkpoint.valkey import ValkeySaver

with ValkeySaver.from_conn_string(
    "valkeys://your-elasticache-cluster.amazonaws.com:6379",
    pool_size=20
) as checkpointer:
    pass
```
If you want to connect to cache from a host outside of VPC, use ElastiCache console to setup a jump host so you could create SSH tunnel to access cache locally. 

#### Using Docker
```bash
# Start Valkey with required modules
docker run --name valkey-bundle -p 6379:6379 -d valkey/valkey-bundle:latest

# Or with custom configuration
docker run --name valkey-custom \
  -p 6379:6379 \
  -v $(pwd)/valkey.conf:/etc/valkey/valkey.conf \
  -d valkey/valkey-bundle:latest
```

## Performance and Best Practices

### Valkey Performance Optimization

#### Connection Pooling
```python
# Use connection pools for better performance
from valkey.connection import ConnectionPool

pool = ConnectionPool.from_url(
    "valkey://localhost:6379",
    max_connections=20,
    retry_on_timeout=True
)

with ValkeySaver.from_pool(pool) as checkpointer:
    # Reuse connections across operations
    pass
```

#### TTL Strategy
```python
# Configure appropriate TTL values
with ValkeySaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600  # 1 hour for active sessions
) as checkpointer:
    pass
```
#### Batch Operations
```python
# Use batch operations for better throughput
cache.set({
    "key1": (value1, 3600),
    "key2": (value2, 1800),
    "key3": (value3, 7200)
})

results = cache.get(["key1", "key2", "key3"])
```

## Security Considerations

* Never commit AWS credentials

* Use environment variables or AWS IAM roles for authentication
* Follow AWS security best practices
* Use IAM roles and temporary credentials when possible
* Implement proper access controls for session management

### Valkey Security
* Use SSL/TLS for production deployments (`valkeys://` protocol), refer [SSL connection examples](https://valkey-py.readthedocs.io/en/latest/examples/ssl_connection_examples.html#Connect-to-a-Valkey-instance-via-SSL,-and-validate-OCSP-stapled-certificates)
* Configure authentication with strong passwords
* Implement network security (VPC, security groups)
* Regular security updates and monitoring
* Use AWS ElastiCache for managed Valkey with encryption

```python
# Secure connection example
import os
import valkey

pki_dir = os.path.join("..", "..", "dockers", "stunnel", "keys")

valkey_client = valkey.Valkey(
    host="localhost",
    port=6666,
    ssl=True,
    ssl_certfile=os.path.join(pki_dir, "client-cert.pem"),
    ssl_keyfile=os.path.join(pki_dir, "client-key.pem"),
    ssl_cert_reqs="required",
    ssl_ca_certs=os.path.join(pki_dir, "ca-cert.pem"),
)

checkpointer = ValkeySaver(valkey_client)
```

## Examples and Samples

Comprehensive examples are available in the `samples/memory/` directory:

## Contributing

* Fork the repository

* Create a feature branch
* Make your changes
* Run tests and linting
* Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* LangChain team for the base LangGraph framework
* AWS Bedrock team for the session management service
* Valkey team for the Redis-compatible storage
