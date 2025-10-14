# LangGraph Checkpoint AWS
A custom AWS-based persistence solution for LangGraph agents that provides multiple storage backends including Bedrock AgentCore Memory and high-performance Valkey (Redis-compatible) storage.

## Overview
This package provides multiple persistence solutions for LangGraph agents:

### AWS Bedrock AgentCore Memory Service
1. Stateful conversations and interactions
2. Resumable agent sessions
3. Efficient state persistence and retrieval
4. Seamless integration with AWS Bedrock

### Valkey Storage Solutions
- **High-performance checkpoint storage** with Valkey (Redis-compatible)
- **Intelligent caching** for LLM responses and computation results
- **Document storage** with vector search capabilities
- **AgentCore integration** for enterprise session management

## Installation

You can install the package using pip:

```bash
pip install langgraph-checkpoint-aws
```

Or with Poetry:

```bash
poetry add langgraph-checkpoint-aws
```

## Requirements

```text
Python >=3.9
langgraph-checkpoint >=2.1.0
langgraph >=0.2.55
boto3 >=1.39.7
valkey >=6.1.1
orjson >=3.9.0
```

## Components

This package provides four main components:

1. **BedrockSessionSaver** - AWS Bedrock-based checkpoint storage
2. **ValkeyCheckpointSaver** - High-performance Valkey checkpoint storage
3. **ValkeyCache** - Intelligent caching for LLM responses and computations
4. **ValkeyStore** - Document storage with vector search capabilities
5. **AgentCoreValkeySaver** - Enterprise session management with Valkey backend

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

# Initialize LLM
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=llm,
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
    """Hook that runs pre-LLM invocation to save the latest human message"""
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    
    # Saving the message to the actor and session combination that we get at runtime
    namespace = (actor_id, thread_id)
    
    messages = state.get("messages", [])
    # Save the last human message we see before LLM invocation
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            store.put(namespace, str(uuid.uuid4()), {"message": msg})
            break
            
    # OPTIONAL: Retrieve user preferences based on the last message and append to state
    # user_preferences_namespace = ("preferences", actor_id)
    # preferences = store.search(user_preferences_namespace, query=msg.content, limit=5)
    # # Add to input messages as needed
    
    return {"llm_input_messages": messages}

# Initialize LLM
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=llm,
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

### 3. Valkey Checkpoint Storage

High-performance checkpoint storage using Valkey (Redis-compatible):

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver

# Using connection string
with ValkeyCheckpointSaver.from_conn_string(
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

### 4. Valkey Cache for LLM Responses

Intelligent caching to improve performance and reduce costs:

```python
from langgraph_checkpoint_aws.cache.valkey import ValkeyCache
from langchain_aws import ChatBedrock

# Initialize cache
with ValkeyCache.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as cache:
    
    # Use cache with your LLM calls
    model = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    
    # Cache expensive computations
    cache_key = "expensive_computation_key"
    result = cache.get([cache_key])
    
    if not result:
        # Compute and cache result
        computation_result = {"data": "expensive computation"}
        cache.set({cache_key: (computation_result, 3600)})  # Cache for 1 hour
```

### 5. Valkey Store for Document Storage

Document storage with vector search capabilities using ValkeyIndexConfig:

```python
from langchain_aws import BedrockEmbeddings
from langgraph_checkpoint_aws.store.valkey import ValkeyStore
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

### 6. AgentCore Valkey Integration

Enterprise session management with AgentCore compatibility:

```python
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

# Initialize AgentCore-compatible checkpointer
with AgentCoreValkeySaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as checkpointer:
    
    # AgentCore-style configuration
    config = {
        "configurable": {
            "thread_id": "session-1",      # Session ID
            "actor_id": "agent-1"         # Agent/Actor ID
        }
    }
    
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke({"messages": [...]}, config)
```

## Async Usage

All components support async operations:

```python
from langgraph_checkpoint_aws.async_saver import AsyncBedrockSessionSaver
from langgraph_checkpoint_aws.checkpoint.valkey import AsyncValkeyCheckpointSaver

# Async Bedrock usage
session_saver = AsyncBedrockSessionSaver(region_name="us-west-2")
session_id = (await session_saver.session_client.create_session()).session_id

# Async Valkey usage
async with AsyncValkeyCheckpointSaver.from_conn_string("valkey://localhost:6379") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(1, {"configurable": {"thread_id": "session-1"}})
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

### Valkey Components

All Valkey components support these common configuration options:

#### Connection Options
- **Connection String**: `valkey://localhost:6379` or `valkeys://localhost:6380` (SSL)
- **Connection Pool**: Reusable connection pools for better performance
- **Pool Size**: Maximum number of connections (default: 10)
- **SSL Support**: Secure connections with certificate validation

#### Performance Options
- **TTL (Time-to-Live)**: Automatic expiration of stored data
- **Batch Operations**: Efficient bulk operations for better throughput
- **Async Support**: Non-blocking operations for high concurrency

#### ValkeyCheckpointSaver Options
```python
ValkeyCheckpointSaver(
    client: Valkey,
    ttl: float | None = None,  # TTL in seconds
    serde: SerializerProtocol | None = None  # Custom serialization
)
```

#### ValkeyCache Options
```python
ValkeyCache(
    client: Valkey,
    prefix: str = "langgraph:cache:",  # Key prefix
    ttl: float | None = None,  # Default TTL in seconds
    serde: SerializerProtocol | None = None
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


#### AgentCoreValkeySaver Options
```python
AgentCoreValkeySaver(
    client: Valkey,
    ttl: float | None = None,  # TTL in seconds
    serde: SerializerProtocol | None = None
)
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

- `client`: boto3 Bedrock runtime client (e.g. boto3.client("bedrock-agent-runtime"))
- `session`: boto3.Session for custom credentials
- `region_name`: AWS region where Bedrock is available
- `credentials_profile_name`: Name of AWS credentials profile to use
- `aws_access_key_id`: AWS access key ID for authentication
- `aws_secret_access_key`: AWS secret access key for authentication
- `aws_session_token`: AWS session token for temporary credentials
- `endpoint_url`: Custom endpoint URL for the Bedrock service
- `config`: Botocore configuration object

### Additional AWS permissions for Session Saver:

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

### Valkey Setup (for Valkey components)

#### Using Docker (Recommended)
```bash
# Start Valkey with required modules
docker run --name valkey-bundle -p 6379:6379 -d valkey/valkey-bundle:latest

# Or with custom configuration
docker run --name valkey-custom \
  -p 6379:6379 \
  -v $(pwd)/valkey.conf:/etc/valkey/valkey.conf \
  -d valkey/valkey-bundle:latest
```

#### Using AWS ElastiCache for Valkey
```python
# Connect to AWS ElastiCache from host running inside VPC with access to cache
from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver

checkpointer = ValkeyCheckpointSaver.from_conn_string(
    "valkeys://your-elasticache-cluster.amazonaws.com:6379",
    pool_size=20
)
```
If you want to connect to cache from a host outside of VPC, use ElastiCache console to setup a jump host so you could create SSH tunnel to access cache locally. 

#### Production Configuration
```bash
# valkey.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
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

with ValkeyCheckpointSaver.from_pool(pool) as checkpointer:
    # Reuse connections across operations
    pass
```

#### TTL Strategy
```python
# Configure appropriate TTL values
checkpointer = ValkeyCheckpointSaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600  # 1 hour for active sessions
)

# Different TTL for different use cases
cache = ValkeyCache.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=300  # 5 minutes for LLM responses
)
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

### Vector Search Optimization

```python
# Configure vector indexing for semantic search with ValkeyIndexConfig
with ValkeyStore.from_conn_string(
    "valkey://localhost:6379",
    index={
        "collection_name": "optimized_search",
        "dims": 1536,  # Match your embedding model
        "embed": "openai:text-embedding-3-small",
        "fields": ["text", "content", "description"],
        "index_type": "hnsw",
        # Performance tuning for your use case
        "hnsw_m": 24,  # Balance between speed and accuracy
        "hnsw_ef_construction": 300,  # Higher for better index quality
        "hnsw_ef_runtime": 15,  # Higher for better search accuracy
    }
) as store:

    # Optimize search queries
    results = store.search(
        ("documents",),
        query="machine learning",
        filter={"category": "research"},
        limit=10,
        offset=0
    )

# Algorithm selection based on use case
# HNSW: Fast approximate search for large datasets
hnsw_config = {
    "collection_name": "large_dataset",
    "index_type": "hnsw",
    "hnsw_m": 16,  # Lower for faster search
    "hnsw_ef_construction": 200,
    "hnsw_ef_runtime": 10
}

# FLAT: Exact search for smaller datasets or high precision needs
flat_config = {
    "collection_name": "small_precise_dataset", 
    "index_type": "flat"  # No approximation, exact results
}
```

## Security Considerations

- Never commit AWS credentials
- Use environment variables or AWS IAM roles for authentication
- Follow AWS security best practices
- Use IAM roles and temporary credentials when possible
- Implement proper access controls for session management

### Valkey Security
* Use SSL/TLS for production deployments (`valkeys://` protocol)
* Configure authentication with strong passwords
* Implement network security (VPC, security groups)
* Regular security updates and monitoring
* Use AWS ElastiCache for managed Valkey with encryption

```python
# Secure connection example
checkpointer = ValkeyCheckpointSaver.from_conn_string(
    "valkeys://username:password@your-secure-host:6380",
    ssl_cert_reqs="required",
    ssl_ca_certs="/path/to/ca.pem"
)
```

## Examples and Samples

Comprehensive examples are available in the `samples/memory/` directory:

- **`agentcore_valkey_checkpointer.ipynb`**: AgentCore integration with enterprise session management
- **`valkey_cache.ipynb`**: LLM response caching and performance optimization
- **`valkey_checkpointer.ipynb`**: Basic checkpoint storage with Valkey
- **`valkey_store.ipynb`**: Document storage with vector search capabilities

## Troubleshooting

### Common Issues

#### Valkey Connection Issues
```python
# Test Valkey connection
from valkey import Valkey

try:
    client = Valkey.from_url("valkey://localhost:6379")
    client.ping()
    print("✅ Valkey connection successful")
except Exception as e:
    print(f"❌ Valkey connection failed: {e}")
```

#### Memory Issues
```bash
# Check Valkey memory usage
valkey-cli info memory

# Configure memory limits
valkey-cli config set maxmemory 2gb
valkey-cli config set maxmemory-policy allkeys-lru
```

#### Performance Issues
- Use connection pooling for high-concurrency applications
- Configure appropriate TTL values to prevent memory bloat
- Monitor key patterns and optimize data structures
- Use batch operations for bulk data operations

## Contributing

- Fork the repository
- Create a feature branch
- Make your changes
- Run tests and linting
- Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain team for the base LangGraph framework
- AWS Bedrock team for the session management service
- Valkey team for the high-performance Redis-compatible storage

