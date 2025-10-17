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
1. **High-performance checkpoint storage** with Valkey (Redis-compatible)

## Installation

You can install the package using pip:

```bash
pip install langgraph-checkpoint-aws
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

This package provides three main components:

1. **AgentCoreMemorySaver** - AWS Bedrock-based checkpoint storage
2. **ValkeyCheckpointSaver** - High-performance Valkey checkpoint storage
3. **AgentCoreMemoryStore** - AWS Bedrock-based document store


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

Valkey components support these common configuration options:

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

### Valkey Setup (for Valkey components)

#### Using AWS ElastiCache for Valkey (Recommended)
```python
# Connect to AWS ElastiCache from host running inside VPC with access to cache
from langgraph_checkpoint_aws.checkpoint.valkey import ValkeyCheckpointSaver

checkpointer = ValkeyCheckpointSaver.from_conn_string(
    "valkeys://your-elasticache-cluster.amazonaws.com:6379",
    pool_size=20
)
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
```

## Security Considerations

* Never commit AWS credentials

* Use environment variables or AWS IAM roles for authentication
* Follow AWS security best practices
* Use IAM roles and temporary credentials when possible
* Implement proper access controls for session management

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
* Valkey team for the high-performance Redis-compatible storage

