# AgentCore Valkey Checkpoint Saver

The `AgentCoreValkeySaver` combines the best of both worlds: **AgentCore session management concepts** with **Valkey's high-performance storage backend**. This provides a scalable, fast, and feature-rich checkpoint storage solution for LangGraph applications.

## ✨ Features

- 🚀 **High Performance**: Leverages Valkey's Redis-compatible performance
- 🎯 **AgentCore Compatible**: Uses session_id/actor_id patterns for consistency
- 🔄 **TTL Support**: Automatic cleanup of old checkpoints
- 🏊 **Connection Pooling**: Scalable connection management
- 🔍 **Metadata Filtering**: Query checkpoints by metadata
- 🧹 **Easy Cleanup**: Simple thread/session deletion
- 📝 **Robust Serialization**: JSON + base64 encoding for complex data
- 🔒 **Thread Safe**: Concurrent operations supported

## 🚀 Quick Start

### Basic Usage

```python
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver
from langgraph.prebuilt import create_react_agent

# Create checkpointer with connection string
with AgentCoreValkeySaver.from_conn_string(
    "valkey://localhost:6379",
    ttl_seconds=3600,  # 1 hour TTL
    pool_size=10
) as checkpointer:

    # Create LangGraph agent
    graph = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer
    )

    # AgentCore-style configuration
    config = {
        "configurable": {
            "thread_id": "user-session-123",     # Session ID
            "actor_id": "assistant-agent",       # Agent/Actor ID
            "checkpoint_ns": "production",       # Namespace
        }
    }

    # Use the agent
    response = graph.invoke({"messages": [...]}, config)
```

### Advanced Usage with Connection Pool

```python
from valkey.connection import ConnectionPool
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

# Create optimized connection pool
pool = ConnectionPool.from_url(
    "valkey://localhost:6379",
    max_connections=20,
    retry_on_timeout=True
)

with AgentCoreValkeySaver.from_pool(pool, ttl_seconds=1800) as checkpointer:
    # Your application logic here
    pass
```

## 🏗️ Architecture

### Key Generation Pattern

The saver uses a structured key naming convention:

```
agentcore:session:{session_id}:{actor_id}:checkpoints       # Session checkpoint list
agentcore:checkpoint:{session_id}:{actor_id}:{ns}:{cp_id}   # Individual checkpoint
agentcore:writes:{session_id}:{actor_id}:{ns}:{cp_id}       # Pending writes
agentcore:channel:{session_id}:{actor_id}:{ns}:{ch}:{cp_id} # Channel data
```

### Data Models

```python
# Checkpoint storage
StoredCheckpoint:
  - checkpoint_id: str
  - session_id: str
  - actor_id: str
  - checkpoint_ns: str
  - parent_checkpoint_id: str | None
  - checkpoint_data: Dict[str, Any]
  - metadata: Dict[str, Any]
  - created_at: float

# Write operations
StoredWrite:
  - checkpoint_id: str
  - task_id: str
  - channel: str
  - value: Any
  - task_path: str
  - created_at: float

# Channel data
StoredChannelData:
  - channel: str
  - version: str
  - value: Any
  - checkpoint_id: str
  - created_at: float
```

## 🔧 Configuration Options

### Connection Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conn_string` | str | Required | Valkey connection URL |
| `ttl_seconds` | float \| None | None | Time-to-live for checkpoints |
| `pool_size` | int | 10 | Max connections in pool |

### Runtime Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `thread_id` | str | ✅ | Session identifier |
| `actor_id` | str | ✅ | Agent/actor identifier |
| `checkpoint_ns` | str | | Checkpoint namespace |
| `checkpoint_id` | str | | Specific checkpoint ID |

## 🔍 Advanced Features

### Metadata Filtering

```python
# Filter checkpoints by metadata
checkpoints = list(checkpointer.list(
    config,
    filter={"user_id": "12345", "environment": "production"}
))
```

### TTL Management

```python
# Checkpoints automatically expire after TTL
checkpointer = AgentCoreValkeySaver(client, ttl=3600)  # 1 hour

# Check TTL on stored keys
ttl = client.ttl("agentcore:checkpoint:session:actor:ns:cp_id")
```

### Concurrent Operations

```python
import threading

def create_checkpoint(session_id, actor_id):
    config = {
        "configurable": {
            "thread_id": session_id,
            "actor_id": actor_id,
            "checkpoint_ns": "concurrent"
        }
    }

    # Thread-safe operations
    checkpointer.put(config, checkpoint, metadata, versions)

# Multiple threads can safely operate concurrently
threads = [
    threading.Thread(target=create_checkpoint, args=(f"session_{i}", "agent"))
    for i in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Batch Operations

```python
# Store multiple writes efficiently
writes = [
    ("messages", {"role": "assistant", "content": "Response 1"}),
    ("context", {"updated_at": time.time()}),
    ("state", {"step": 5})
]

checkpointer.put_writes(config, writes, "batch_task_1")
```

## 🚀 Performance Considerations

### Connection Pooling

For high-throughput applications, use connection pooling:

```python
# Recommended pool settings
pool = ConnectionPool.from_url(
    conn_string,
    max_connections=min(cpu_count * 2, 20),
    retry_on_timeout=True,
    socket_keepalive=True,
    socket_keepalive_options={}
)
```

### Memory Usage

- Large checkpoints are automatically compressed
- Base64 encoding adds ~33% overhead for binary data
- Consider TTL settings to prevent memory bloat

### Network Optimization

```python
# Use pipeline for batch operations (if needed)
pipe = client.pipeline()
for operation in operations:
    pipe.set(key, value)
pipe.execute()
```

## 🧪 Testing

### Unit Tests

```bash
cd libs/langgraph-checkpoint-aws
pytest tests/unit_tests/agentcore/valkey/ -v
```

### Integration Tests

```bash
# Requires Valkey server running on localhost:6379
pytest tests/integration_tests/agentcore/valkey/ -v
```

### Test Configuration

```python
# Use separate database for testing
test_checkpointer = AgentCoreValkeySaver.from_conn_string(
    "valkey://localhost:6379/1",  # Database 1 for tests
    ttl_seconds=60  # Short TTL for tests
)
```

## 🔧 Troubleshooting

### Connection Issues

```python
try:
    with AgentCoreValkeySaver.from_conn_string(conn_string) as checkpointer:
        # Test connection
        checkpointer.client.ping()
except ConnectionError as e:
    print(f"Cannot connect to Valkey: {e}")
```

### Memory Issues

```python
# Monitor memory usage
info = client.info('memory')
print(f"Used memory: {info['used_memory_human']}")
print(f"Max memory: {info['maxmemory_human']}")
```

### Key Debugging

```python
# List all keys for a session
session_keys = client.keys("agentcore:*session-123*")
print(f"Found {len(session_keys)} keys for session")

# Inspect key contents
checkpoint_data = client.get("agentcore:checkpoint:session-123:agent:ns:cp_id")
print(json.loads(checkpoint_data))
```

## 📊 Comparison with Other Savers

| Feature | AgentCoreValkey | AgentCoreMemory | ValkeyCheckpoint |
|---------|-----------------|-----------------|------------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AWS Integration | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Session Management | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Local Development | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Scalability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Management Overhead | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🛠️ Migration Guide

### From AgentCoreMemorySaver

```python
# Before: AgentCoreMemorySaver
from langgraph_checkpoint_aws import AgentCoreMemorySaver

checkpointer = AgentCoreMemorySaver(memory_id, region_name="us-west-2")

# After: AgentCoreValkeySaver
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

checkpointer = AgentCoreValkeySaver.from_conn_string("valkey://localhost:6379")

# Config remains the same!
config = {
    "configurable": {
        "thread_id": "session-123",
        "actor_id": "agent-456"
    }
}
```

### From ValkeyCheckpointSaver

```python
# Before: ValkeyCheckpointSaver
config = {
    "configurable": {
        "thread_id": "thread-123",
        "checkpoint_ns": ""
    }
}

# After: AgentCoreValkeySaver
config = {
    "configurable": {
        "thread_id": "session-123",      # session_id
        "actor_id": "agent-456",         # NEW: required actor_id
        "checkpoint_ns": "namespace"
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
