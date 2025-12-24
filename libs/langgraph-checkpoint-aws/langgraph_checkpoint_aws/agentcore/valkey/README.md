# AgentCore Valkey Checkpoint Saver

The `AgentCoreValkeySaver` combines the best of both worlds: **AgentCore session management concepts** with **Valkey's high-performance storage backend**. This provides a scalable, fast, and feature-rich checkpoint storage solution for LangGraph applications.

## 🤔 Choosing the Right Saver

**This package provides multiple Valkey checkpoint savers. Choose the right one for your use case:**

### Quick Decision Tree

```
Do you use AWS Bedrock AgentCore Memory APIs?
├─ YES → Use AgentCoreValkeySaver ✓
│         (Required for AgentCore integration)
│
└─ NO  → Do you need multi-actor session management?
          ├─ YES → Use AgentCoreValkeySaver ✓
          │        (Future-proof for AgentCore)
          │
          └─ NO  → Use ValkeySaver from checkpoint.valkey
                   (Simpler configuration)
```

### AgentCoreValkeySaver vs ValkeySaver

| Aspect | **AgentCoreValkeySaver** | **ValkeySaver** |
|--------|-------------------------|-----------------|
| **Import Path** | `langgraph_checkpoint_aws.agentcore.valkey` | `langgraph_checkpoint_aws.checkpoint.valkey` |
| **Config Requirements** | `thread_id`, **`actor_id`**, `checkpoint_ns` | `thread_id`, `checkpoint_ns` |
| **Key Prefix** | `agentcore:checkpoint:{session_id}:{actor_id}:` | `checkpoint:{thread_id}:` |
| **AgentCore Compatible** | ✅ Yes | ❌ No |
| **Retry Logic** | ✅ Built-in exponential backoff | ❌ No retries |
| **Data Models** | ✅ Pydantic validation | Basic serialization |
| **Use Case** | AgentCore integration, multi-actor apps | Standard LangGraph apps |

### ⚠️ **Critical: Data Incompatibility**

**AgentCoreValkeySaver and ValkeySaver use INCOMPATIBLE storage formats:**

- ❌ Data stored by one **CANNOT** be read by the other
- ❌ Different key structures prevent interoperability
- ❌ Migration requires data export/transform/reimport
- ✅ **Choose ONE at project start and stick with it**

### When to Use AgentCoreValkeySaver

✅ **Use AgentCoreValkeySaver when:**
- Integrating with AWS Bedrock AgentCore Memory services
- Building multi-agent systems with actor-based routing
- Need `actor_id` for session organization
- Want built-in retry logic for production resilience
- Require AgentCore-compatible key structures
- Need Pydantic validation for data integrity

### When to Use ValkeySaver

✅ **Use ValkeySaver when:**
- Building standard LangGraph applications
- Simple thread-based checkpointing is sufficient
- Don't need `actor_id` in configuration
- Want minimal dependencies and overhead
- Prefer simpler key structure
- Don't plan to use AgentCore services

---

## ✨ Features

- 🚀 **High Performance**: Leverages Valkey's Redis-compatible performance
- 🎯 **AgentCore Compatible**: Uses session_id/actor_id patterns for consistency
- 🔄 **TTL Support**: Automatic cleanup of old checkpoints
- 🏊 **Connection Pooling**: Scalable connection management
- 🔍 **Metadata Filtering**: Query checkpoints by metadata
- 🧹 **Easy Cleanup**: Simple thread/session deletion
- 📝 **Robust Serialization**: JSON + base64 encoding for complex data
- 🔒 **Thread Safe**: Concurrent operations supported
- 🔁 **Retry Logic**: Exponential backoff for transient failures
- ✅ **Pydantic Validation**: Type-safe data models

## 🚀 Quick Start

### Basic Usage

```python
from langgraph_checkpoint_aws import AgentCoreValkeySaver
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
            "actor_id": "assistant-agent",       # Agent/Actor ID (REQUIRED)
            "checkpoint_ns": "production",       # Namespace
        }
    }

    # Use the agent
    response = graph.invoke({"messages": [...]}, config)
```

### Advanced Usage with Connection Pool

```python
from valkey.connection import ConnectionPool
from langgraph_checkpoint_aws import AgentCoreValkeySaver

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

### Storage Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Application                │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ config: {thread_id, actor_id, ...}
                     │
         ┌───────────▼──────────────┐
         │  AgentCoreValkeySaver    │
         │                          │
         │  • Pydantic Models       │
         │  • Retry Logic           │
         │  • Session Management    │
         └───────────┬──────────────┘
                     │
                     │ agentcore:* keys
                     │
         ┌───────────▼──────────────┐
         │     Valkey Storage       │
         │                          │
         │  • High Performance      │
         │  • Redis Compatible      │
         │  • TTL Support           │
         └──────────────────────────┘
```

### Key Generation Pattern

The saver uses a structured key naming convention:

```
agentcore:session:{session_id}:{actor_id}:checkpoints       # Session checkpoint list
agentcore:checkpoint:{session_id}:{actor_id}:{ns}:{cp_id}   # Individual checkpoint
agentcore:writes:{session_id}:{actor_id}:{ns}:{cp_id}       # Pending writes
agentcore:channel:{session_id}:{actor_id}:{ns}:{ch}:{cp_id} # Channel data
```

**Example Keys:**
```
agentcore:session:user-123_prod:assistant-agent:checkpoints
agentcore:checkpoint:user-123_prod:assistant-agent:prod:cp_001
agentcore:writes:user-123_prod:assistant-agent:prod:cp_001
agentcore:channel:user-123_prod:assistant-agent:prod:messages:cp_001
```

### Session ID Generation

The `session_id` is automatically generated from `thread_id` and `checkpoint_ns`:

```python
# Empty namespace
thread_id = "user-123"
checkpoint_ns = ""
session_id = "user-123"  # Just thread_id

# With namespace
thread_id = "user-123"
checkpoint_ns = "production"
session_id = "user-123_production"  # Combined with underscore
```

### Data Models

```python
# Checkpoint storage
StoredCheckpoint:
  - checkpoint_id: str
  - session_id: str
  - actor_id: str
  - thread_id: str
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
| `max_retries` | int | 3 | Max retry attempts for failed operations |
| `retry_delay` | float | 0.1 | Base delay for exponential backoff (seconds) |

### Runtime Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `thread_id` | str | ✅ | Session identifier |
| `actor_id` | str | ✅ | Agent/actor identifier (**Required for AgentCore**) |
| `checkpoint_ns` | str | | Checkpoint namespace |
| `checkpoint_id` | str | | Specific checkpoint ID |

## 🔍 Advanced Features

### Retry Logic with Exponential Backoff

Built-in retry logic handles transient failures:

```python
# Automatic retries with exponential backoff
checkpointer = AgentCoreValkeySaver(
    client,
    max_retries=5,        # Retry up to 5 times
    retry_delay=0.2,      # Base delay: 200ms
)

# Retry delays: 200ms, 400ms, 800ms, 1600ms, 3200ms
# With jitter to prevent thundering herd
```

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
from multiprocessing import cpu_count

# Recommended pool settings
pool = ConnectionPool.from_url(
    conn_string,
    max_connections=min(cpu_count() * 2, 20),
    retry_on_timeout=True,
    socket_keepalive=True,
    socket_keepalive_options={}
)
```

### Memory Usage

- Large checkpoints are automatically compressed
- Base64 encoding adds ~33% overhead for binary data
- Consider TTL settings to prevent memory bloat
- Monitor with `client.info('memory')`

### Network Optimization

```python
# Use pipeline for batch operations (if needed)
pipe = client.pipeline()
for operation in operations:
    pipe.set(key, value)
pipe.execute()
```

### Retry Tuning

```python
# Production-grade retry configuration
checkpointer = AgentCoreValkeySaver(
    client,
    ttl=3600,
    max_retries=5,         # More retries for critical systems
    retry_delay=0.1,       # Aggressive initial delay
)

# Development/testing configuration
checkpointer = AgentCoreValkeySaver(
    client,
    max_retries=1,         # Fast failure for debugging
    retry_delay=0.05,
)
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

### Common Errors

#### Missing `actor_id`

```python
# ❌ ERROR: Will raise InvalidConfigError
config = {
    "configurable": {
        "thread_id": "session-123",
        # Missing actor_id!
    }
}

# ✅ CORRECT: Include actor_id
config = {
    "configurable": {
        "thread_id": "session-123",
        "actor_id": "agent-456",  # Required!
    }
}
```

#### Wrong Import Path

```python
# ✅ RECOMMENDED: Import from package root
from langgraph_checkpoint_aws import AgentCoreValkeySaver

# ✅ ALSO WORKS: Direct import from submodule
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

# ❌ ERROR: This is the WRONG saver
from langgraph_checkpoint_aws.checkpoint.valkey import ValkeySaver  # Not AgentCore compatible!

# ❌ ERROR: This is also the WRONG saver
from langgraph_checkpoint_aws import ValkeySaver  # Not AgentCore compatible!
```

## 📊 Comparison with Other Savers

| Feature | AgentCoreValkey | ValkeySaver | AgentCoreMemory |
|---------|-----------------|-------------|-----------------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **AgentCore Integration** | ⭐⭐⭐⭐⭐ | ❌ None | ⭐⭐⭐⭐⭐ |
| **Actor Management** | ⭐⭐⭐⭐⭐ | ❌ None | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **AWS Managed** | ❌ Self-hosted | ❌ Self-hosted | ⭐⭐⭐⭐⭐ |
| **Retry Logic** | ⭐⭐⭐⭐⭐ | ❌ None | ⭐⭐⭐ |
| **Local Development** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Management Overhead** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Data Compatible With** | AgentCore only | None | AgentCore only |

### Use Case Recommendations

**AgentCoreValkeySaver:**
- ✅ AWS Bedrock AgentCore projects
- ✅ Multi-agent systems with actor routing
- ✅ High-performance local/cloud deployments
- ✅ Production systems needing retry logic

**ValkeySaver:**
- ✅ Standard LangGraph applications
- ✅ Simple single-agent use cases
- ✅ Minimal configuration requirements
- ✅ Thread-only session management

**AgentCoreMemory:**
- ✅ Fully managed AWS solution
- ✅ Zero infrastructure management
- ✅ Enterprise-scale deployments
- ✅ AWS-native integration

## 🛠️ Migration Guide

### From AgentCoreMemorySaver

**Good News:** Configuration is identical! Just swap the checkpointer.

```python
# Before: AgentCoreMemorySaver
from langgraph_checkpoint_aws import AgentCoreMemorySaver

checkpointer = AgentCoreMemorySaver(memory_id, region_name="us-west-2")

# After: AgentCoreValkeySaver
from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver

checkpointer = AgentCoreValkeySaver.from_conn_string("valkey://localhost:6379")

# Config remains EXACTLY the same! ✓
config = {
    "configurable": {
        "thread_id": "session-123",
        "actor_id": "agent-456"
    }
}
```

**Benefits of switching:**
- ✅ Lower latency (local storage)
- ✅ No AWS API calls
- ✅ Better for development
- ⚠️ Requires managing Valkey infrastructure

---

### From ValkeySaver (checkpoint.valkey)

**⚠️ WARNING: Data Migration Required**

AgentCoreValkeySaver and ValkeySaver use **INCOMPATIBLE** storage formats and key structures. Existing data will NOT be automatically accessible.

#### Configuration Changes

```python
# Before: ValkeySaver (checkpoint.valkey)
from langgraph_checkpoint_aws import ValkeySaver

checkpointer = ValkeySaver.from_conn_string("valkey://localhost:6379")

config = {
    "configurable": {
        "thread_id": "session-123",
        "checkpoint_ns": ""
        # No actor_id
    }
}

# After: AgentCoreValkeySaver
from langgraph_checkpoint_aws import AgentCoreValkeySaver

checkpointer = AgentCoreValkeySaver.from_conn_string("valkey://localhost:6379")

config = {
    "configurable": {
        "thread_id": "session-123",      # Maps to session_id
        "actor_id": "agent-456",         # NEW: Required for AgentCore!
        "checkpoint_ns": "namespace"
    }
}
```

#### Key Structure Changes

```python
# ValkeySaver keys
checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}
writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}

# AgentCoreValkeySaver keys
agentcore:checkpoint:{session_id}:{actor_id}:{checkpoint_ns}:{checkpoint_id}
agentcore:writes:{session_id}:{actor_id}:{checkpoint_ns}:{checkpoint_id}
```

#### Data Migration Steps

```python
# 1. Export data from ValkeySaver
from langgraph_checkpoint_aws import ValkeySaver

with ValkeySaver.from_conn_string("valkey://localhost:6379") as old_saver:
    old_config = {"configurable": {"thread_id": "session-123"}}

    # Get all checkpoints
    old_checkpoints = list(old_saver.list(old_config))

# 2. Transform and import to AgentCoreValkeySaver
from langgraph_checkpoint_aws import AgentCoreValkeySaver

with AgentCoreValkeySaver.from_conn_string("valkey://localhost:6379") as new_saver:
    for checkpoint_tuple in old_checkpoints:
        # Add actor_id to config
        new_config = {
            "configurable": {
                "thread_id": old_config["configurable"]["thread_id"],
                "actor_id": "default-agent",  # Choose appropriate actor_id
                "checkpoint_ns": old_config["configurable"].get("checkpoint_ns", "")
            }
        }

        # Save to new format
        new_saver.put(
            new_config,
            checkpoint_tuple.checkpoint,
            checkpoint_tuple.metadata,
            checkpoint_tuple.checkpoint["channel_versions"]
        )
    }
```

#### When Migration is Worth It

✅ **Migrate when:**
- Planning to use AgentCore Memory services
- Need multi-actor session management
- Want retry logic and Pydantic validation
- Have < 10,000 checkpoints to migrate

❌ **Don't migrate when:**
- Simple single-agent use case
- Large existing dataset (millions of checkpoints)
- No need for actor_id organization
- Migration downtime is unacceptable

**Alternative:** Start fresh with AgentCoreValkeySaver for new sessions while keeping old data in ValkeySaver (read-only).

---

## 💡 Best Practices

### 1. Always Include `actor_id`

```python
# ✅ GOOD: Explicit actor_id
config = {
    "configurable": {
        "thread_id": "user-session-123",
        "actor_id": "customer-support-agent",  # Clear actor identification
        "checkpoint_ns": "production"
    }
}

# ❌ BAD: Will raise InvalidConfigError
config = {
    "configurable": {
        "thread_id": "user-session-123",
        # Missing actor_id
    }
}
```

### 2. Use Meaningful Actor IDs

```python
# ✅ GOOD: Descriptive actor IDs
actor_ids = [
    "customer-support-agent",
    "sales-assistant",
    "technical-advisor"
]

# ❌ BAD: Generic IDs
actor_ids = ["agent1", "agent2", "agent3"]
```

### 3. Set Appropriate TTLs

```python
# Development: Short TTL
dev_checkpointer = AgentCoreValkeySaver(client, ttl=300)  # 5 minutes

# Production: Longer TTL
prod_checkpointer = AgentCoreValkeySaver(client, ttl=86400)  # 24 hours

# Long-term storage: No TTL
archive_checkpointer = AgentCoreValkeySaver(client, ttl=None)  # Never expire
```

### 4. Monitor Key Space

```python
# Regularly check key count
def monitor_keys():
    client = Valkey.from_url("valkey://localhost:6379")

    # Count all agentcore keys
    all_keys = client.keys("agentcore:*")
    print(f"Total AgentCore keys: {len(all_keys)}")

    # Count by type
    checkpoints = client.keys("agentcore:checkpoint:*")
    writes = client.keys("agentcore:writes:*")

    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Writes: {len(writes)}")
```

### 5. Clean Up Old Sessions

```python
# Implement session cleanup
def cleanup_old_sessions(checkpointer, cutoff_days=30):
    """Remove sessions older than cutoff_days."""
    import time
    cutoff_timestamp = time.time() - (cutoff_days * 86400)

    # This would require custom logic to iterate through sessions
    # and check their timestamps, then call delete_thread()
    pass
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Valkey Documentation](https://valkey.io/docs/)
- [AWS Bedrock AgentCore](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Main Package README](../../../README.md)

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/langchain-ai/langchain-aws/issues)
- **Discussions:** [GitHub Discussions](https://github.com/langchain-ai/langchain-aws/discussions)
- **Documentation:** [Official Docs](https://python.langchain.com/docs/integrations/providers/aws)
