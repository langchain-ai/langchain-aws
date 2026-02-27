# Valkey VectorStore for LangChain-AWS

**Status:** ✅ Complete  
**Last Updated:** 2026-02-02

---

## Overview

Complete vector store integration for Valkey (Redis-compatible) in the `langchain-aws` package, using the synchronous Valkey GLIDE client for optimal performance and simplicity.

---

## Implementation

### Core Files

```
libs/aws/langchain_aws/
├── vectorstores/valkey/
│   ├── __init__.py
│   ├── base.py              # ValkeyVectorStore class (411 lines)
│   ├── filters.py           # Metadata filtering system
│   └── constants.py         # Distance metrics, data types
├── utilities/
│   ├── valkey.py            # GLIDE client management
│   └── redis.py             # Shared utilities (_array_to_buffer)
└── examples/
    ├── valkey_vectorstore_bedrock_example.py
    └── valkey_vectorstore_ollama_example.py
```

### Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
valkey = ["valkey-glide-sync>=2.0.0"]
```

---

## Features

### Core Functionality
- Vector similarity search (COSINE, L2, IP distance metrics)
- Document storage with metadata
- Batch document addition
- Query by text or embedding vector
- Relevance score computation
- Document deletion
- Connection to existing indexes

### Advanced Features
- Metadata filtering (tag, numeric, text fields)
- Custom vector schemas (FLAT and HNSW algorithms)
- Cluster support
- SSL/TLS connections
- Authentication support

### AWS Integration
- AWS ElastiCache for Valkey
- Amazon MemoryDB for Valkey
- Secure connection handling

---

## Architecture

### Client Management

**Synchronous GLIDE Client** (`utilities/valkey.py`)
```python
from glide_sync import GlideClient, GlideClusterClient

def get_client(valkey_url: str, **kwargs) -> GlideClientType:
    """Create sync GLIDE client - auto-detects cluster vs standalone."""
    host, port = _parse_valkey_url(valkey_url)
    addresses = [NodeAddress(host, port)]
    
    try:
        config = GlideClusterClientConfiguration(addresses=addresses, **kwargs)
        return GlideClusterClient.create(config)
    except Exception:
        config = GlideClientConfiguration(addresses=addresses, **kwargs)
        return GlideClient.create(config)
```

### Vector Search

**Direct Synchronous Operations** (`base.py`)
```python
from glide_sync import ft
from glide_shared.commands.server_modules.ft_options.ft_search_options import FtSearchOptions

# Search implementation
results = ft.search(
    self.client,
    self.index_name,
    f"*=>[KNN {k} @{vector_field} $vector AS score]",
    options=FtSearchOptions(params={"vector": embedding_buffer})
)
```

### Filter System

**Metadata Filtering** (`filters.py`)
```python
from langchain_aws.vectorstores.valkey.filters import ValkeyTag, ValkeyNum, ValkeyText

# Tag filter
filter_expr = ValkeyTag("category") == "technology"

# Numeric filter
filter_expr = ValkeyNum("year") >= 2020

# Combined filters
filter_expr = (ValkeyTag("category") == "ai") & (ValkeyNum("year") >= 2023)

# Use in search
results = vectorstore.similarity_search("query", k=3, filter=str(filter_expr))
```

---

## Usage

### Basic Example

```python
from langchain_aws.vectorstores import ValkeyVectorStore
from langchain_aws.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings()

# Create vector store
vectorstore = ValkeyVectorStore.from_texts(
    texts=["Hello world", "Valkey is fast"],
    embedding=embeddings,
    valkey_url="valkey://localhost:6379",
    index_name="my_index"
)

# Search
results = vectorstore.similarity_search("fast database", k=2)
```

### With Metadata Filtering

```python
# Add documents with metadata
vectorstore.add_texts(
    texts=["Document 1", "Document 2"],
    metadatas=[
        {"category": "technology", "year": 2024},
        {"category": "science", "year": 2023}
    ]
)

# Search with filter
from langchain_aws.vectorstores.valkey.filters import ValkeyTag, ValkeyNum

filter_expr = (ValkeyTag("category") == "technology") & (ValkeyNum("year") >= 2024)
results = vectorstore.similarity_search("query", k=5, filter=str(filter_expr))
```

### Connection URLs

```python
# Standalone
valkey_url = "valkey://localhost:6379"

# With authentication
valkey_url = "valkey://username:password@host:6379"

# SSL
valkey_url = "valkeyss://host:6379"
```

### Vector Schema Configuration

```python
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="my_index",
    embedding=embeddings,
    vector_schema={
        "name": "content_vector",
        "algorithm": "HNSW",  # or "FLAT"
        "dims": 1536,         # Titan: 1536, Nomic: 768
        "distance_metric": "COSINE",  # or "L2", "IP"
        "datatype": "FLOAT32",
    }
)
```

---

## Testing

### Unit Tests
**Location:** `tests/unit_tests/vectorstores/valkey/`
- `test_filters.py` - 39 tests for filter expressions
- Coverage: Tag, numeric, text filters, logical operators

### Integration Tests
**Location:** `tests/integration_tests/vectorstores/`
- `test_valkey_filters.py` - 14 tests against real Valkey instance
- Requires: `VALKEY_HOST` environment variable

**Run Integration Tests:**
```bash
export VALKEY_HOST=localhost
pytest tests/integration_tests/vectorstores/test_valkey_filters.py -v
```

---

## Configuration

### Environment Variables

```bash
# For Ollama example
OLLAMA_HOST=http://localhost:11434
VALKEY_HOST=localhost

# For Bedrock example
AWS_REGION=us-east-1
AWS_PROFILE=default  # optional
```

### Default Vector Schema

```python
DEFAULT_VECTOR_SCHEMA = {
    "name": "content_vector",
    "algorithm": "FLAT",
    "dims": 1536,
    "distance_metric": "COSINE",
    "datatype": "FLOAT32",
}
```

---

## Design Decisions

1. **Synchronous Client**: Uses `valkey-glide-sync` for native sync API matching LangChain's VectorStore interface
2. **Minimal Implementation**: 411 lines in core module, focused and maintainable
3. **Type Safety**: Full type hints throughout
4. **Optional Dependency**: Valkey is optional to avoid bloating base package
5. **Pattern Consistency**: Follows existing LangChain vectorstore patterns

---

## File Structure Summary

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | 411 | Core ValkeyVectorStore implementation |
| `filters.py` | ~200 | Metadata filtering system |
| `constants.py` | ~30 | Constants and enums |
| `valkey.py` | ~80 | Client creation utilities |
| Examples | ~150 each | Bedrock and Ollama demos |

**Total Implementation:** ~900 lines

---

## Compatibility

- **Python:** >= 3.10
- **Valkey GLIDE Sync:** >= 2.0.0
- **LangChain Core:** >= 1.1.0
- **AWS Services:** ElastiCache for Valkey, MemoryDB for Valkey
- **Valkey/Redis:** >= 6.0.0 with search module

---

## Known Issues

### Sort By with KNN Queries
**Issue:** Adding `.sort_by()` to KNN queries causes errors  
**Workaround:** Don't use `.sort_by()` (results already sorted by score)  
**Location:** Documented in example scripts

---

## Future Enhancements

1. Semantic cache implementation (similar to InMemorySemanticCache)
2. MMR (Maximal Marginal Relevance) search
3. Async VectorStore variant (if LangChain adds async support)
4. Batch operations optimization
5. Performance benchmarks
6. Additional documentation and tutorials

---

## References

### Documentation
- [Valkey GLIDE Docs](https://glide.valkey.io/)
- [Valkey FT.SEARCH Command](https://valkey.io/commands/ft.search/)
- [LangChain VectorStore Interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

### Code Locations
- **Main Implementation:** `libs/aws/langchain_aws/vectorstores/valkey/base.py`
- **Filters:** `libs/aws/langchain_aws/vectorstores/valkey/filters.py`
- **Utilities:** `libs/aws/langchain_aws/utilities/valkey.py`
- **Tests:** `libs/aws/tests/{unit_tests,integration_tests}/vectorstores/`
- **Examples:** `libs/aws/examples/valkey_vectorstore_*.py`

---

## Git Repository

**Branch:** `feature/vs/valkey`  
**Remote:** `git@improving.github.com:Bit-Quill/langchain-aws.git` (fork)  
**Upstream:** `git@github.com:langchain-ai/langchain-aws.git`

---

**End of Summary**
