# Valkey Vector Store

This module provides a vector store implementation for [Valkey](https://valkey.io/), a Redis-compatible in-memory data store that supports vector search capabilities. Uses the [Valkey GLIDE](https://glide.valkey.io/) synchronous client for optimal performance.

## Installation

```bash
pip install langchain-aws[valkey]
```

This installs `valkey-glide-sync>=2.0.0` as the client library.

## Usage

### Basic Example

```python
from langchain_aws.vectorstores import ValkeyVectorStore
from langchain_aws.embeddings import BedrockEmbeddings

# Initialize embeddings
embeddings = BedrockEmbeddings()

# Create vector store from texts
texts = ["Hello world", "Valkey is fast", "Vector search is powerful"]
metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]

vectorstore = ValkeyVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    valkey_url="valkey://localhost:6379",
    index_name="my_index"
)

# Similarity search
results = vectorstore.similarity_search("fast database", k=2)
for doc in results:
    print(doc.page_content)
```

### Connect to Existing Index

```python
vectorstore = ValkeyVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="my_index",
    valkey_url="valkey://localhost:6379"
)
```

### AWS ElastiCache for Valkey

```python
# Connect to AWS ElastiCache Valkey cluster
vectorstore = ValkeyVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    valkey_url="valkeyss://username:password@my-cluster.cache.amazonaws.com:6379",
    index_name="my_index"
)
```

## Features

- **Vector Similarity Search**: Find similar documents using cosine similarity, L2, or inner product
- **Metadata Filtering**: Filter search results using tag, numeric, and text filters
- **Scalable**: Built on Valkey's high-performance architecture
- **AWS Integration**: Works seamlessly with AWS ElastiCache for Valkey and Amazon MemoryDB
- **Cluster Support**: Automatic detection and support for Valkey clusters
- **Synchronous API**: Native sync interface using Valkey GLIDE

## Connection URL Formats

- `valkey://host:port` - Simple connection
- `valkey://username:password@host:port` - With authentication
- `valkeyss://host:port` - SSL connection
- `valkeyss://username:password@host:port` - SSL with authentication

## Filtering

```python
from langchain_aws.vectorstores.valkey.filters import ValkeyTag, ValkeyNum, ValkeyText

# Tag filter
filter_expr = ValkeyTag("category") == "technology"

# Numeric filter
filter_expr = ValkeyNum("year") >= 2020

# Combined filters
filter_expr = (ValkeyTag("category") == "technology") & (ValkeyNum("year") >= 2020)

# Search with filter
results = vectorstore.similarity_search(
    "machine learning",
    k=5,
    filter=str(filter_expr)
)
```

## Configuration

### Vector Schema

Customize the vector index configuration:

```python
vector_schema = {
    "name": "content_vector",
    "algorithm": "HNSW",  # or "FLAT"
    "dims": 1536,
    "distance_metric": "COSINE",  # or "L2", "IP"
    "datatype": "FLOAT32"
}

vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="my_index",
    embedding=embeddings,
    vector_schema=vector_schema
)
```

## Requirements

- Python >= 3.10
- valkey-glide-sync >= 2.0.0
- Valkey server with vector search support (Valkey 8.0+ or Redis 7.2+ with RediSearch)

## AWS Services

This vector store works with:
- [AWS ElastiCache for Valkey](https://aws.amazon.com/elasticache/)
- [Amazon MemoryDB for Valkey](https://aws.amazon.com/memorydb/)

## Additional Resources

- [Valkey GLIDE Documentation](https://glide.valkey.io/)
- [Valkey Vector Search](https://valkey.io/commands/ft.search/)
- [LangChain VectorStores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
