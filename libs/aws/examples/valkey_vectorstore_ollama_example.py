"""Example usage of ValkeyVectorStore with Ollama.

This example demonstrates how to use the ValkeyVectorStore with Ollama embeddings
for vector similarity search.

Requirements:
    - Valkey server with search module (e.g., docker run -p 6379:6379 valkey/valkey-bundle:latest)
    - Ollama running with nomic-embed-text model
    - Install: pip install langchain-aws[valkey] langchain-ollama
    - Set OLLAMA_HOST environment variable (defaults to http://localhost:11434)
    - Set VALKEY_HOST environment variable (defaults to localhost:6379)
"""

import os

from langchain_ollama import OllamaEmbeddings
from valkey import Valkey
from valkey.commands.search.field import VectorField
from valkey.commands.search.indexDefinition import IndexDefinition, IndexType

from langchain_aws.vectorstores import ValkeyVectorStore

# Sample documents
texts = [
    "Valkey is a high-performance key-value store",
    "Vector databases enable semantic search",
    "AWS ElastiCache supports Valkey",
    "Machine learning models generate embeddings",
    "Similarity search finds related documents",
]

metadatas = [
    {"category": "database", "year": 2024},
    {"category": "ai", "year": 2024},
    {"category": "cloud", "year": 2024},
    {"category": "ai", "year": 2023},
    {"category": "search", "year": 2024},
]

# Initialize Ollama embeddings
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=ollama_host
)

# Get Valkey connection details
valkey_host = os.getenv("VALKEY_HOST", "localhost")
valkey_url = f"valkey://{valkey_host}"
index_name = "ollama_example_index"

# Create index if it doesn't exist
print("Setting up Valkey index...")
client = Valkey.from_url(valkey_url)
try:
    client.ft(index_name).info()
    print(f"Index '{index_name}' already exists")
except:
    # Create index with vector field for nomic embeddings (768 dimensions)
    # Note: Metadata fields are stored but not indexed for simplicity
    schema = (
        VectorField(
            "content_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 768,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    definition = IndexDefinition(prefix=["doc:ollama_example_index:"], index_type=IndexType.HASH)
    client.ft(index_name).create_index(fields=schema, definition=definition)
    print(f"Created index '{index_name}'")

# Create vector store
print("Creating vector store...")
vectorstore = ValkeyVectorStore(
    embedding=embeddings,
    valkey_url=valkey_url,
    index_name=index_name,
    vector_schema={
        "name": "content_vector",
        "algorithm": "FLAT",
        "dims": 768,
        "distance_metric": "COSINE",
        "datatype": "FLOAT32",
    }
)

# Add texts
vectorstore.add_texts(texts, metadatas=metadatas)

# Perform similarity search
print("\nSearching for: 'fast database'")
results = vectorstore.similarity_search("fast database", k=3)

print("\nTop 3 results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
    print(f"   Metadata: {doc.metadata}\n")

# Search with scores
print("\nSearching with scores: 'AWS cloud services'")
results_with_scores = vectorstore.similarity_search_with_score(
    "AWS cloud services", k=2
)

print("\nTop 2 results with scores:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"{i}. {doc.page_content}")
    print(f"   Score: {score:.4f}")
    print(f"   Metadata: {doc.metadata}\n")

print("Example completed successfully!")
