"""Example usage of ValkeyVectorStore.

This example demonstrates how to use the ValkeyVectorStore for vector similarity search.

Requirements:
    - Valkey server running (e.g., docker run -p 6379:6379 valkey/valkey:latest)
    - AWS credentials configured for Bedrock access
    - Install: pip install langchain-aws[valkey]
"""

from langchain_aws.embeddings import BedrockEmbeddings
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

# Initialize embeddings (requires AWS credentials)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

# Create vector store
print("Creating vector store...")
vectorstore = ValkeyVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    valkey_url="valkey://localhost:6379",
    index_name="example_index"
)

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

# Example with filtering (if filters are implemented)
# from langchain_aws.vectorstores.valkey import ValkeyFilter
# filter_expr = ValkeyFilter.tag("category") == "ai"
# results = vectorstore.similarity_search(
#     "machine learning",
#     k=2,
#     filter=str(filter_expr)
# )

print("Example completed successfully!")
