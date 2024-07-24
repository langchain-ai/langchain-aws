# langchain-aws

This package contains the LangChain integrations with AWS.

## Installation

```bash
pip install -U langchain-aws
```
All integrations in this package assume that you have the credentials setup to connect with AWS services.

## Chat Models

`ChatBedrock` class exposes chat models from Bedrock.

```python
from langchain_aws import ChatBedrock

llm = ChatBedrock()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`BedrockEmbeddings` class exposes embeddings from Bedrock.

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`BedrockLLM` class exposes LLMs from Bedrock.

```python
from langchain_aws import BedrockLLM

llm = BedrockLLM()
llm.invoke("The meaning of life is")
```

## Retrievers
`AmazonKendraRetriever` class provides a retriever to connect with Amazon Kendra.

```python
from langchain_aws import AmazonKendraRetriever

retriever = AmazonKendraRetriever(
    index_id="561be2b6d-9804c7e7-f6a0fbb8-5ccd350"
)

retriever.get_relevant_documents(query="What is the meaning of life?")
```

`AmazonKnowledgeBasesRetriever` class provides a retriever to connect with Amazon Knowledge Bases.

```python
from langchain_aws import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="IAPJ4QPUEU",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

retriever.get_relevant_documents(query="What is the meaning of life?")
```
## VectorStores 
`InMemoryVectorStore` class provides a vectorstore to connect with Amazon MemoryDB.

```python
from langchain_aws.vectorstores.inmemorydb import InMemoryVectorStore

vds = InMemoryVectorStore.from_documents(
            chunks,
            embeddings,
            redis_url="rediss://cluster_endpoint:6379/ssl=True ssl_cert_reqs=none",
            vector_schema=vector_schema,
            index_name=INDEX_NAME,
        )
```

## MemoryDB as Retriever

Here we go over different options for using the vector store as a retriever.

There are three different search methods we can use to do retrieval. By default, it will use semantic similarity.

```python
retriever=vds.as_retriever()
```