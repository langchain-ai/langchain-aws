# langchain-aws

This package contains the LangChain integrations with AWS.

## Installation

```bash
pip install -U langchain-aws
```
All integrations in this package assume that you have the credentials setup to connect with AWS services.

## Authentication

In order to use Amazon Bedrock models, you need to configure AWS credentials. One of the options is to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables. More information can be found [here](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html). 
Alternatively, set the `AWS_BEARER_TOKEN_BEDROCK` environment variable locally for API Key authentication. For additional API key details, refer to [docs](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).

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

### InMemoryVectorStore

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

### MemoryDB as Retriever

Here we go over different options for using the vector store as a retriever.

There are three different search methods we can use to do retrieval. By default, it will use semantic similarity.

```python
retriever = vds.as_retriever()
```

### AmazonS3Vectors

`AmazonS3Vectors` class provides a vectorstore to connect with Amazon S3 Vectors.

```python
from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors

vector_store = AmazonS3Vectors.from_documents(
            chunks,
            vector_bucket_name=S3_VECTOR_BUCKET_NAME,
            index_name=INDEX_NAME,
            embeddings,
        )
```

### AmazonS3Vectors as Retriever

`AmazonS3VectorsRetriever` class initialized from this AmazonS3Vectors.

```python
retriever = vector_store.as_retriever()
```
