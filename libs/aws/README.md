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
