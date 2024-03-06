# langchain-aws

This package contains the LangChain integration with Bedrock

## Installation

```bash
pip install -U langchain-aws
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

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
