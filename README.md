# 🦜️🔗 LangChain 🤝 Amazon Web Services (AWS)

This monorepo provides LangChain and LangGraph components for various AWS services. It aims to replace and expand upon the existing LangChain AWS components found in the `langchain-community` package in the LangChain repository.

The following packages are hosted in this repository:
- `langchain-aws` ([PyPi](https://pypi.org/project/langchain-aws/))
- `langgraph-checkpoint-aws` ([PyPi](https://pypi.org/project/langgraph-checkpoint-aws/))

## Features

### LangChain
- **LLMs**: Includes LLM classes for AWS services like [Bedrock](https://aws.amazon.com/bedrock) and [SageMaker Endpoints](https://aws.amazon.com/sagemaker/deploy/), allowing you to leverage their language models within LangChain.
- **Retrievers**: Supports retrievers for services like [Amazon Kendra](https://aws.amazon.com/kendra/) and [KnowledgeBases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/), enabling efficient retrieval of relevant information in your RAG applications.
- **Graphs**: Provides components for working with [AWS Neptune](https://aws.amazon.com/neptune/) graphs within LangChain.
- **Agents**: Includes Runnables to support [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/), allowing you to leverage Bedrock Agents within LangChain and LangGraph.

### LangGraph
- **Checkpointers**: Provides a custom checkpointing solution for LangGraph agents using the [AWS Bedrock Session Management Service](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html).

...and more to come. This repository will continue to expand and offer additional components for various AWS services as development progresses.

**Note**: This repository will replace all AWS integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Installation

You can install the `langchain-aws` package from PyPI.

```bash
pip install langchain-aws
```

The `langgraph-checkpoint-aws` package can also be installed from PyPI.

```bash
pip install langgraph-checkpoint-aws
```

## Usage

### LangChain

Here's a simple example of how to use the `langchain-aws` package.

```python
from langchain_aws import ChatBedrock

# Initialize the Bedrock chat model
llm = ChatBedrock(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    beta_use_converse_api=True
)

# Invoke the llm
response = llm.invoke("Hello! How are you today?")
print(response)
```

For more detailed usage examples and documentation, please refer to the [LangChain docs](https://python.langchain.com/docs/integrations/platforms/aws/).

### LangGraph

You can find usage examples for `langgraph-checkpoint-aws` [here](https://github.com/michaelnchin/langchain-aws/blob/main/libs/langgraph-checkpoint-aws/README.md#usage).

## Contributing

We welcome contributions to this repository! To get started, please follow the contribution guide for your specific project of interest:

- For `langchain-aws`, see [here](https://github.com/langchain-ai/langchain-aws/blob/main/libs/aws/CONTRIBUTING.md).
- For `langgraph-checkpointer-aws`, see [here](https://github.com/langchain-ai/langchain-aws/blob/main/libs/langgraph-checkpoint-aws/CONTRIBUTING.md).

Each guide provides detailed instructions on how to set up the project for development and guidance on how to contribute effectively.

## License

This project is licensed under the [MIT License](LICENSE).
