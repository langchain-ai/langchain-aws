# 🦜️🔗 LangChain 🤝 Amazon Web Services (AWS)

This repository provides LangChain components for various AWS services. It aims to replace and expand upon the existing LangChain AWS components found in the `langchain-community` package in the LangChain repository.

## Features

- **LLMs**: Includes LLM classes for AWS services like [Bedrock](https://aws.amazon.com/bedrock) and [SageMaker Endpoints](https://aws.amazon.com/sagemaker/deploy/), allowing you to leverage their language models within LangChain.
- **Retrievers**: Supports retrievers for services like [Amazon Kendra](https://aws.amazon.com/kendra/) and [KnowledgeBases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/), enabling efficient retrieval of relevant information in your RAG applications.
- **Graphs**: Provides components for working with [AWS Neptune](https://aws.amazon.com/neptune/) graphs within LangChain.
- **More to come**: This repository will continue to expand and offer additional components for various AWS services as development progresses.

**Note**: This repository will replace all AWS integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Installation

You can install the `langchain-aws` package from PyPI.

```bash
pip install langchain-aws
```

## Usage

Here's a simple example of how to use the `langchain-aws` package.

```python
from langchain_aws import BedrockLLM

# Initialize the Bedrock LLM
llm = BedrockLLM(
    model_id="anthropic.claude-v2:1"
)

# Invoke the llm
response = llm.invoke("Hello! How are you today?")
print(response)
```

For more detailed usage examples and documentation, please refer to the [LangChain docs](https://python.langchain.com/docs/integrations/platforms/aws/).

## Contributing

We welcome contributions to this project! Please follow the [contribution guide](https://github.com/langchain-ai/langchain-aws/blob/main/.github/CONTRIBUTING.md) for instructions to setup the project for development and guidance on how to contribute effectively.

## License

This project is licensed under the [MIT License](LICENSE).
