# 🦜️🔗 LangChain 🤝 Amazon Web Services (AWS)

This monorepo provides LangChain and LangGraph components for various AWS services. It aims to replace and expand upon the existing LangChain AWS components found in the `langchain-community` package in the LangChain repository.

The following packages are hosted in this repository:

- `langchain-aws` ([PyPI](https://pypi.org/project/langchain-aws/))
- `langgraph-checkpoint-aws` ([PyPI](https://pypi.org/project/langgraph-checkpoint-aws/))

## Features

### LangChain

- **LLMs**: Includes LLM classes for AWS services like [Bedrock](https://aws.amazon.com/bedrock) and [SageMaker Endpoints](https://aws.amazon.com/sagemaker/deploy/), allowing you to leverage their language models within LangChain.
- **VectorStores**: Supports vectorstores for services like [Amazon MemoryDB](https://aws.amazon.com/memorydb/), [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/), and [AWS ElastiCache for Valkey](https://aws.amazon.com/elasticache/), providing efficient and scalable vector database for your applications.
- **Retrievers**: Supports retrievers for services like [Amazon Kendra](https://aws.amazon.com/kendra/) and [KnowledgeBases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/), enabling efficient retrieval of relevant information in your RAG applications.
- **Graphs**: Provides components for working with [AWS Neptune](https://aws.amazon.com/neptune/) graphs within LangChain.
- **Agents**: Includes Runnables to support [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/), allowing you to leverage Bedrock Agents within LangChain and LangGraph.
- **Tools**: Includes tools and toolkits to enable use of [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)'s built-in tools with LangChain and LangGraph agents.

### LangGraph

- **Checkpointers**: Provides custom checkpointing solutions for LangGraph agents using several AWS services, including [Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html), [Bedrock Session Management](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html), [DynamoDB](https://aws.amazon.com/dynamodb/), and [ElastiCache Valkey](https://aws.amazon.com/elasticache/).
- **Memory Stores** - Provides memory store solutions for saving, processing, and retrieving intelligent long term memories using services like [Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html) and [ElastiCache Valkey](https://aws.amazon.com/elasticache/).

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

### `langchain-aws`

Here's a simple example of how to use the `langchain-aws` package.

```python
from langchain_aws import ChatBedrockConverse

# Initialize the Bedrock chat model
model = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Invoke the model
response = model.invoke("Hello! How are you today?")
print(response)
```

### AgentCore Tools

```python
from langchain_aws.tools import create_browser_toolkit, create_code_interpreter_toolkit

# Browser automation
browser_toolkit, browser_tools = create_browser_toolkit(region="us-west-2")

# Code execution (async)
code_toolkit, code_tools = await create_code_interpreter_toolkit(region="us-west-2")

# Use with LangGraph agent
agent = create_react_agent(model, tools=browser_tools + code_tools)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Navigate to example.com"}]},
    config={"configurable": {"thread_id": "session-1"}}
)

# Cleanup
await browser_toolkit.cleanup()
await code_toolkit.cleanup()
```

For more detailed usage examples and documentation, please refer to the [LangChain docs](https://python.langchain.com/docs/integrations/platforms/aws/).

### `langgraph-checkpoint-aws`

You can find usage examples for `langgraph-checkpoint-aws` [in the README](https://github.com/langchain-ai/langchain-aws/blob/main/libs/langgraph-checkpoint-aws/README.md).

## Contributing

We welcome contributions to this repository! To get started, please follow the [Contributing Guide](https://github.com/langchain-ai/langchain-aws/blob/main/.github/CONTRIBUTING.md).

This guide provides detailed instructions on how to set up each project for development and guidance on how to contribute effectively.

## License

This project is licensed under the [MIT License](LICENSE).
