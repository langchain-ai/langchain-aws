# AGENTS.md

This file provides guidance for AI coding assistants when working with code in this repository.

## Project Overview

This is a monorepo containing LangChain and LangGraph integrations for AWS services. It replaces and expands upon AWS components previously found in `langchain-community`. The repository contains two main packages:

- `langchain-aws`: Core LangChain integrations for AWS services (Bedrock, SageMaker, Kendra, etc.)
- `langgraph-checkpoint-aws`: LangGraph checkpointing solutions using AWS services

## Development Commands

### For langchain-aws (libs/aws/)

**Setup:**

```bash
cd libs/aws
make install_dev
```

**Testing:**

```bash
make tests                    # Run all unit tests
make test TEST_FILE=path      # Run specific test file
make integration_tests        # Run all integration tests
make test_watch              # Interactive test watching
```

**Code Quality:**

```bash
make lint                    # Check code with ruff
make format                  # Format code with ruff
make spell_check            # Check spelling
make check_imports          # Validate imports
```

**Coverage:**

```bash
make coverage_tests                    # Unit test coverage
make coverage_integration_tests        # Integration test coverage
```

### For langgraph-checkpoint-aws (libs/langgraph-checkpoint-aws/)

**Setup:**

```bash
cd libs/langgraph-checkpoint-aws
make install_all
```

**Testing and Linting:**

```bash
make tests                   # Unit tests
make integration_tests       # Integration tests
make lint                   # Linting
make format                 # Formatting
```

## Architecture

### langchain-aws Structure

The package is organized by AWS service integration type:

- **LLMs**: `llms/` - Bedrock LLM and SageMaker Endpoint integrations
- **Chat Models**: `chat_models/` - ChatBedrock, ChatBedrockConverse, SageMaker chat models
- **Embeddings**: `embeddings/` - BedrockEmbeddings for vector generation
- **Retrievers**: `retrievers/` - Kendra, Knowledge Bases, S3 Vectors retrieval
- **Vector Stores**: `vectorstores/` - InMemoryDB and S3 Vectors storage
- **Graphs**: `graphs/` - Neptune graph database integrations
- **Tools**: `tools/` - AgentCore browser and code interpreter toolkits
- **Agents**: `agents/` - Bedrock Agents integration utilities
- **Chains**: `chains/` - Neptune Cypher and SPARQL QA chains

### Key Components

- **Unified AWS Client Configuration**: Automatic boto3 client patching with framework user-agent headers
- **Debug Logging**: Controlled via `LANGCHAIN_AWS_DEBUG` environment variable
- **Optional Dependencies**: Tools functionality requires extra dependencies (`pip install langchain-aws[tools]`)

### langgraph-checkpoint-aws Structure

- **Core Savers**: Async and sync checkpoint saving implementations
- **AgentCore Integration**: Memory service integration for LangGraph agents
- **Session Management**: AWS Bedrock session management integration

## Testing Strategy

- **Unit Tests**: Fast, isolated tests in `tests/unit_tests/`
- **Integration Tests**: AWS service integration tests in `tests/integration_tests/`
- **Snapshot Testing**: Uses `syrupy` for response validation
- **Coverage Requirements**: Comprehensive coverage with branch testing

## Important Notes

- Both packages use uv for dependency management
- Code style enforced by ruff (formatting and linting)
- MyPy used for type checking with strict configuration
- Import validation ensures proper module organization
- AWS credentials required for integration tests
- Optional tool dependencies must be explicitly installed for browser/code interpreter functionality
