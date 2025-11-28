# Global development guidelines for the LangChain monorepo

This document provides context to understand the LangChain Python project and assist with development.

## Project architecture and context

This is a monorepo containing LangChain and LangGraph integrations for AWS services. The repository contains two main packages:

- `langchain-aws`: Core LangChain integrations for AWS services (Bedrock, SageMaker, Kendra, etc.)
- `langgraph-checkpoint-aws`: LangGraph checkpointing solutions using AWS services

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

### Development tools & commands**

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `make` – Task runner for common development commands. Feel free to look at the `Makefile` for available commands and usage patterns.
- `ruff` – Fast Python linter and formatter
- `mypy` – Static type checking
- `pytest` – Testing framework

This monorepo uses `uv` for dependency management. Local development uses editable installs: `[tool.uv.sources]`

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

#### Key config files

- pyproject.toml: Main workspace configuration with dependency groups
- uv.lock: Locked dependencies for reproducible builds
- Makefile: Development tasks

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Refer to .github/workflows/pr_lint for allowed types and scopes.

#### Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the known_users set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)
- [ ] Does the test suite fail if your new logic is broken?

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")

## Additional resources

- **Documentation:** <https://docs.langchain.com/oss/python/langchain/overview> and source at <https://github.com/langchain-ai/docs> or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [`.github/CONTRIBUTING.md`](https://docs.langchain.com/oss/python/contributing/overview)

## Important Notes

- Import validation ensures proper module organization
- AWS credentials required for integration tests
- Optional tool dependencies must be explicitly installed for browser/code interpreter functionality
