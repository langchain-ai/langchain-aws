# LangGraph AgentCore Checkpoint
A custom LangGraph checkpointer implementation that uses AWS Bedrock AgentCore Memory to enable stateful and resumable LangGraph agents through efficient state persistence and retrieval.

## Overview
This package provides a custom checkpointing solution for LangGraph agents using AWS Bedrock AgentCore Memory. It enables:
1. Stateful conversations and interactions
2. Resumable agent sessions 
3. Efficient state persistence and retrieval 
4. Seamless integration with AWS Bedrock AgentCore

## Installation
You can install the package using pip:

```bash
pip install langgraph-agentcore-checkpoint
```
Or with Poetry:
```bash
poetry add langgraph-agentcore-checkpoint
```

## Requirements
```text
Python >=3.9
langgraph-checkpoint >=2.0.0
langgraph >=0.2.55
boto3 >=1.37.3
```

## Usage

```python
from langgraph.graph import StateGraph
from langgraph_agentcore_checkpoint import AgentCoreMemorySaver

# Initialize the saver
memory_saver = AgentCoreMemorySaver(
    memory_id="your-memory-id",
    actor_id="agent1",
    region_name="us-west-2"
)

# Use with LangGraph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=memory_saver)
config = {"configurable": {"thread_id": "session_id"}}
graph.invoke(1, config)
```

## Configuration Options

`AgentCoreMemorySaver` accepts the following parameters:

```python
def __init__(
    memory_id: str,
    actor_id: str = "agent",
    *,
    serde: Optional[SerializerProtocol] = None,
    **boto3_kwargs: Any,
) -> None:
```

- `memory_id`: The AgentCore memory identifier
- `actor_id`: Actor identifier for the memory operations
- `serde`: Optional serializer protocol
- `**boto3_kwargs`: Additional boto3 client arguments

## AWS Configuration

Ensure you have AWS credentials configured using one of these methods:
1. Environment variables
2. AWS credentials file (~/.aws/credentials)
3. IAM roles
4. Direct credential injection via constructor parameters

## Required AWS permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateMemory",
        "bedrock-agentcore:GetMemory",
        "bedrock-agentcore:UpdateMemory",
        "bedrock-agentcore:DeleteMemory",
        "bedrock-agentcore:ListMemories",
        "bedrock-agentcore:CreateEvent",
        "bedrock-agentcore:GetEvent",
        "bedrock-agentcore:ListEvents",
        "bedrock-agentcore:DeleteEvent"
      ],
      "Resource": "*"
    }
  ]
}
```

## Development
Setting Up Development Environment

* Clone the repository:
```bash
git clone <repository-url>
cd libs/langgraph-agentcore-checkpoint
```
* Install development dependencies:
```bash
make install_all
```

## Running Tests
```bash
make tests         # Run all tests
make test_watch   # Run tests in watch mode
```

## Code Quality
```bash
make lint           # Run linter
make format         # Format code
make spell_check    # Check spelling
```

## Clean Up
```bash
make clean          # Remove all generated files
```

## Security Considerations
* Never commit AWS credentials
* Use environment variables or AWS IAM roles for authentication
* Follow AWS security best practices
* Use IAM roles and temporary credentials when possible
* Implement proper access controls for memory management

## Contributing
* Fork the repository
* Create a feature branch
* Make your changes
* Run tests and linting
* Submit a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
* LangChain team for the base LangGraph framework
* AWS Bedrock team for the AgentCore Memory service