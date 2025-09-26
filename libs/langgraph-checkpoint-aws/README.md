# LangGraph Checkpoint AWS
A custom LangChain checkpointer implementation that uses Bedrock Session Management Service to enable stateful and resumable LangGraph agents through efficient state persistence and retrieval.

## Overview
This package provides a custom checkpointing solution for LangGraph agents using AWS Bedrock Session Management Service. It enables:
1. Stateful conversations and interactions
2. Resumable agent sessions
3. Efficient state persistence and retrieval
4. Seamless integration with AWS Bedrock

## Installation
You can install the package using pip:

```bash
pip install langgraph-checkpoint-aws
```
Or with Poetry:
```bash
poetry add langgraph-checkpoint-aws
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
from langgraph_checkpoint_aws.saver import BedrockSessionSaver

# Initialize the saver
session_saver = BedrockSessionSaver(
    region_name="us-west-2",  # Your AWS region
    credentials_profile_name="default",  # Optional: AWS credentials profile
)

# Create a session
session_id = session_saver.session_client.create_session().session_id

# Use with LangGraph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=session_saver)
config = {"configurable": {"thread_id": session_id}}
graph.invoke(1, config)
```


You can also invoke the graph asynchronously:

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_aws.async_saver import AsyncBedrockSessionSaver

# Initialize the saver
session_saver = AsyncBedrockSessionSaver(
    region_name="us-west-2",  # Your AWS region
    credentials_profile_name="default",  # Optional: AWS credentials profile
)

# Create a session
session_create_response = await session_saver.session_client.create_session()
session_id = session_create_response.session_id

# Use with LangGraph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

graph = builder.compile(checkpointer=session_saver)
config = {"configurable": {"thread_id": session_id}}
await graph.ainvoke(1, config)
```

## Configuration Options

`BedrockSessionSaver` and `AsyncBedrockSessionSaver`  accepts the following parameters:

```python
def __init__(
    client: Optional[Any] = None,
    session: Optional[boto3.Session] = None,
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[SecretStr] = None,
    aws_secret_access_key: Optional[SecretStr] = None,
    aws_session_token: Optional[SecretStr] = None,
    endpoint_url: Optional[str] = None,
    config: Optional[Config] = None,
)
```

- `client`: boto3 Bedrock runtime client (e.g. boto3.client("bedrock-agent-runtime"))
- `session`: boto3.Session for custom credentials
- `region_name`: AWS region where Bedrock is available
- `credentials_profile_name`: Name of AWS credentials profile to use
- `aws_access_key_id`: AWS access key ID for authentication
- `aws_secret_access_key`: AWS secret access key for authentication
- `aws_session_token`: AWS session token for temporary credentials
- `endpoint_url`: Custom endpoint URL for the Bedrock service
- `config`: Botocore configuration object
## Development
Setting Up Development Environment

* Clone the repository:
```bash
git clone <repository-url>
cd libs/aws/langgraph-checkpoint-aws
```
* Install development dependencies:
```bash
make install_all
```
* Or install specific components:
```bash
make install_dev        # Basic development tools
make install_test       # Testing tools
make install_lint       # Linting tools
make install_typing     # Type checking tools
make install_codespell  # Spell checking tools
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
            "Sid": "Statement1",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateSession",
                "bedrock:GetSession",
                "bedrock:UpdateSession",
                "bedrock:DeleteSession",
                "bedrock:EndSession",
                "bedrock:ListSessions",
                "bedrock:CreateInvocation",
                "bedrock:ListInvocations",
                "bedrock:PutInvocationStep",
                "bedrock:GetInvocationStep",
                "bedrock:ListInvocationSteps"
            ],
            "Resource": [
                "*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey",
                "kms:DescribeKey"
            ],
            "Resource": "arn:aws:kms:{region}:{account}:key/{kms-key-id}"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:TagResource",
                "bedrock:UntagResource",
                "bedrock:ListTagsForResource"
            ],
            "Resource": "arn:aws:bedrock:{region}:{account}:session/*"
        }
    ]
}
```

## Security Considerations
* Never commit AWS credentials
* Use environment variables or AWS IAM roles for authentication
* Follow AWS security best practices
* Use IAM roles and temporary credentials when possible
* Implement proper access controls for session management

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
* AWS Bedrock team for the session management service
