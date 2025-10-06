# LangGraph Checkpoint AWS
A custom LangChain checkpointer implementation that uses Bedrock AgentCore Memory to enable stateful and resumable LangGraph agents through efficient state persistence and retrieval.

## Overview
This package provides a custom checkpointing solution for LangGraph agents using AWS Bedrock AgentCore Memory Service. It enables:
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
langgraph >=0.2.55
boto3 >=1.39.7
```

## Usage - Checkpointer

```python
# Import LangGraph and LangChain components
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# Import the AgentCoreMemory integrations
from langgraph_checkpoint_aws import AgentCoreMemorySaver

REGION = "us-west-2"
MEMORY_ID = "YOUR_MEMORY_ID"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Initialize checkpointer for state persistence. No additional setup required.
# Sessions will be saved and persisted for actor_id/session_id combinations
checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)

# Initialize LLM
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer, # AgentCoreMemorySaver we created above
)

# Specify config at runtime for ACTOR and SESSION
config = {
    "configurable": {
        "thread_id": "session-1", # REQUIRED: This maps to Bedrock AgentCore session_id under the hood
        "actor_id": "react-agent-1", # REQUIRED: This maps to Bedrock AgentCore actor_id under the hood
    }
}

# Invoke the agent
response = graph.invoke(
    {"messages": [("human", "I like sushi with tuna. In general seafood is great.")]},
    config=config
)
```

## Usage - Memory Store

```python
# Import LangGraph and LangChain components
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langgraph_checkpoint_aws import (
    AgentCoreMemoryStore
)

REGION = "us-west-2"
MEMORY_ID = "YOUR_MEMORY_ID"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Initialize store for saving and searching over long term memories
# such as preferences and facts across sessions
store = AgentCoreMemoryStore(MEMORY_ID, region_name=REGION)

# Pre-model hook runs and saves messages of your choosing to AgentCore Memory
# for async processing and extraction
def pre_model_hook(state, config: RunnableConfig, *, store: BaseStore):
    """Hook that runs pre-LLM invocation to save the latest human message"""
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    
    # Saving the message to the actor and session combination that we get at runtime
    namespace = (actor_id, thread_id)
    
    messages = state.get("messages", [])
    # Save the last human message we see before LLM invocation
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            store.put(namespace, str(uuid.uuid4()), {"message": msg})
            break
            
    # OPTIONAL: Retrieve user preferences based on the last message and append to state
    # user_preferences_namespace = ("preferences", actor_id)
    # preferences = store.search(user_preferences_namespace, query=msg.content, limit=5)
    # # Add to input messages as needed
    
    return {"llm_input_messages": messages}

# Initialize LLM
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a pre-built langgraph agent (configurations work for custom agents too)
graph = create_react_agent(
    model=llm,
    tools=[],
    pre_model_hook=pre_model_hook,
)

# Specify config at runtime for ACTOR and SESSION
config = {
    "configurable": {
        "thread_id": "session-1", # REQUIRED: This maps to Bedrock AgentCore session_id under the hood
        "actor_id": "react-agent-1", # REQUIRED: This maps to Bedrock AgentCore actor_id under the hood
    }
}

# Invoke the agent
response = graph.invoke(
    {"messages": [("human", "I like sushi with tuna. In general seafood is great.")]},
    config=config
)
```
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
                "bedrock-agentcore:CreateEvent",
                "bedrock-agentcore:ListEvents",
                "bedrock-agentcore:GetEvent",
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}
```

## Bedrock Session Saver (Alternative Implementation)

This package also provides an alternative checkpointing solution using AWS Bedrock Session Management Service:

### Usage

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

### Configuration Options

`BedrockSessionSaver` and `AsyncBedrockSessionSaver` accepts the following parameters:

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

### Additional AWS permissions for Session Saver:

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
