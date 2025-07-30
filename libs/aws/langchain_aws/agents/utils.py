from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

import boto3
from botocore.client import Config
from langchain_core.tools import BaseTool

from langchain_aws.agents.types import (
    _DEFAULT_ACTION_GROUP_NAME,
    BedrockAgentAction,
    BedrockAgentFinish,
    GuardrailConfiguration,
    OutputType,
)

logger = logging.getLogger(__name__)

# Bedrock agents version is being used to specify the version of the agent impl on langchain.
# This should be updated on any major change where we want to detect usage increase/decrease from the change.
__bedrock_agents_version__ = "0.1.0"
SDK_USER_AGENT = f"LangChainAWS#Agents#{__bedrock_agents_version__}"

# Set default client parameters
DEFAULT_CONFIG_VALUES = {
    "connect_timeout": 120,
    "read_timeout": 120,
    "retries": {"max_attempts": 0},
}


def get_boto_session(
    credentials_profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    config: Optional[Config] = None,
) -> Any:
    """
    Construct the boto3 session

    Args:
        credentials_profile_name: AWS profile name to use for credentials
        region_name: AWS region to use
        endpoint_url: Custom endpoint URL to use
        config: Optional boto3 Config object
    """
    if credentials_profile_name:
        session = boto3.Session(profile_name=credentials_profile_name)
    else:
        # use default credentials
        session = boto3.Session()

    # If a custom config is provided, ensure our defaults are maintained
    config = config or Config(**DEFAULT_CONFIG_VALUES)
    # Set default values if not present in custom config
    for key, default_value in DEFAULT_CONFIG_VALUES.items():
        if getattr(config, key, None) is None:
            setattr(config, key, default_value)

    # Update user agent
    existing_user_agent = getattr(config, "user_agent_extra", "") or ""
    config.user_agent_extra = (
        f"{existing_user_agent} x-client-framework:langchain-aws "
        f"md/sdk_user_agent/{SDK_USER_AGENT}".strip()
    )
    client_params = {"config": config}

    if region_name:
        client_params["region_name"] = region_name
    if endpoint_url:
        client_params["endpoint_url"] = endpoint_url

    return client_params, session


def parse_agent_response(response: Any) -> OutputType:
    """
    Parses the raw response from Bedrock Agent

    Args:
        response: The raw response from Bedrock Agent

    Returns
        Either a BedrockAgentAction or a BedrockAgentFinish
    """
    response_text = ""
    event_stream = response["completion"]
    session_id = response["sessionId"]
    trace_log_elements = []
    files = []
    for event in event_stream:
        if "trace" in event:
            trace_log_elements.append(event["trace"])

        if "returnControl" in event:
            response_text = json.dumps(event)
            break

        if "chunk" in event:
            response_text = event["chunk"]["bytes"].decode("utf-8")

        if "files" in event:
            files = event["files"]["files"]

    trace_log = json.dumps(trace_log_elements)

    agent_finish = BedrockAgentFinish(
        return_values={"output": response_text, "files": files},
        log=response_text,
        session_id=session_id,
        trace_log=trace_log,
    )
    if not response_text:
        return agent_finish

    if "returnControl" not in response_text:
        return agent_finish

    return_control = json.loads(response_text).get("returnControl")
    if not return_control:
        return agent_finish

    invocation_inputs = return_control.get("invocationInputs")
    if not invocation_inputs:
        return agent_finish

    try:
        invocation_input = invocation_inputs[0].get("functionInvocationInput", {})
        action_group = invocation_input.get("actionGroup", "")
        function = invocation_input.get("function", "")
        parameters = invocation_input.get("parameters", [])
        parameters_json = {}
        for parameter in parameters:
            parameters_json[parameter.get("name")] = parameter.get("value", "")

        tool = f"{action_group}::{function}"
        if _DEFAULT_ACTION_GROUP_NAME in action_group:
            tool = f"{function}"
        return [
            BedrockAgentAction(
                tool=tool,
                tool_input=parameters_json,
                log=response_text,
                session_id=session_id,
                trace_log=trace_log,
                invocation_id=return_control.get("invocationId"),
            )
        ]
    except IndexError as ex:
        raise IndexError(f"No invocation inputs available: {repr(ex)}") from ex
    except KeyError as ex:
        raise KeyError(
            f"Missing required key for BedrockAgentAction in agent response: {repr(ex)}"
        ) from ex
    except (ValueError, TypeError) as ex:
        raise ValueError(
            f"Invalid arguments for BedrockAgentAction: {repr(ex)}"
        ) from ex
    except Exception as ex:
        raise Exception(
            "Exception encountered while parsing tool request {}".format(repr(ex))
        ) from ex


def _create_bedrock_agent(
    bedrock_client: Any,
    agent_name: str,
    agent_resource_role_arn: str,
    instruction: str,
    foundation_model: str,
    client_token: Optional[str] = None,
    customer_encryption_key_arn: Optional[str] = None,
    description: Optional[str] = None,
    guardrail_configuration: Optional[GuardrailConfiguration] = None,
    idle_session_ttl_in_seconds: Optional[int] = None,
) -> Union[str, None]:
    """
    Creates the bedrock agent
    """
    create_agent_request: dict = {
        "agentName": agent_name,
        "agentResourceRoleArn": agent_resource_role_arn,
        "foundationModel": foundation_model,
        "instruction": instruction,
    }

    if description:
        create_agent_request["description"] = description

    if client_token:
        create_agent_request["clientToken"] = client_token

    if customer_encryption_key_arn:
        create_agent_request["customerEncryptionKeyArn"] = customer_encryption_key_arn

    if guardrail_configuration is not None:
        create_agent_request["guardrailConfiguration"] = {
            "guardrailIdentifier": guardrail_configuration["guardrail_identifier"],
            "guardrailVersion": guardrail_configuration["guardrail_version"] or "DRAFT",
        }

    if idle_session_ttl_in_seconds:
        create_agent_request["idleSessionTTLInSeconds"] = idle_session_ttl_in_seconds

    try:
        create_agent_response = bedrock_client.create_agent(**create_agent_request)
    except Exception as ex:
        # See full exception list here: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_CreateAgent.html#API_agent_CreateAgent_Errors
        raise Exception(
            "Exception encountered with create_agent {}".format(repr(ex))
        ) from ex

    request_id = create_agent_response.get("ResponseMetadata", {}).get("RequestId", "")
    logger.info(f"Create bedrock agent call successful with request id: {request_id}")
    agent_id = create_agent_response["agent"]["agentId"]
    create_agent_start_time = time.time()
    while time.time() - create_agent_start_time < 10:
        agent_creation_status = (
            bedrock_client.get_agent(agentId=agent_id)
            .get("agent", {})
            .get("agentStatus", {})
        )
        if agent_creation_status == "NOT_PREPARED":
            return agent_id
        else:
            time.sleep(2)

    logger.error(f"Failed to create bedrock agent {agent_id}")
    raise TimeoutError(
        f"Failed to create bedrock agent within 10s. AgentId: {agent_id}"
    )


def _get_action_group_and_function_names(tool: BaseTool) -> Tuple[str, str]:
    """
    Convert the LangChain 'Tool' into Bedrock Action Group name and Function name
    """
    action_group_name = _DEFAULT_ACTION_GROUP_NAME
    function_name = tool.name
    tool_name_split = tool.name.split("::")
    if len(tool_name_split) > 1:
        action_group_name = tool_name_split[0]
        function_name = tool_name_split[1]
    return action_group_name, function_name


def _create_bedrock_action_groups(
    bedrock_client: Any,
    agent_id: str,
    tools: List[BaseTool],
    enable_human_input: Optional[bool] = False,
    enable_code_interpreter: Optional[bool] = False,
) -> None:
    """Create the bedrock action groups for the agent"""

    tools_by_action_group = defaultdict(list)
    for tool in tools:
        action_group_name, function_name = _get_action_group_and_function_names(tool)
        tools_by_action_group[action_group_name].append(tool)

    for action_group_name, functions in tools_by_action_group.items():
        bedrock_client.create_agent_action_group(
            actionGroupName=action_group_name,
            actionGroupState="ENABLED",
            actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            functionSchema={
                "functions": [_tool_to_function(function) for function in functions]
            },
            agentId=agent_id,
            agentVersion="DRAFT",
        )

    if enable_human_input:
        bedrock_client.create_agent_action_group(
            actionGroupName="UserInputAction",
            parentActionGroupSignature="AMAZON.UserInput",
            actionGroupState="ENABLED",
            agentId=agent_id,
            agentVersion="DRAFT",
        )

    if enable_code_interpreter:
        bedrock_client.create_agent_action_group(
            actionGroupName="CodeInterpreterAction",
            parentActionGroupSignature="AMAZON.CodeInterpreter",
            actionGroupState="ENABLED",
            agentId=agent_id,
            agentVersion="DRAFT",
        )


def _tool_to_function(tool: BaseTool) -> dict:
    """
    Convert LangChain tool to a Bedrock function schema
    """
    _, function_name = _get_action_group_and_function_names(tool)
    function_parameters = {}
    for arg_name, arg_details in tool.args.items():
        function_parameters[arg_name] = {
            "description": arg_details.get(
                "description", arg_details.get("title", arg_name)
            ),
            "type": arg_details.get("type", "string"),
            "required": not bool(arg_details.get("default", None)),
        }
    return {
        "description": tool.description,
        "name": function_name,
        "parameters": function_parameters,
    }


def _prepare_agent(bedrock_client: Any, agent_id: str) -> None:
    """
    Prepare the agent for invocations
    """
    bedrock_client.prepare_agent(agentId=agent_id)
    prepare_agent_start_time = time.time()
    while time.time() - prepare_agent_start_time < 10:
        agent_status = bedrock_client.get_agent(agentId=agent_id)
        if agent_status.get("agent", {}).get("agentStatus", "") == "PREPARED":
            return
        else:
            time.sleep(2)
    raise Exception(f"Timed out while preparing the agent with id {agent_id}")


def _get_bedrock_agent(bedrock_client: Any, agent_name: str) -> Any:
    """
    Get the agent by name
    """
    next_token = None
    while True:
        if next_token:
            list_agents_response = bedrock_client.list_agents(
                maxResults=1000, nextToken=next_token
            )
        else:
            list_agents_response = bedrock_client.list_agents(maxResults=1000)
        agent_summaries = list_agents_response.get("agentSummaries", [])
        next_token = list_agents_response.get("nextToken")
        agent_summary = next(
            (x for x in agent_summaries if x.get("agentName") == agent_name), None
        )
        if agent_summary:
            return agent_summary
        if next_token is None:
            return None
