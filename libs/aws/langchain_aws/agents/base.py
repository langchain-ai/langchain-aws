from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import botocore
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool

_DEFAULT_ACTION_GROUP_NAME = "DEFAULT_AG_"
_TEST_AGENT_ALIAS_ID = "TSTALIASID"


def get_bedrock_agents_runtime_client():
    bedrock_config = botocore.client.Config(
        connect_timeout=120,
        read_timeout=120,
        retries={"max_attempts": 3},
        region_name="us-west-2",
    )
    return boto3.client("bedrock-agent-runtime", config=bedrock_config)


def get_bedrock_agents_client():
    bedrock_config = botocore.client.Config(
        connect_timeout=120,
        read_timeout=120,
        retries={"max_attempts": 3},
        region_name="us-west-2",
    )
    return boto3.client("bedrock-agent", config=bedrock_config)


def parse_agent_response(response: Any) -> OutputType:
    response_text = ""
    event_stream = response["completion"]
    for event in event_stream:
        if "returnControl" in event:
            response_text = json.dumps(event)
            break

        if "chunk" in event:
            response_text = event["chunk"]["bytes"].decode("utf-8")

    agent_finish = AgentFinish({"output": response_text}, log=response_text)
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
        return [AgentAction(tool=tool, tool_input=parameters_json, log=response_text)]
    except Exception as ex:
        raise Exception("Parse exception encountered {}".format(repr(ex)))


OutputType = Union[List[AgentAction], AgentFinish]


def _create_bedrock_agent(bedrock_client, agent_name, agent_resource_role_arn, instructions, model):
    create_agent_response = bedrock_client.create_agent(
        agentName=agent_name,
        agentResourceRoleArn=agent_resource_role_arn,
        foundationModel=model,
        instruction=instructions
    )
    request_id = create_agent_response.get('ResponseMetadata', {}).get('RequestId', '')
    logging.info(f'Create bedrock agent call successful with request id: {request_id}')
    agent_id = create_agent_response['agent']['agentId']
    create_agent_start_time = time.time()
    while time.time() - create_agent_start_time < 10:
        agent_creation_status = bedrock_client.get_agent(agentId=agent_id).get('agent', {}).get('agentStatus', {})
        if agent_creation_status == 'NOT_PREPARED':
            return agent_id
        else:
            time.sleep(2)

    logging.error(f'Failed to create bedrock agent {agent_id}')
    raise


def _tool_to_action_group_function(tool: BaseTool) -> Tuple[str, str]:
    action_group_name = _DEFAULT_ACTION_GROUP_NAME
    function_name = tool.name
    tool_name_split = tool.name.split("::")
    if len(tool_name_split) > 1:
        action_group_name = tool_name_split[0]
        function_name = tool_name_split[1]
    return action_group_name, function_name


def _create_bedrock_action_groups(bedrock_client, agent_id, tools):
    tools_by_action_group = defaultdict(list)
    for tool in tools:
        action_group_name, function_name = _tool_to_action_group_function(tool)
        tools_by_action_group[action_group_name].append(tool)
    for action_group_name, functions in tools_by_action_group.items():
        bedrock_client.create_agent_action_group(
            actionGroupName=action_group_name,
            actionGroupState="ENABLED",
            actionGroupExecutor={
                'customControl': 'RETURN_CONTROL'
            },
            functionSchema={
                'functions': [_tool_to_function(function) for function in functions]
            },
            agentId=agent_id,
            agentVersion='DRAFT',
        )


def _tool_to_function(tool: BaseTool):
    _, function_name = _tool_to_action_group_function(tool)
    function_parameters = {}
    for arg_name, arg_details in tool.args.items():
        function_parameters[arg_name] = {
            'description': arg_details.get('description', arg_details.get('title', arg_name)),
            'type': arg_details.get('type', 'string'),
            'required': not bool(arg_details.get('default', None))
        }
    return {
        'description': tool.description,
        'name': function_name,
        'parameters': function_parameters
    }


def _prepare_agent(bedrock_client, agent_id):
    bedrock_client.prepare_agent(agentId=agent_id)
    prepare_agent_start_time = time.time()
    while time.time() - prepare_agent_start_time < 10:
        agent_status = bedrock_client.get_agent(agentId=agent_id)
        if agent_status.get('agent', {}).get('agentStatus', '') == 'PREPARED':
            return
        else:
            time.sleep(2)
    raise Exception(f'Timed out while preparing the agent with id {agent_id}')


class BedrockAgentsRunnable(RunnableSerializable[Dict, OutputType]):
    client: Any = Field(default_factory=get_bedrock_agents_runtime_client)
    agent_alias_id: str = _TEST_AGENT_ALIAS_ID
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    enable_trace: bool = False

    @root_validator
    def validate_agent(cls, values: dict) -> dict:
        # Can create and prepare agents here
        # Set agent_id, agent_alias_id, session_id etc.
        if not values["session_id"]:
            values["session_id"] = uuid.uuid4().hex

        return values

    @classmethod
    def create_agent(cls,
                     agent_name: str,
                     agent_resource_role_arn: str,
                     model: str,
                     instructions: str,
                     tools: List[BaseTool],
                     agent_alias_id: str = _TEST_AGENT_ALIAS_ID,
                     **kwargs: Any,
                     ) -> BedrockAgentsRunnable:
        try:
            bedrock_client = get_bedrock_agents_client()
            agent_id = _create_bedrock_agent(bedrock_client, agent_name, agent_resource_role_arn, instructions, model)
            _create_bedrock_action_groups(bedrock_client, agent_id, tools)
            _prepare_agent(bedrock_client, agent_id)

            return cls(
                agent_id=agent_id,
                **kwargs
            )
        except Exception as exception:
            logging.error(f'Error in create agent call: {exception}')
            raise exception

    def invoke(
            self, input: Dict, config: Optional[RunnableConfig] = None
    ) -> OutputType:
        config = ensure_config(config)
        agent_input = {
            "agentId": self.agent_id,
            "agentAliasId": self.agent_alias_id,
            "enableTrace": self.enable_trace,
            "sessionId": self.session_id,
        }
        if input.get("intermediate_steps"):
            session_state = self._parse_intermediate_steps(
                input.get("intermediate_steps")  # type: ignore[arg-type]
            )
            if session_state is not None:
                agent_input["sessionState"] = session_state
        else:
            agent_input["inputText"] = input.get("input", "")

        output = self.client.invoke_agent(**agent_input)
        return parse_agent_response(output)

    def _parse_intermediate_steps(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Any:
        last_step = max(0, len(intermediate_steps) - 1)
        action = intermediate_steps[last_step][0]
        tool_invoked = action.tool
        messages = action.messages

        if tool_invoked:
            action_group_name = _DEFAULT_ACTION_GROUP_NAME
            function_name = tool_invoked
            tool_name_split = tool_invoked.split("::")
            if len(tool_name_split) > 1:
                action_group_name = tool_name_split[0]
                function_name = tool_name_split[1]

            if messages:
                last_message = max(0, len(messages) - 1)
                message = messages[last_message]
                if type(message) is AIMessage:
                    response = intermediate_steps[last_step][1]
                    session_state = {
                        "invocationId": json.loads(message.content)  # type: ignore[arg-type]
                        .get("returnControl", {})
                        .get("invocationId", ""),
                        "returnControlInvocationResults": [
                            {
                                "functionResult": {
                                    "actionGroup": action_group_name,
                                    "function": function_name,
                                    "responseBody": {"TEXT": {"body": response}},
                                }
                            }
                        ],
                    }

                    return session_state
