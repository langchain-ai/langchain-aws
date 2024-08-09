import json
import logging
import re
import time
import uuid
from typing import Any, Callable, List, Tuple, Union, Optional, Dict

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.manager import (
    Callbacks,
)
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables import (
    Runnable,
)
from langchain_core.tools import Tool, tool

from langchain_aws.agents.bedrock.agent_client import bedrock_agent, bedrock_agent_runtime, iam, sts

_DEFAULT_ACTION_GROUP_NAME = 'DEFAULT_AG_'


def agent_tool(
        action: str = None,
        action_group: str = None,
        method: str = None,
        **kwargs
) -> Callable:
    def atool(
            *args: Union[str, Callable, Runnable]
    ):
        """
        Decorator for agent tools, which allows for action groups, actions and other BedRockAgent constructs.
        """
        if action_group and action and method:
            args[0].__name__ = method + "::" + action_group + "::" + action
        elif action_group and action:
            args[0].__name__ = action_group + "::" + action
        elif action:
            args[0].__name__ = action + "ActionGroup::" + action
        else:
            args[0].__name__ = args[0].__name__ + "ActionGroup::" + args[0].__name__

        args[0].__doc__ = f"<agent_tool_doc>{args[0].__doc__}</agent_tool_doc>"
        return tool(*args)

    return atool


class BedrockAgentMetadata:
    """
    BedrockAgent Build construct
    """
    __SINGLE_SPACE = ' '
    __EMPTY = ' '

    def __init__(
            self,
            agent_name: str,
            agent_instruction: str,
            agent_region: str,
            agent_description: str = __SINGLE_SPACE,
            agent_tools: List[Tool] = None,
            agent_resource_role_arn: str = None,
            data_privacy: dict = None,
            idle_session_ttl_seconds: int = 900,
            agent_foundation_model: str = 'anthropic.claude-3-sonnet-20240229-v1:0'
    ):
        self.agent_name = agent_name
        self.agent_resource_role_arn = agent_resource_role_arn
        self.agent_instruction = agent_instruction
        self.agent_description = agent_description
        self.data_privacy = data_privacy
        self.idle_session_ttl_seconds = idle_session_ttl_seconds,
        self.agent_foundation_model = agent_foundation_model
        self.agent_region = agent_region
        self.agent_tools = [] if agent_tools is None else agent_tools


class BedrockAgentRuntimeConstruct:
    """
    BedrockAgent Runtime construct
    """

    def __init__(
            self,
            agent_id: str,
            agent_alias_id: str,
            agent_region: str,
            agent_tools: List[Tool] = None,
    ):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.agent_region = agent_region
        self.agent_tools = [] if agent_tools is None else agent_tools


class BedrockAgentManager:
    """
    BedrockAgentManager handling the lifecycle of an agent
    """

    name: str = 'BedrockAgentManager'
    bedrock_buildtime: Any = None
    bedrock_agent_permission: Any = None

    def __init__(
            self,
            **kwargs: {}
    ):
        super().__init__(**kwargs)
        self.bedrock_buildtime = bedrock_agent()
        self.bedrock_agent_permission = iam()
        self.bedrock_agent_sts = sts()
        self.account_id = self.bedrock_agent_sts.get_caller_identity().get('Account')

    def create_agent(
            self,
            bedrock_agent_metadata: BedrockAgentMetadata
    ) -> BedrockAgentRuntimeConstruct:
        """
        Create a Bedrock agent with the given metadata and return it.

        Args:
            bedrock_agent_metadata: Meta data for Bedrock Agents
        Returns:
           BedrockAgent runtime construct with information required during inference
        """
        try:
            create_agent_response = self.bedrock_buildtime.create_agent(
                agentName=bedrock_agent_metadata.agent_name,
                agentResourceRoleArn=bedrock_agent_metadata.agent_resource_role_arn,
                description=bedrock_agent_metadata.agent_description,
                foundationModel=bedrock_agent_metadata.agent_foundation_model,
                instruction=bedrock_agent_metadata.agent_instruction
            )

            request_id = create_agent_response.get('ResponseMetadata', {}).get('RequestId', '')
            logging.info(f'Created AWS Bedrock agent with request id: {request_id}')
            agent_data = create_agent_response['agent']
            agent_id = agent_data['agentId']

            agent_creation_status = self.bedrock_buildtime.get_agent(agentId=agent_id) \
                .get('agent', {}).get('agentStatus', '')
            create_agent_start_time = time.time()
            while agent_creation_status != 'NOT_PREPARED' and (time.time() - create_agent_start_time) < 10:
                time.sleep(2)
                agent_creation_status = self.bedrock_buildtime.get_agent(agentId=agent_id) \
                    .get('agent', {}).get('agentStatus', '')

            if agent_creation_status != 'NOT_PREPARED':
                logging.error(f'Failed to create Bedrock agent {agent_id}')
                raise

            bedrock_agent_runtime_construct = BedrockAgentRuntimeConstruct(
                agent_id=agent_data['agentId'],
                agent_alias_id='TSTALIASID',
                agent_region=bedrock_agent_metadata.agent_region,
                agent_tools=bedrock_agent_metadata.agent_tools
            )
            return bedrock_agent_runtime_construct
        except Exception as exception:
            logging.error(f'Error in create agent call: {exception}')
            raise exception

    def prepare_agent(
            self,
            agent_id: str
    ):
        """
        Prepares a Bedrock agent with the given id.

        Args:
            agent_id: AgentId to prepare
        """
        try:
            prepared_agent_details = self.bedrock_buildtime.prepare_agent(
                agentId=agent_id
            )

            request_id = prepared_agent_details['ResponseMetadata']['RequestId']
            logging.info(f'Prepared AWS Bedrock agent with request id: {request_id}')
            self._wait_on_agent_prepare(agent_id=agent_id)

        except Exception as exception:
            logging.error(f'Error in prepare agent call: {exception}')
            raise exception

    def create_function(
            self,
            bedrock_agent_runtime_construct: BedrockAgentRuntimeConstruct
    ) -> BedrockAgentRuntimeConstruct:
        """
        Create a Bedrock agent function with the given metadata and return it.

        Args:
            bedrock_agent_runtime_construct: BedrockAgent runtime construct with information required during inference
        Returns:
           BedrockAgent runtime construct with information required during inference
        """
        try:
            agent_tools = self._create_function_definitions(bedrock_agent_runtime_construct)

            for action_group_name, action in agent_tools.items():
                create_function_response = self.bedrock_buildtime.create_agent_action_group(
                    actionGroupName=action_group_name,
                    agentId=bedrock_agent_runtime_construct.agent_id,
                    agentVersion=action.get('agentVersion', 'DRAFT'),
                    actionGroupExecutor=action.get('actionGroupExecutor'),
                    actionGroupState='ENABLED',
                    functionSchema=action.get('functionSchema'),
                )

                request_id = create_function_response.get('ResponseMetadata', {}).get('RequestId', '')
                logging.info(f'Created Bedrock agent ActionGroup {action_group_name} '
                             f'and Function with request id: {request_id}')

            return bedrock_agent_runtime_construct
        except Exception as exception:
            logging.error(f'Error in create ActionGroups and Functions: {exception}')
            raise exception

    def delete_agent(
            self,
            agent_id: str,
            agent_resource_role_arn: str
    ):
        """
        Deletes the Bedrock agent with the given id.

        Args:
            agent_id: AgentId to delete
            agent_resource_role_arn: Associated Agent execution role arn
        """
        try:
            delete_agent_response = self.bedrock_buildtime.delete_agent(
                agentId=agent_id,
                skipResourceInUseCheck=True
            )

            request_id = delete_agent_response['ResponseMetadata']['RequestId']
            logging.info(f'Deleted Bedrock agent call request id: {request_id}')

            if delete_agent_response['agentStatus'] != 'DELETING':
                logging.error(f'Failed to delete Bedrock agent {agent_id}')
                return None

            self.delete_agent_role(agent_resource_role_arn)

        except Exception as exception:
            logging.error(f'Error in delete agent call: {exception}')
            raise exception

    def create_agent_role(
            self,
            agent_region,
            foundational_model
    ) -> str:
        """
        Create agent resource role prior to creation of agent, at this point we do not have agentId, keep it as wildcard

        Args:
            agent_region: AWS region in which is the Agent if available
            foundational_model: The model used for inference in AWS BedrockAgents
        Returns:
           Agent execution role arn
        """
        try:
            assume_role_policy_document = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "bedrock.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole",
                        "Condition": {
                            "ArnLike": {
                                "aws:SourceArn": f"arn:aws:bedrock:{agent_region}:{self.account_id}:agent/*"
                            }
                        }
                    }
                ]
            })
            managed_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AmazonBedrockAgentBedrockFoundationModelStatement",
                        "Effect": "Allow",
                        "Action": "bedrock:InvokeModel",
                        "Resource": [
                            f"arn:aws:bedrock:{agent_region}::foundation-model/{foundational_model}"
                        ]
                    }
                ]
            }
            role_name = f'bedrock_agent_{uuid.uuid4()}'
            response = self.bedrock_agent_permission.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=assume_role_policy_document,
                Description='Role for Bedrock Agent'
            )
            self.bedrock_agent_permission.put_role_policy(
                RoleName=role_name,
                PolicyName=f'AmazonBedrockAgentBedrockFoundationModelPolicy_{uuid.uuid4()}',
                PolicyDocument=json.dumps(managed_policy)
            )
            time.sleep(2)
            return response.get('Role', {}).get('Arn', '')

        except Exception as exception:
            logging.error(f'Error in create agent execution role: {exception}')
            raise exception

    def delete_agent_role(
            self,
            agent_resource_role_arn: str
    ):
        """
        Delete agent resource role

        Args:
           agent_resource_role_arn: Associated Agent execution role arn
        """
        try:
            role_name = agent_resource_role_arn.split('/')[-1]
            inline_policies = self.bedrock_agent_permission.list_role_policies(
                RoleName=role_name
            )
            for inline_policy_name in inline_policies.get('PolicyNames', []):
                self.bedrock_agent_permission.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=inline_policy_name
                )
            self.bedrock_agent_permission.delete_role(
                RoleName=role_name
            )
        except Exception as exception:
            logging.error(f'Error in delete agent execution role: {exception}')
            raise exception

    def update_agent_resource_policy(
            self,
            agent_resource_role_arn: str,
            agent_region: str,
            agent_id: str,
    ) -> str:
        """
        Update agent resource role to restrict to the specific agent.

        Args:
            agent_resource_role_arn: Associated Agent execution role arn
            agent_region: AWS region in which is the Agent if available
            agent_id: Id of the Agent whose resource policy will be updated
        Returns:
           Updated Agent execution role arn

        """
        assume_role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock:{agent_region}:{self.account_id}:agent/{agent_id}"
                        }
                    }
                }
            ]
        })
        response = self.bedrock_agent_permission.update_assume_role_policy(
            RoleName=agent_resource_role_arn.split('/')[-1],
            PolicyDocument=assume_role_policy_document
        )
        time.sleep(2)
        return response.get('Role', {}).get('Arn', '')

    def _create_function_definitions(
            self,
            bedrock_agent_runtime_construct: BedrockAgentRuntimeConstruct
    ) -> dict:
        """
        Create a Bedrock agent function construct with the given metadata and return it.

        Args:
            bedrock_agent_runtime_construct: BedrockAgent runtime construct with information required during inference
        Returns:
           dictionary of grouped actions (functions) in an AgentActionGroup
        """
        try:
            agent_tools = bedrock_agent_runtime_construct.agent_tools
            functions = {}

            for bedrock_agent_tool in agent_tools:
                action_group_name, function_name, function_description, function_args_descriptions \
                    = self._get_function_definitions(bedrock_agent_tool)

                args = bedrock_agent_tool.args
                function_parameters = {}
                for arg_name, arg_detail in args.items():
                    function_parameters[arg_name] = \
                        {
                            'description': function_args_descriptions.get(arg_name, arg_name),
                            'required': True,
                            'type': 'string' if not type(arg_detail) is dict else arg_detail.get('type', 'string')
                        }

                action_group_detail = functions.get(action_group_name, {})
                if action_group_detail:
                    action_group_detail.get('functionSchema', {}).get('functions', []) \
                        .append(
                        self._get_function_definition_json(
                            function_description=function_description if function_description else function_name,
                            function_name=function_name,
                            function_parameters=function_parameters
                        )
                    )
                else:
                    functions[action_group_name] = {
                        'actionGroupName': action_group_name,
                        'agentId': bedrock_agent_runtime_construct.agent_id,
                        'agentVersion': 'DRAFT',
                        'actionGroupExecutor': {'customControl': 'RETURN_CONTROL'},
                        'actionGroupState': 'ENABLED',
                        'functionSchema': {
                            'functions': [
                                self._get_function_definition_json(
                                    function_description=function_description,
                                    function_name=function_name,
                                    function_parameters=function_parameters
                                )
                            ]
                        }
                    }

            return functions
        except Exception as exception:
            logging.error(f'Error in create agent function: {exception}')
            raise exception

    def _get_function_definition_json(
            self,
            function_description: str,
            function_name: str,
            function_parameters: dict
    ) -> dict:
        """
        Helper to create simplified function definition

        Args:
            function_description: Text describing the function
            function_name: Name of the function
            function_parameters: Parameters which are required for the function
        Returns:
           definition of an action inside an ActionGroup
        """
        return {
            'description': function_description,
            'name': function_name,
            'parameters': function_parameters
        }

    def _get_function_definitions(
            self,
            bedrock_agent_tool: Tool
    ) -> Tuple[str, str, Union[str, Any], dict]:

        """
        Translate Tool to ActionGroups and Actions compatible to Bedrock Agents

        Args:
            bedrock_agent_tool: Langchain tool
        Returns:
           Returns a tuple with:
             action_group_name - Name of the action group
             function_name - Name of the action (function)
             function_description - Action(function) description
             function_args_descriptions - Action(Function) parameters
        """
        tool_name = bedrock_agent_tool.name
        action_group_name = _DEFAULT_ACTION_GROUP_NAME
        function_name = tool_name

        # In BedrockAgents functions the naming format for Tools is <ActionGroupName>::<ActionName>
        tool_name_split = tool_name.split("::")
        if len(tool_name_split) > 1:
            action_group_name = tool_name_split[0]
            function_name = tool_name_split[1]

        function_description = bedrock_agent_tool.description
        args = bedrock_agent_tool.args
        function_args_descriptions = {}
        description = re.findall(r"<agent_tool_doc>(.*?)</agent_tool_doc>", bedrock_agent_tool.description, re.DOTALL)
        if not description:
            description = [bedrock_agent_tool.description]
        if description:
            function_description_received = re.findall(r"(.*?)Args:", description[0], re.DOTALL)
            args_description_received = re.findall(r"Args:(.*?)(?=Returns|$)", description[0], re.DOTALL)
            if function_description_received:
                function_description = function_description_received[0].strip()
            if args_description_received:
                for args_description in args_description_received:
                    args_docs = re.findall(r"(.+):(.+)", args_description)
                    for tup in args_docs:
                        arg_name, args_description = tup
                        if arg_name.strip() in args:
                            function_args_descriptions[arg_name.strip()] = args_description.strip()
        return action_group_name, function_name, function_description, function_args_descriptions

    def _wait_on_agent_prepare(
            self,
            agent_id: str
    ):
        """
        Wait for the Agent to get prepared. Monitor for the creation and wait until the Agent status becomes 'PREPARED'
        Timeout is 10s

        Args:
            agent_id: AWS Bedrock AgentId

        """
        prepare_agent_start_time = time.time()
        agent_status = self.bedrock_buildtime.get_agent(agentId=agent_id) \
            .get('agent', {}).get('agentStatus', '')
        while agent_status != 'PREPARED' and (time.time() - prepare_agent_start_time) < 10:
            time.sleep(2)
            agent_status = self.bedrock_buildtime.get_agent(agentId=agent_id) \
                .get('agent', {}).get('agentStatus', '')

        if agent_status != 'PREPARED':
            logging.error(f'Failed to prepare Bedrock agent {agent_id}')
            raise


class BedrockAgentBase:
    """
    BedrockAgentBase to encapsulate the lifecycle processes for -
    1. Agent creation and deletion
    2. Action creation and deletion
    3. Agent Role creation and deletion
    """
    __SINGLE_SPACE = ' '
    __TRACE_PARTS_TO_EMIT = [
        # 'modelInvocationInput', # This trace contains prompts used by the BedrockAgents
        'rationale',  # Trace with the model thoughts
        'invocationInput',  # This trace is emitted when BedrockAgents makes an tool invocation
    ]

    name: str = 'BedrockAgentBase'
    agent_id: str = __SINGLE_SPACE
    agent_alias_id: str = __SINGLE_SPACE
    agent_region: str = __SINGLE_SPACE
    agent_resource_role_arn: str = None
    bedrock_runtime: Any = None
    agent_tools: List[Tool] = []
    output_parser: BaseOutputParser = None
    prompt_template: BasePromptTemplate = None
    trace_handler: Callable = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def agent(
            self,
            bedrock_agent_runtime_construct: BedrockAgentRuntimeConstruct,
            output_parser: BaseOutputParser = None,
            prompt_template: BasePromptTemplate = None,
            trace_handler: Callable = None
    ):
        """
        Use existing agents for inference
        """

        self.agent_id = bedrock_agent_runtime_construct.agent_id
        self.agent_alias_id = bedrock_agent_runtime_construct.agent_alias_id
        self.agent_region = bedrock_agent_runtime_construct.agent_region
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.agent_tools = bedrock_agent_runtime_construct.agent_tools
        self.trace_handler = trace_handler
        self.bedrock_runtime = bedrock_agent_runtime()

    def create(
            self,
            bedrock_agent_metadata: BedrockAgentMetadata,
            output_parser: BaseOutputParser = None,
            prompt_template: BasePromptTemplate = None,
            trace_handler: Callable = None
    ):
        """
        Creates a new agent
        """

        bedrock_agent_manager = BedrockAgentManager()
        self.agent_region = bedrock_agent_metadata.agent_region
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.trace_handler = trace_handler

        new_role_created = False
        if not bedrock_agent_metadata.agent_resource_role_arn:
            agent_resource_role_arn = bedrock_agent_manager.create_agent_role(
                bedrock_agent_metadata.agent_region, bedrock_agent_metadata.agent_foundation_model
            ),
            new_role_created = True
            bedrock_agent_metadata.agent_resource_role_arn = agent_resource_role_arn[0]
            self.agent_resource_role_arn = agent_resource_role_arn[0]

        self.agent_id, self.agent_alias_id \
            = self._prepare(bedrock_agent_metadata=bedrock_agent_metadata)

        if new_role_created:
            bedrock_agent_manager.update_agent_resource_policy(
                bedrock_agent_metadata.agent_resource_role_arn, bedrock_agent_metadata.agent_region, self.agent_id,
            )

        self.agent_tools = bedrock_agent_metadata.agent_tools
        self.bedrock_runtime = bedrock_agent_runtime()

    def delete_agent(
            self,
    ):
        """
        Delete an agent
        """

        bedrock_agent_manager = BedrockAgentManager()
        bedrock_agent_manager.delete_agent(self.agent_id, self.agent_resource_role_arn)

    def _prepare(
            self,
            bedrock_agent_metadata: BedrockAgentMetadata
    ):
        """
        On Bedrock Agent creation, we prepare the agent to make it ready for inference

        Args:
            bedrock_agent_metadata: Meta data for Bedrock Agents
        """

        bedrock_agent_manager = BedrockAgentManager()
        bedrock_agent_create = bedrock_agent_manager.create_agent(bedrock_agent_metadata)
        if bedrock_agent_create and bedrock_agent_create.agent_id:
            bedrock_agent_manager.create_function(bedrock_agent_create)
            bedrock_agent_manager.prepare_agent(bedrock_agent_create.agent_id)
        return bedrock_agent_create.agent_id, bedrock_agent_create.agent_alias_id

    @property
    def input_keys(self) -> List[str]:
        """
        Return the input keys.
        """
        return []

    @property
    def _agent_type(self) -> str:
        """
        Return Identifier of agent type.
        """
        raise NotImplementedError

    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Callable implementation for langchain plan()

        Args:
            intermediate_steps: Langchain intermediate steps during execution
            callbacks: Callbacks like custom parsers, formatter, tools.
                 custom parsers - Used to parse the BedrockAgent response for custom handling
                 custom formatter - Used to format the BedrockAgent request for specific usecases
                 tools - Langchain tools used in conjunction to Agents
        Returns:
            Either the iteration of a Bedrock Agent can be AgentFinish or AgentAction
            AgentFinish - Final answer returned by BedrockAgents
            AgentAction - When a tool/API is returned by BedrockAgents
        """

        if self.prompt_template:
            kwargs['intermediate_steps'] = intermediate_steps
            kwargs.__setitem__('input', self.prompt_template.format(**kwargs))

        agent_response = self.invoke_agent_base(**kwargs)

        if self.output_parser:
            agent_response = self.output_parser.parse(agent_response)

        if type(agent_response) is AgentAction or type(agent_response) is AgentFinish:
            return agent_response

        return AgentFinish(
            {
                "output": agent_response,
            },
            agent_response
        )

    async def aplan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Callable implementation for langchain async plan()

        Args:
            intermediate_steps: Langchain intermediate steps during execution
            callbacks: Callbacks like custom parsers, formatter, tools.
                 custom parsers - Used to parse the BedrockAgent response for custom handling
                 custom formatter - Used to format the BedrockAgent request for specific usecases
                 tools - Langchain tools used in conjunction to Agents
        Returns:
            Either the iteration of a Bedrock Agent can be AgentFinish or AgentAction
            AgentFinish - Final answer returned by BedrockAgents
            AgentAction - When a tool/API is returned by BedrockAgents
        """

        agent_response = self.invoke_agent_base(**kwargs)
        if self.output_parser:
            agent_response = self.output_parser.parse(agent_response)

        return AgentFinish(
            {
                "output": agent_response,
            },
            agent_response
        )

    def invoke_agent_base(
            self,
            **kwargs: Any,
    ) -> str:
        """
        Callable implementation for langchain invoke()
        """

        agent_input = kwargs.get('input', '')
        session_state = kwargs.get('session_state', {})

        if type(agent_input) is dict:
            input_session_state = agent_input.get('session_state', {})
            if input_session_state:
                session_state = input_session_state
            agent_input = agent_input.get('input', '')

        session_id = kwargs.get('session_id', str(uuid.uuid4()))

        return self._invoke_bedrock_agent(
            agent_input=agent_input,
            session_state=session_state,
            session_id=session_id
        )

    def _invoke_bedrock_agent(
            self,
            agent_input: str,
            session_state: dict,
            session_id: str = str(uuid.uuid4()),
            trace_enabled: bool = True
    ) -> str:
        """
        Invoke Bedrock Agent and return the output.
        Send trace event to callback handler.

        Args:
            agent_input: Input text to BedrockAgents
            session_state: Input session state to BedrockAgents
            session_id: Session Ids are unique values, identifying a list of turns pertaining to a single conversation
            trace_enabled: Enabling the flag will emit a trace on each step in BedrockAgents.

        Returns:
            The response from BedrockAgents
        """

        try:
            invoke_agent_response = self.bedrock_runtime.invoke_agent(
                inputText=agent_input,
                sessionState=session_state,
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                enableTrace=trace_enabled,
            )

            response_text = ''
            request_id = invoke_agent_response['ResponseMetadata']['RequestId']
            logging.info(f'Bedrock agent call request id: {request_id}')
            event_stream = invoke_agent_response['completion']

            logging.info("\n********************** Bedrock Agents Input **********************\n")
            logging.info(f"Input: {agent_input} \n")
            logging.info(f"SessionState: {json.dumps(session_state, indent=4)} \n")

            # Iterate over events in the event stream as they come
            for event in event_stream:
                # handle trace event
                self._handle_trace(event)

                if 'returnControl' in event:
                    response_text = json.dumps(event)
                    break

                if 'chunk' in event:
                    response_text = event['chunk']['bytes'].decode("utf-8")

            logging.info("\n********************** Bedrock Agents Output *********************\n")
            logging.info(f"{response_text} \n")

            return response_text
        except Exception as exception:
            logging.error(f'Error in invoke agent call: {exception}')

    def _handle_trace(
            self,
            event: dict
    ):
        """
        Filter and invoke callback for trace handling on each event

        Args:
            event: Trace event emitted by BedrockAgents
        """

        trace_value = self._trace_filter(event)
        if self.trace_handler and trace_value:
            self.trace_handler(trace_value)

    def _trace_filter(
            self,
            event: dict
    ) -> Optional[Dict[Any, Optional[Any]]]:
        """
        Filter Bedrock Agents trace events

        Args:
            event: Trace event emitted by BedrockAgents

        Returns:
            The orchestration trace texts which may contain any of the __TRACE_PARTS_TO_EMIT
        """

        trace_object = event.get('trace', {})
        trace = trace_object.get('trace', {})
        orchestration_trace = trace.get('orchestrationTrace', {})
        for trace_part_to_emit in self.__TRACE_PARTS_TO_EMIT:
            trace_part = orchestration_trace.get(trace_part_to_emit, None)
            if trace_part:
                return {
                    trace_part_to_emit: trace_part
                }
        return None
