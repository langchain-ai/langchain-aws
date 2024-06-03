import json
from asyncio import run
from typing import Callable
from unittest import mock

from langchain.agents import AgentOutputParser
from langchain_community.agents_bedrock.agent_base import (
    agent_tool,
    BedrockAgentMetadata,
    BedrockAgentRuntimeConstruct,
    BedrockAgentManager,
    BedrockAgentBase
)
from langchain_community.agents_bedrock.agent_client import (
    bedrock_agent_runtime
)
from langchain_core.agents import AgentAction
from langchain_core.prompts.base import BaseOutputParser, BasePromptTemplate


@agent_tool(
        action_group='testActionGroup',
        action='testAction',
        method='testMethod')
def getTestFunction1(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action group, action and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(action_group='testActionGroup', action='testAction')
def getTestFunction2(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action group and action

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(action='testAction', method='testMethod')
def getTestFunction3(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(action_group='testActionGroup', method='testMethod')
def getTestFunction4(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action group and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(action_group='testActionGroup')
def getTestFunction5(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action group only

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(method='testMethod')
def getTestFunction6(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with a method only

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool(action='testAction')
def getTestFunction7(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action only

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@agent_tool()
def getTestFunction8(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with no arguments

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


def test_construct_bedrock_agent_metadata():
    """
    Test BedrockAgentMetadata construct with valid inputs
    """
    agent_construct = BedrockAgentMetadata(
        agent_name='testAgent',
        agent_instruction='You are a testing agent',
        agent_description='An agent created for testing purposes',
        agent_tools=[
            getTestFunction1,
            getTestFunction2,
            getTestFunction3,
            getTestFunction4],
        agent_resource_role_arn='testAgentRoleArn',
        data_privacy=None,
        idle_session_ttl_seconds=600,
        agent_foundation_model='anthropic.claude-3-haiku-20240307',
        agent_region='us-west-2'
    )
    assert agent_construct is not None
    assert agent_construct.agent_name == 'testAgent'
    assert agent_construct.agent_instruction == 'You are a testing agent'
    assert agent_construct.agent_description == 'An agent created for testing purposes'
    assert agent_construct.agent_resource_role_arn == 'testAgentRoleArn'
    assert agent_construct.idle_session_ttl_seconds == (600,)
    assert agent_construct.agent_foundation_model == 'anthropic.claude-3-haiku-20240307'
    assert agent_construct.agent_region == 'us-west-2'

    # Validate agent tools
    assert agent_construct.agent_tools[0].name == 'testMethod::testActionGroup::testAction'
    assert agent_construct.agent_tools[0].description == """testMethod::testActionGroup::testAction(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action group, action and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_construct.agent_tools[1].name == 'testActionGroup::testAction'
    assert agent_construct.agent_tools[1].description == """testActionGroup::testAction(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action group and action

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_construct.agent_tools[2].name == 'testActionActionGroup::testAction'
    assert agent_construct.agent_tools[2].description == """testActionActionGroup::testAction(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_construct.agent_tools[3].name == 'getTestFunction4ActionGroup::getTestFunction4'
    assert agent_construct.agent_tools[3].description == """getTestFunction4ActionGroup::getTestFunction4(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action group and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""


def test_construct_bedrock_agent_runtime():
    """
    Test BedrockAgentRuntimeConstruct construct with valid inputs
    """
    agent_runtime_construct = BedrockAgentRuntimeConstruct(
        agent_id='ABC1234',
        agent_alias_id='TSTALIASID',
        agent_region='us-west-2',
        agent_tools=[
            getTestFunction5,
            getTestFunction6,
            getTestFunction7,
            getTestFunction8],
    )
    assert agent_runtime_construct is not None
    assert agent_runtime_construct.agent_id == 'ABC1234'
    assert agent_runtime_construct.agent_alias_id == 'TSTALIASID'
    assert agent_runtime_construct.agent_region == 'us-west-2'

    # Validate agent tools
    assert agent_runtime_construct.agent_tools[0].name == 'getTestFunction5ActionGroup::getTestFunction5'
    assert agent_runtime_construct.agent_tools[0].description == """getTestFunction5ActionGroup::getTestFunction5(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action group only

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_runtime_construct.agent_tools[1].name == 'getTestFunction6ActionGroup::getTestFunction6'
    assert agent_runtime_construct.agent_tools[1].description == """getTestFunction6ActionGroup::getTestFunction6(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with a method only

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_runtime_construct.agent_tools[2].name == 'testActionActionGroup::testAction'
    assert agent_runtime_construct.agent_tools[2].description == """testActionActionGroup::testAction(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with an action only

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""

    assert agent_runtime_construct.agent_tools[3].name == 'getTestFunction8ActionGroup::getTestFunction8'
    assert agent_runtime_construct.agent_tools[3].description == """getTestFunction8ActionGroup::getTestFunction8(parameter: str = ' ') -> str - <agent_tool_doc>
    Test function for agent tools with no arguments

    Args:
        parameter: argument of the test function
    Returns:
        str
    </agent_tool_doc>"""


@mock.patch("boto3.client")
def test_create_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager create with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }

    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None
    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_create_agent_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager create_agent method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.return_value = {
        "agent":
        {
            "agentStatus": "NOT_PREPARED"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    # Create Bedrock agent metadata
    agent_construct = BedrockAgentMetadata(
        agent_name='testAgent',
        agent_instruction='You are a testing agent',
        agent_description='An agent created for testing purposes',
        agent_tools=[
            getTestFunction1,
            getTestFunction2,
            getTestFunction3,
            getTestFunction4],
        agent_resource_role_arn='testAgentRoleArn',
        data_privacy=None,
        idle_session_ttl_seconds=600,
        agent_foundation_model='anthropic.claude-3-haiku-20240307',
        agent_region='us-west-2'
    )

    createAgentResponse = agent_manager.create_agent(agent_construct)

    assert createAgentResponse is not None
    assert createAgentResponse.agent_id == 'test_agent_id'
    assert createAgentResponse.agent_alias_id == 'TSTALIASID'
    assert createAgentResponse.agent_region == 'us-west-2'
    assert createAgentResponse.agent_tools == [
        getTestFunction1,
        getTestFunction2,
        getTestFunction3,
        getTestFunction4]

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_create_agent_from_bedrock_agent_manager_with_failure(
    mock_client
):
    """
    Test BedrockAgentManager create_agent method with invalid inputs, specifically by
    returning FAILURE for the GetAgent call. Then, verify the failure at the end.
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.return_value = {
        "agent":
        {
            "agentStatus": "FAILED"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    # Create Bedrock agent metadata
    agent_construct = BedrockAgentMetadata(
        agent_name='testAgent',
        agent_instruction='You are a testing agent',
        agent_description='An agent created for testing purposes',
        agent_tools=[
            getTestFunction1,
            getTestFunction2,
            getTestFunction3,
            getTestFunction4],
        agent_resource_role_arn='testAgentRoleArn',
        data_privacy=None,
        idle_session_ttl_seconds=600,
        agent_foundation_model='anthropic.claude-3-haiku-20240307',
        agent_region='us-west-2'
    )

    try:
        agent_manager.create_agent(agent_construct)
    except RuntimeError:
        assert True

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_prepare_agent_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager prepare_agent method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.return_value = {
        "agent":
        {
            "agentStatus": "PREPARED"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    agent_manager.prepare_agent('test_agent_id')

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_prepare_agent_from_bedrock_agent_manager_with_failure(
    mock_client
):
    """
    Test BedrockAgentManager prepare_agent method with invalid inputs,
    specifically by returning NOT_PREPARED for the GetAgent call. Then, verify
    the failure at the end.
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.return_value = {
        "agent":
        {
            "agentStatus": "NOT_PREPARED"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    try:
        agent_manager.prepare_agent('test_agent_id')
    except RuntimeError:
        assert True

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_create_function_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager create_function method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_create_agent_action_group = mock_client.return_value.create_agent_action_group
    mock_create_agent_action_group.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    # Create Bedrock agent runtime metadata
    agent_runtime_construct = BedrockAgentRuntimeConstruct(
        agent_id='ABC1234',
        agent_alias_id='TSTALIASID',
        agent_region='us-west-2',
        agent_tools=[getTestFunction2]
    )

    createFunctionResponse = agent_manager.create_function(
        agent_runtime_construct
    )

    assert createFunctionResponse is not None
    assert createFunctionResponse.agent_id == 'ABC1234'
    assert createFunctionResponse.agent_alias_id == 'TSTALIASID'
    assert createFunctionResponse.agent_region == 'us-west-2'
    assert createFunctionResponse.agent_tools == [getTestFunction2]

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_delete_agent_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager delete_agent method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }

    mock_delete = mock_client.return_value.delete_agent
    mock_delete.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agentStatus": "DELETING"
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    agent_manager.delete_agent(
        agent_id='ABC1234',
        agent_resource_role_arn='testRoleArn/RoleName'
    )

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_delete_agent_from_bedrock_agent_manager_with_failure(
    mock_client
):
    """
    Test BedrockAgentManager delete_agent method with invalid inputs,
    specifically by returning PREPARED for the DeleteAgent call. Then, verify
    the failure at the end.
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_delete = mock_client.return_value.delete_agent
    mock_delete.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agentStatus": "PREPARED"
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    try:
        agent_manager.delete_agent(
            agent_id='ABC1234',
            agent_resource_role_arn='testRoleArn/RoleName'
        )
    except RuntimeError:
        assert True

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_create_agent_role_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager create_agent_role method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }
    mock_create_role = mock_client.return_value.create_role
    mock_create_role.return_value = {
        "Role":
        {
            "Arn": "testRoleArn",
            "RoleName": "testRoleName"
        }
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    createAgentRoleResponse = agent_manager.create_agent_role(
        agent_region='us-west-2',
        foundational_model='anthropic.claude-3-haiku-20240307'
    )

    assert createAgentRoleResponse is not None
    assert createAgentRoleResponse == 'testRoleArn'

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


@mock.patch("boto3.client")
def test_delete_agent_role_from_bedrock_agent_manager(
    mock_client
):
    """
    Test BedrockAgentManager delete_agent_role method with valid inputs
    """
    mock_get_caller_identity = mock_client.return_value.get_caller_identity
    mock_get_caller_identity.return_value = {
        "Account": "123456789012"
    }

    # Create Bedrock agent manager
    agent_manager = BedrockAgentManager()
    assert agent_manager is not None
    assert agent_manager.account_id == '123456789012'
    assert agent_manager.name == 'BedrockAgentManager'
    assert agent_manager.bedrock_buildtime is not None
    assert agent_manager.bedrock_agent_permission is not None
    assert agent_manager.bedrock_agent_sts is not None

    agent_manager.delete_agent_role(
        agent_resource_role_arn='testRoleArn/RoleName'
    )

    agent_manager.bedrock_buildtime.close()
    agent_manager.bedrock_agent_permission.close()
    agent_manager.bedrock_agent_sts.close()


def test_create_bedrock_agent_base():
    """
    Test BedrockAgentBase class with valid inputs
    """
    # Create Bedrock agent base
    agent_base = BedrockAgentBase()

    assert agent_base is not None
    assert agent_base.name == 'BedrockAgent'
    assert agent_base.agent_id == ' '
    assert agent_base.agent_alias_id == ' '
    assert agent_base.agent_region == ' '
    assert agent_base.agent_resource_role_arn is None
    assert agent_base.bedrock_runtime is None
    assert agent_base.agent_tools == []
    assert agent_base.output_parser is None
    assert agent_base.prompt_template is None
    assert agent_base.trace_handler is None


def test_agent_from_bedrock_agent_base():
    """
    Test creating an agent from BedrockAgentBase class with valid inputs
    """
    # Create Bedrock agent base
    agent_base = BedrockAgentBase()

    # Set input classes
    agent_runtime_construct = BedrockAgentRuntimeConstruct(
        agent_id='ABC1234',
        agent_alias_id='TSTALIASID',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    class TestAgentOutputParser(AgentOutputParser):
        def parse(self, output):
            return output
    agent_output_parser = TestAgentOutputParser(
        name='agentOutputParser'
    )

    class TestBaseOutputParser(BaseOutputParser):
        def parse(self, output):
            return output
    agent_base_output_parser = TestBaseOutputParser(
        name='baseOutputParser'
    )

    class TestBasePromptTemplate(BasePromptTemplate):
        def format(self, input):
            return input

        def format_prompt(self, input):
            return input

    base_prompt_template = TestBasePromptTemplate(
        name='basePromptTemplate',
        input_variables=['testInputVariable'],
        input_types={
            'testInputType': 'string'
        },
        output_parser=agent_base_output_parser,
        partial_variables={
            'testPartialVariable': 'testPartialVariable'
        },
        metadata={
            'testMetadata': 'testMetadata'
        },
        tags=['testTag']
    )

    class TestCallable(Callable):
        def __call__(self, input):
            return input
    trace_handler = TestCallable()

    # Call agent method
    agent_base.agent(
        bedrock_agent_runtime_construct=agent_runtime_construct,
        output_parser=agent_output_parser,
        prompt_template=base_prompt_template,
        trace_handler=trace_handler
    )

    assert agent_base is not None
    assert agent_base.name == 'BedrockAgent'
    assert agent_base.agent_id == 'ABC1234'
    assert agent_base.agent_alias_id == 'TSTALIASID'
    assert agent_base.agent_region == 'us-west-2'
    assert agent_base.agent_resource_role_arn is None
    assert agent_base.bedrock_runtime is not None
    assert agent_base.agent_tools == [getTestFunction1]
    assert agent_base.output_parser is agent_output_parser
    assert agent_base.prompt_template is base_prompt_template
    assert agent_base.trace_handler is trace_handler

    agent_base.bedrock_runtime.close()


@mock.patch("boto3.client")
def test_create_agent_from_bedrock_agent_base(
    mock_client
):
    """
    Test creating an agent from BedrockAgentBase class with valid inputs
    """
    # Mock methods
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]

    # Create Bedrock agent base
    agent_base = BedrockAgentBase()

    # Set input classes
    agent_construct = BedrockAgentMetadata(
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_description='An agent created for testing purposes',
        agent_tools=[getTestFunction1],
        agent_resource_role_arn='testAgentRoleArn',
        data_privacy=None,
        idle_session_ttl_seconds=600,
        agent_foundation_model='anthropic.claude-3-haiku-20240307',
        agent_region='us-west-2'
    )

    class TestAgentOutputParser(AgentOutputParser):
        def parse(self, output):
            return output
    agent_output_parser = TestAgentOutputParser(
        name='agentOutputParser'
    )

    class TestBaseOutputParser(BaseOutputParser):
        def parse(self, output):
            return output
    agent_base_output_parser = TestBaseOutputParser(
        name='baseOutputParser'
    )

    class TestBasePromptTemplate(BasePromptTemplate):
        def format(self, input):
            return input

        def format_prompt(self, input):
            return input

    base_prompt_template = TestBasePromptTemplate(
        name='basePromptTemplate',
        input_variables=['testInputVariable'],
        input_types={
            'testInputType': 'string'
        },
        output_parser=agent_base_output_parser,
        partial_variables={
            'testPartialVariable': 'testPartialVariable'
        },
        metadata={
            'testMetadata': 'testMetadata'
        },
        tags=['testTag']
    )

    class TestCallable(Callable):
        def __call__(self, input):
            return input
    trace_handler = TestCallable()

    # Call create method
    agent_base.create(
        bedrock_agent_metadata=agent_construct,
        output_parser=agent_output_parser,
        prompt_template=base_prompt_template,
        trace_handler=trace_handler
    )

    assert agent_base is not None
    assert agent_base.name == 'testAgent'
    assert agent_base.agent_id == 'test_agent_id'
    assert agent_base.agent_alias_id == 'TSTALIASID'
    assert agent_base.agent_region == 'us-west-2'
    assert agent_base.agent_resource_role_arn is None
    assert agent_base.bedrock_runtime is not None
    assert agent_base.agent_tools == [getTestFunction1]
    assert agent_base.output_parser is agent_output_parser
    assert agent_base.prompt_template is base_prompt_template
    assert agent_base.trace_handler is trace_handler

    agent_base.bedrock_runtime.close()


@mock.patch("boto3.client")
def test_delete_agent_from_bedrock_agent_base(
    mock_client
):
    """
    Test deleting an agent from BedrockAgentBase class with valid inputs
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]
    mock_delete = mock_client.return_value.delete_agent
    mock_delete.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agentStatus": "DELETING"
    }

    # Create Bedrock agent base
    agent_base = BedrockAgentBase()

    agent_construct = BedrockAgentMetadata(
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_description='An agent created for testing purposes',
        agent_tools=[getTestFunction1],
        data_privacy=None,
        idle_session_ttl_seconds=600,
        agent_foundation_model='anthropic.claude-3-haiku-20240307',
        agent_region='us-west-2'
    )

    agent_base.create(
        bedrock_agent_metadata=agent_construct
    )

    # Call delete method
    agent_base.delete()


def test_bedrock_agent_base_input_keys_method():
    """
    Test BedrockAgentBase class input_keys method
    """
    # Create Bedrock agent base
    agent_base = BedrockAgentBase()

    # Call input_keys method
    agent_base.input_keys.append('testInputKey')


@mock.patch("boto3.client")
def test_plan_from_bedrock_agent_base(
    mock_client
):
    """
    Test calling the plan method from BedrockAgentBase class with valid inputs
    and validate the response.
    """
    mock_invoke = mock_client.return_value.invoke_agent
    mock_invoke.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "completion":
        [
            {
                "chunk":
                {
                    "bytes": bytes(b'Hello from Bedrock')
                },
                "trace":
                {
                    'agentAliasId': 'string',
                    'agentId': 'string',
                    'agentVersion': 'string',
                    'sessionId': 'string',
                    'trace': {
                        "orchestrationTrace": {
                            "observation": {
                                "finalResponse": {
                                    "text": "Final response text"
                                }
                            },
                            "traceId": "efc43f51-28bd-43c0-b5d2-96c64b085af6-1",
                            "type": "FINISH"
                        }
                    }
                }
            }
        ]
    }

    agent_base = BedrockAgentBase()
    agent_base.bedrock_runtime = bedrock_agent_runtime()

    # Call the plan method
    agentBaseResponse = agent_base.plan(
        intermediate_steps=[
            (
                AgentAction(
                    tool='testAction',
                    tool_input='testInput',
                    log='testlog'
                ),
                'testAgentActionStep'
            )
        ],
        callbacks=None,
        input='Hello',
        session_id='testSessionId'
    )

    assert agentBaseResponse.return_values == {
        'output': 'Hello from Bedrock'
    }
    assert agentBaseResponse.log == 'Hello from Bedrock'


@mock.patch("boto3.client")
def test_aplan_from_bedrock_agent_base(
    mock_client
):
    """
    Test calling the aplan method from BedrockAgentBase class with valid inputs
    and validate the response.
    """
    mock_invoke = mock_client.return_value.invoke_agent
    mock_invoke.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "completion":
        [
            {
                "chunk":
                {
                    "bytes": bytes(b'Hello from Bedrock')
                },
                "trace":
                {
                    'agentAliasId': 'string',
                    'agentId': 'string',
                    'agentVersion': 'string',
                    'sessionId': 'string',
                    'trace': {
                        "orchestrationTrace": {
                            "observation": {
                                "finalResponse": {
                                    "text": "Final response text"
                                }
                            },
                            "traceId": "efc43f51-28bd-43c0-b5d2-96c64b085af6-1",
                            "type": "FINISH"
                        }
                    }
                }
            }
        ]
    }

    agent_base = BedrockAgentBase()
    agent_base.bedrock_runtime = bedrock_agent_runtime()

    # Call the plan method
    aplanResponse = run(agent_base.aplan(
        intermediate_steps=[
            (
                AgentAction(
                    tool='testAction',
                    tool_input='testInput',
                    log='testlog'
                ),
                'testAgentActionStep'
            )
        ],
        callbacks=None,
        input='Hello',
        session_id='testSessionId'
    ))

    assert aplanResponse.return_values == {
        'output': 'Hello from Bedrock'
    }
    assert aplanResponse.log == 'Hello from Bedrock'


@mock.patch("boto3.client")
def test_invoke_from_bedrock_agent_base(
    mock_client
):
    """
    Test calling the invoke method from BedrockAgentBase class with valid inputs
    and validate the response.
    """
    mock_invoke = mock_client.return_value.invoke_agent
    mock_invoke.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "completion":
        [
            {
                "returnControl": {
                    "invocationInputs": [{
                        "functionInvocationInput": {
                            "actionGroup": "WeatherAPIs",
                            "function": "getWeather",
                            "parameters": [
                                {
                                    "name": "location",
                                    "type": "string",
                                    "value": "seattle"
                                },
                                {
                                    "name": "date",
                                    "type": "string",
                                    "value": "2024-09-15"
                                }
                            ]
                        }
                    }],
                    "invocationId": "79e0feaa-c6f7-49bf-814d-b7c498505172"
                }
            }
        ]
    }

    agent_base = BedrockAgentBase()
    agent_base.bedrock_runtime = bedrock_agent_runtime()

    # Call the plan method
    agentBaseResponse = agent_base.invoke(
        input='Let\'s test the return of control function',
        session_id='testSessionId'
    )

    expectedReturnofControl = {
        "invocationInputs": [{
            "functionInvocationInput": {
                "actionGroup": "WeatherAPIs",
                "function": "getWeather",
                "parameters": [
                    {
                        "name": "location",
                        "type": "string",
                        "value": "seattle"
                    },
                    {
                        "name": "date",
                        "type": "string",
                        "value": "2024-09-15"
                    }
                ]
            }
        }],
        "invocationId": "79e0feaa-c6f7-49bf-814d-b7c498505172"
    }

    assert json.loads(agentBaseResponse)['returnControl'] == expectedReturnofControl
