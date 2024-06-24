# flake8: noqa
from unittest.mock import MagicMock, patch

import pytest

from langchain_aws.evaluators.action_registry import ActionInterface
from langchain_aws.evaluators.inference_evaluator import (
    RunnableSageMakerInferenceEvaluator,
)
from langchain_aws.evaluators.inference_evaluator_exception import (
    InferenceEvaluatorError,
)

RUNNABLE_INFERENCE_EVALUATOR_PATH = (
    "langchain_aws.evaluators.inference_evaluator.RunnableSageMakerInferenceEvaluator"
)

TEST_INFERENCE_EVALUATOR_CONFIG = {
    "name": "example_config_1",
    "policies": [
        {
            "evaluation_algorithm": {
                "type": "toxicity",
                "evaluation_algorithm_config": {"model_type": "toxigen"},
            },
            "actions": [
                {
                    "type": "Deny",
                    "criteria": {"category": "toxicity", "operator": ">", "value": 0.4},
                    "configuration": {
                        "alternative_response": "Please provide a safe prompt."
                        + "The toxicity score is above 0.5.",
                    },
                },
            ],
        },
        {
            "evaluation_algorithm": {
                "type": "toxicity",
                "evaluation_algorithm_config": {"model_type": "detoxify"},
            },
            "actions": [
                {
                    "type": "Redact",
                    "criteria": {"category": "toxicity", "operator": "<", "value": 0.4},
                    "configuration": {
                        "action_name": "AppendMessage",
                        "additional_parameters": {"message": "RAG_CONTEXT=API"},
                    },
                },
            ],
        },
        {
            "evaluation_algorithm": {
                "type": "rule",
                "evaluation_algorithm_config": {
                    "model_type": "regex",
                    "pattern": ".*Instruct.*",
                },
            },
            "actions": [
                {
                    "type": "Instruct",
                    "criteria": {"category": "RULE", "operator": "=", "value": 1},
                    "configuration": {
                        "action_name": "EventBridge",
                        "additional_parameters": {
                            "api_invokes": {
                                "create_event_bus": {"Name": "TestEventBridgeName"},
                                "put_events": {
                                    "Entries": [
                                        {
                                            "Source": "com.mycompany.myapp",
                                            "Detail": '{ "key1": "value1", '
                                            + '"key2": "value2" }',
                                            "Resources": ["resource1", "resource2"],
                                            "DetailType": "myDetailType",
                                        },
                                    ],
                                },
                            }
                        },
                    },
                }
            ],
        },
    ],
}


def inference_evaluator_method_path(method_name: str) -> str:
    return "{0}.{1}".format(RUNNABLE_INFERENCE_EVALUATOR_PATH, method_name)


@pytest.fixture
def runnable_sagemaker_inference_evaluator_fixture() -> (
    RunnableSageMakerInferenceEvaluator
):
    # Mock boto3 import
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.client.return_value = (
        MagicMock()
    )  # Mocking the client method of the Session object

    with patch.dict(
        "sys.modules", {"boto3": mock_boto3}
    ):  # Mocking boto3 at the top level using patch.dict
        return RunnableSageMakerInferenceEvaluator(
            inference_evaluator_name="test-inference-evaluator",
            inference_evaluators_configuration=TEST_INFERENCE_EVALUATOR_CONFIG,
        )


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_fixture(
    runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    runnable_sagemaker_inference_evaluator_fixture._endpoint_exists = MagicMock(  # type: ignore
        return_value=True
    )
    runnable_sagemaker_inference_evaluator_fixture._inference_component_exists = (  # type: ignore
        MagicMock(return_value=True)
    )

    runnable_sagemaker_inference_evaluator_fixture.deploy(
        endpoint_name="test-endpoint",
        inference_component_name="test-inference-component",
        execution_role="arn:aws:iam::<account-number>:role/test-exec-role",
    )

    return runnable_sagemaker_inference_evaluator_fixture


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_with_deny_action(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    inference_evaluator_output = {
        "name": "test-inference-evaluation",
        "actions": [
            {
                "type": "Deny",
                "configuration": {
                    "alternative_response": "Please provide a safe prompt."
                    + "The toxicity score is above 0.5."
                },
            }
        ],
        "evaluation_scores": [
            {
                "evaluation_algorithm_type": "toxicity",
                "model_type": "toxigen",
                "scores": [{"name": "toxicity", "value": 0.9}],
            }
        ],
    }

    deployed_runnable_sagemaker_inference_evaluator_fixture._call_inference_evaluator = MagicMock(  # type: ignore
        return_value=inference_evaluator_output
    )

    return deployed_runnable_sagemaker_inference_evaluator_fixture


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_with_redact_action(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    inference_evaluator_output = {
        "name": "test-inference-evaluation",
        "actions": [
            {
                "type": "Redact",
                "configuration": {
                    "action_name": "AppendMessage",
                    "additional_parameters": {"message": "RAG_CONTEXT=API"},
                },
            }
        ],
        "evaluation_scores": [
            {
                "evaluation_algorithm_type": "toxicity",
                "model_type": "toxigen",
                "scores": [{"name": "toxicity", "value": 0.9}],
            }
        ],
    }

    deployed_runnable_sagemaker_inference_evaluator_fixture._call_inference_evaluator = MagicMock(  # type: ignore
        return_value=inference_evaluator_output
    )

    return deployed_runnable_sagemaker_inference_evaluator_fixture


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_with_instruct_action(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    inference_evaluator_output = {
        "name": "test-inference-evaluation",
        "actions": [
            {
                "type": "Instruct",
                "configuration": {
                    "action_name": "EventBridge",
                    "additional_parameters": {
                        "api_invokes": {
                            "create_event_bus": {"Name": "TestEventBridgeName"},
                            "put_events": {
                                "Entries": [
                                    {
                                        "Source": "com.mycompany.myapp",
                                        "Detail": (
                                            '{ "key1": "value1", "key2": "value2" }'
                                        ),
                                        "Resources": ["resource1", "resource2"],
                                        "DetailType": "myDetailType",
                                    }
                                ]
                            },
                        }
                    },
                },
            }
        ],
        "evaluation_scores": [
            {
                "evaluation_algorithm_type": "toxicity",
                "model_type": "toxigen",
                "scores": [{"name": "toxicity", "value": 0.9}],
            }
        ],
    }

    deployed_runnable_sagemaker_inference_evaluator_fixture._call_inference_evaluator = MagicMock(  # type: ignore
        return_value=inference_evaluator_output
    )

    return deployed_runnable_sagemaker_inference_evaluator_fixture


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_without_event_bus_name(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    inference_evaluator_output = {
        "name": "test-inference-evaluation",
        "actions": [
            {
                "type": "Instruct",
                "configuration": {
                    "action_name": "EventBridge",
                    "additional_parameters": {
                        "api_invokes": {
                            "create_event_bus": {},
                        }
                    },
                },
            }
        ],
        "evaluation_scores": [
            {
                "evaluation_algorithm_type": "toxicity",
                "model_type": "toxigen",
                "scores": [{"name": "toxicity", "value": 0.9}],
            }
        ],
    }

    deployed_runnable_sagemaker_inference_evaluator_fixture._call_inference_evaluator = MagicMock(  # type: ignore
        return_value=inference_evaluator_output
    )

    return deployed_runnable_sagemaker_inference_evaluator_fixture


@pytest.fixture
def deployed_runnable_sagemaker_inference_evaluator_with_custom_action(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> RunnableSageMakerInferenceEvaluator:
    inference_evaluator_output = {
        "name": "test-inference-evaluation",
        "actions": [
            {
                "type": "Custom",
                "configuration": {
                    "action_name": "MyCustomAction",
                    "additional_parameters": {"message": "Message for custom action!"},
                },
            }
        ],
        "evaluation_scores": [
            {
                "evaluation_algorithm_type": "toxicity",
                "model_type": "toxigen",
                "scores": [{"name": "toxicity", "value": 0.9}],
            }
        ],
    }

    deployed_runnable_sagemaker_inference_evaluator_fixture._call_inference_evaluator = MagicMock(  # type: ignore
        return_value=inference_evaluator_output
    )

    return deployed_runnable_sagemaker_inference_evaluator_fixture


def test_inference_evaluator_invoke_without_deploying(
    runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(ValueError) as invoke_error:
        runnable_sagemaker_inference_evaluator_fixture.invoke(input="toxic text")
        assert invoke_error.value == (
            "Inference evaluator must be deployed prior to invocation"
        )


def test_convert_text_to_prompt(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(ValueError) as invoke_error:
        deployed_runnable_sagemaker_inference_evaluator_fixture._convert_text_to_prompt(
            None, "mock-text"
        )


def test_convert_prompt_to_text(
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(ValueError) as invoke_error:
        deployed_runnable_sagemaker_inference_evaluator_fixture._convert_prompt_to_text(
            None
        )


@patch(inference_evaluator_method_path("_deploy_inference_component"))
@patch(inference_evaluator_method_path("_deploy_endpoint"))
def test_inference_evaluator_deploy_when_resources_exist(
    mock_deploy_inference_component: MagicMock,
    mock_deploy_endpoint: MagicMock,
    deployed_runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> None:
    mock_deploy_inference_component.assert_not_called()
    mock_deploy_endpoint.assert_not_called()

    assert (
        deployed_runnable_sagemaker_inference_evaluator_fixture.endpoint_name
        is not None
    )
    assert (
        deployed_runnable_sagemaker_inference_evaluator_fixture.inference_component_name
        is not None
    )


@patch(inference_evaluator_method_path("_deploy_inference_component"))
@patch(inference_evaluator_method_path("_deploy_endpoint"))
def test_inference_evaluator_deploy_when_resources_does_not_exist(
    mock_deploy_inference_component: MagicMock,
    mock_deploy_endpoint: MagicMock,
    runnable_sagemaker_inference_evaluator_fixture: RunnableSageMakerInferenceEvaluator,
) -> None:
    runnable_sagemaker_inference_evaluator_fixture._endpoint_exists = MagicMock(  # type: ignore
        return_value=False
    )
    runnable_sagemaker_inference_evaluator_fixture._inference_component_exists = (  # type: ignore
        MagicMock(return_value=False)
    )

    runnable_sagemaker_inference_evaluator_fixture.deploy(
        endpoint_name="test-endpoint",
        inference_component_name="test-inference-component",
        execution_role="arn:aws:iam::<account-number>:role/test-exec-role",
    )

    mock_deploy_inference_component.assert_called_once()
    mock_deploy_endpoint.assert_called_once()

    assert runnable_sagemaker_inference_evaluator_fixture.endpoint_name is not None
    assert (
        runnable_sagemaker_inference_evaluator_fixture.inference_component_name
        is not None
    )


def test_inference_evaluator_when_deny_action_is_triggered(
    deployed_runnable_sagemaker_inference_evaluator_with_deny_action: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(InferenceEvaluatorError) as inference_evaluator_error:
        deployed_runnable_sagemaker_inference_evaluator_with_deny_action.invoke(
            input="toxic text"
        )

        assert inference_evaluator_error.value == (
            "Please provide a safe prompt. The toxicity score is above 0.5."
        )


def test_inference_evaluator_when_redact_action_is_triggered(
    deployed_runnable_sagemaker_inference_evaluator_with_redact_action: RunnableSageMakerInferenceEvaluator,
) -> None:
    output = deployed_runnable_sagemaker_inference_evaluator_with_redact_action.invoke(
        input="toxic text"
    )

    assert output == "toxic text\nRAG_CONTEXT=API"


@patch("boto3.client")
def test_inference_evaluator_when_instruct_action_is_triggered(
    mock_boto_client: MagicMock,
    deployed_runnable_sagemaker_inference_evaluator_with_instruct_action: RunnableSageMakerInferenceEvaluator,
) -> None:
    with patch(
        "boto3.client.return_value.describe_event_bus"
    ) as mock_describe_event_bus:
        mock_describe_event_bus.side_effect = Exception("event-bus does not exist.")

        output = (
            deployed_runnable_sagemaker_inference_evaluator_with_instruct_action.invoke(
                input="toxic text"
            )
        )

        mock_boto_client.assert_called_once_with("events")
        mock_boto_client.return_value.create_event_bus.assert_called()
        mock_boto_client.return_value.put_events.assert_called()

        assert output == "toxic text"


def test_inference_evaluator_when_custom_action_is_not_registered(
    deployed_runnable_sagemaker_inference_evaluator_with_custom_action: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(ValueError) as inference_evaluator_error:
        deployed_runnable_sagemaker_inference_evaluator_with_custom_action.invoke(
            input="toxic text"
        )
        assert inference_evaluator_error.value == (
            "Invalid action name MyCustomAction for CustomBehavior"
        )


def test_inference_evaluator_when_custom_action_is_triggered(
    deployed_runnable_sagemaker_inference_evaluator_with_custom_action: RunnableSageMakerInferenceEvaluator,
) -> None:
    class MyCustomAction(ActionInterface):
        def run(self, input_text: str, configuration: dict) -> str:
            message = configuration["additional_parameters"]["message"]
            return message

    output = deployed_runnable_sagemaker_inference_evaluator_with_custom_action.invoke(
        input="toxic text"
    )
    assert output == "Message for custom action!"


def test_inference_evaluator_when_instruct_action_is_triggered_without_event_bus_name(
    deployed_runnable_sagemaker_inference_evaluator_without_event_bus_name: RunnableSageMakerInferenceEvaluator,
) -> None:
    with pytest.raises(ValueError) as create_bus_error:
        deployed_runnable_sagemaker_inference_evaluator_without_event_bus_name.invoke(
            input="toxic text"
        )
        assert create_bus_error.value == (
            "Name for the event bus is not provided in the inference evaluator config"
        )
