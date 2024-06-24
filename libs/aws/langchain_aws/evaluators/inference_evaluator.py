import json
import time
from typing import Any, Optional, cast

from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from langchain_aws.evaluators.action_registry import ACTION_REGISTRY


class RunnableSageMakerInferenceEvaluator(Runnable[Input, Output]):
    inference_evaluator_image = (
        "551039116605.dkr.ecr.us-west-2.amazonaws.com/mockingbird-beta"
    )

    # Endpoint name and inference component name is populated
    #  after "RunnableSageMakerInferenceEvaluator" is deployed
    endpoint_name = None
    inference_component_name = None

    def __init__(
        self, inference_evaluator_name: str, inference_evaluators_configuration: dict
    ):
        self.inference_evaluator_name = inference_evaluator_name
        self.inference_evaluators_configuration = inference_evaluators_configuration

        try:
            import boto3

            my_session = boto3.session.Session()

            self.sm_client = my_session.client("sagemaker")
            self.sm_runtime_client = boto3.client(service_name="sagemaker-runtime")

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )

        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "NeptuneGraph requires a boto3 version 1.28.38 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

    def _endpoint_exists(self, endpoint_name: str) -> bool:
        """
        Return True if and only if endpoint exists with name <endpoint_name>
        otherwise return False.

        :param endpoint_name: name of a sagemaker endpoint
        """
        try:
            self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            return True
        except Exception:
            return False

    def _inference_component_exists(self, inference_component_name: str) -> bool:
        """
        Return True if and only if inference component exists
        with name <inference_component_name> otherwise return False.

        :param inference_component_name: name of an inference component
        """
        try:
            self.sm_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
            return True
        except Exception:
            return False

    def _deploy_endpoint(
        self, endpoint_name: str, execution_role: str, instance_type: str
    ) -> None:
        endpoint_config_name = f"{endpoint_name}-config"
        self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ExecutionRoleArn=execution_role,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                    "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
                }
            ],
        )
        self.sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

        status = "Creating"
        while status == "Creating":
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            time.sleep(30)

        if status == "Failed":
            raise ClientError(
                "Failed to create a new endpoint {0}"
                "See inference component logs for more details".format(endpoint_name)
            )

    def _deploy_inference_component(
        self, endpoint_name: str, inference_component_name: str
    ) -> None:
        # Do not change this variable name before changing it in the
        # mockingbird container
        environment_variables = {
            "INFERENCE_EVALUATOR_CONFIG": json.dumps(
                self.inference_evaluators_configuration
            )
        }

        self.sm_client.create_inference_component(
            InferenceComponentName=inference_component_name,
            EndpointName=endpoint_name,
            VariantName="AllTraffic",
            Specification={
                "ComputeResourceRequirements": {
                    "NumberOfAcceleratorDevicesRequired": 1,
                    "MinMemoryRequiredInMb": 1024,
                },
                "Container": {
                    "Environment": environment_variables,
                    "Image": self.inference_evaluator_image,
                },
            },
            RuntimeConfig={"CopyCount": 1},
        )

        status = "Creating"
        while status == "Creating":
            response = self.sm_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
            status = response["InferenceComponentStatus"]
            time.sleep(30)

        if status == "Failed":
            raise ClientError(
                "Failed to create a new inference component {0}. "
                "See inference component logs for more details".format(
                    inference_component_name
                )
            )

    def _convert_text_to_prompt(self, prompt: Any, text: str) -> Any:
        if isinstance(prompt, StringPromptValue):
            return StringPromptValue(text=text)
        elif isinstance(prompt, str):
            return text
        elif isinstance(prompt, ChatPromptValue):
            # Copy the messages because we may need to mutate them.
            messages = prompt.to_messages()
            message = messages[self.chat_message_index]

            if isinstance(message, HumanMessage):
                messages[self.chat_message_index] = HumanMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            if isinstance(message, AIMessage):
                messages[self.chat_message_index] = AIMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            return ChatPromptValue(messages=messages)
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

    def _convert_prompt_to_text(self, prompt: Any) -> str:
        input_text = str()
        if isinstance(prompt, StringPromptValue):
            input_text = prompt.text
        elif isinstance(prompt, str):
            input_text = prompt
        elif isinstance(prompt, ChatPromptValue):
            message = prompt.messages[-1]
            self.chat_message_index = len(prompt.messages) - 1
            if isinstance(message, HumanMessage):
                input_text = cast(str, message.content)

            if isinstance(message, AIMessage):
                input_text = cast(str, message.content)
        else:
            raise ValueError(
                f"Invalid input type {type(prompt)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
        return input_text

    def _call_inference_evaluator(self, request: str) -> dict:
        if self.endpoint_name and self.inference_component_name:
            response = self.sm_runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                InferenceComponentName=self.inference_component_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(request),
            )
        else:
            raise ValueError("Inference evaluator must be deployed prior to invocation")

        return json.loads(response["Body"].read().decode("utf-8"))

    def _execute_actions(
        self, input_text: str, inference_evaluator_response: dict
    ) -> str:
        # The input_text could be modified with redact action
        output_text = input_text
        for action_data in inference_evaluator_response["actions"]:
            if action_data["type"] in ACTION_REGISTRY:
                action = ACTION_REGISTRY[action_data["type"]]()  # type: ignore

            elif action_data["type"] == "Custom":
                if action_data["configuration"]["action_name"] in ACTION_REGISTRY:
                    action = ACTION_REGISTRY[  # type: ignore
                        action_data["configuration"]["action_name"]
                    ]()
                else:
                    action_name = action_data["configuration"]["action_name"]
                    raise ValueError(
                        f"Invalid action name {action_name} for CustomBehavior"
                    )
            else:
                raise ValueError(f'Invalid action type {action_data["type"]}.')

            output_text = action.run(input_text, action_data["configuration"])
        return output_text

    def deploy(
        self,
        endpoint_name: str,
        inference_component_name: str,
        execution_role: str,
        instance_type: str = "ml.g5.xlarge",
    ) -> None:
        if not self._endpoint_exists(endpoint_name):
            self._deploy_endpoint(endpoint_name, execution_role, instance_type)

        if not self._inference_component_exists(inference_component_name):
            self._deploy_inference_component(endpoint_name, inference_component_name)

        # Set the endpoint name and inference component name once they are deployed
        # These variables are used to verify if the resources are deployed
        # before invocation.
        self.endpoint_name = endpoint_name
        self.inference_component_name = inference_component_name

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        input_text = self._convert_prompt_to_text(prompt=input)
        inference_evaluator_response = self._call_inference_evaluator(input_text)

        output_text = self._execute_actions(input_text, inference_evaluator_response)
        return self._convert_text_to_prompt(prompt=input, text=output_text)
