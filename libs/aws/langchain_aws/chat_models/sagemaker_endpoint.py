"""Sagemaker Chat Model."""

import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_aws.utils import ContentHandlerBase

logger = logging.getLogger(__name__)


class ChatModelContentHandler(ContentHandlerBase[List[Dict[str, Any]], BaseMessage]):
    """Content handler for ChatSagemakerEndpoint class."""


class ChatSagemakerEndpoint(BaseChatModel):
    """A chat model that uses a HugguingFace TGI compatible SageMaker Endpoint.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """

    """
    Args:        

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        client: boto3 client for Sagemaker Endpoint

        content_handler: Implementation for model specific ChatContentHandler 


    Example:
        .. code-block:: python

            from langchain_aws.chat_models.sagemaker_endpoint import 
            ChatSagemakerEndpoint
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = ChatSagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )
        
            # Usage with Inference Component
            se = ChatSagemakerEndpoint(
                endpoint_name=endpoint_name,
                inference_component_name=inference_component_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )

        #Use with boto3 client
            client = boto3.client(
                        "sagemaker-runtime",
                        region_name=region_name
                    )

            se = ChatSagemakerEndpoint(
                endpoint_name=endpoint_name,
                client=client
            )

    """
    client: Any = None
    """Boto3 client for sagemaker runtime"""

    endpoint_name: str = ""
    """The name of the endpoint from the deployed Sagemaker model.
    Must be unique within an AWS Region."""

    inference_component_name: Optional[str] = None
    """Optional name of the inference component to invoke 
    if specified with endpoint name."""

    region_name: str = ""
    """The aws region where the Sagemaker model is deployed, eg. `us-west-2`."""

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    content_handler: ChatModelContentHandler
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    streaming: bool = False
    """Whether to stream the results."""

    """
     Example:
        .. code-block:: python

        from langchain_community.llms.sagemaker_endpoint import ChatContentHandler

        class ContentHandler(ChatContentHandler):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: List[Dict[str, Any]], model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> BaseMessage:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Dont do anything if client provided externally"""
        if self.client is not None:
            return self

        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if self.credentials_profile_name is not None:
                    session = boto3.Session(profile_name=self.credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                self.client = session.client(
                    "sagemaker-runtime", region_name=self.region_name
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return self

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"inference_component_name": self.inference_component_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_endpoint"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_sagemaker_chat"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "sagemaker"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.region_name:
            attributes["region_name"] = self.region_name

        return attributes
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        sagemaker_messages = _messages_to_sagemaker(messages)
        logger.debug(f"input message to sagemaker: {sagemaker_messages}")
        invocation_params = {
            "EndpointName": self.endpoint_name,
            "Body": self.content_handler.transform_input(
                sagemaker_messages, _model_kwargs
            ),
            "ContentType": self.content_handler.content_type,
            "Accept": self.content_handler.accepts,
            **_endpoint_kwargs,
        }

        # If inference_compoent_name is specified, append it to invocation_params
        if self.inference_component_name:
            invocation_params["InferenceComponentName"] = self.inference_component_name

        try:
            response = self.client.invoke_endpoint(**invocation_params)
        except Exception as e:
            logging.error(f"Error raised by inference endpoint: {e}")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e
        logger.info(f"The message received from SageMaker: {response['Body']}")

        response_message = self.content_handler.transform_output(response["Body"])

        return ChatResult(generations=[ChatGeneration(message=response_message)])


def _messages_to_sagemaker(
    messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    # Merge system, human, ai message runs because Anthropic expects (at most) 1
    # system message then alternating human/ai messages.
    sagemaker_messages: List[Dict[str, Any]] = []
    if not isinstance(messages, list):
        messages = [messages]

    messages = merge_message_runs(messages)
    for msg in messages:
        content = msg.content
        if isinstance(msg, HumanMessage):
            # If there's a human, tool, human message sequence, the
            # tool message will be merged with the first human message, so the second
            # human message will now be preceded by a human message and should also
            # be merged with it.
            if sagemaker_messages and sagemaker_messages[-1]["role"] == "user":
                sagemaker_messages[-1]["content"].extend(content)
            else:
                sagemaker_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            sagemaker_messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, SystemMessage):
            sagemaker_messages.insert(0, {"role": "system", "content": content})
        else:
            raise ValueError(f"Unsupported message type {type(msg)}")
    return sagemaker_messages
