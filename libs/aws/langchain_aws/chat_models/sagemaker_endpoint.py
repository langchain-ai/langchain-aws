"""Sagemaker Chat Model."""
import io
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional

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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_aws.utils import (
    ContentHandlerBase,
    create_aws_client,
    enforce_stop_tokens,
)

logger = logging.getLogger(__name__)


class ChatLineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:

    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...

    While usually each PayloadPart event from the event stream will
    contain a byte array with a full json, this is not guaranteed
    and some of the json objects may be split acrossPayloadPart events.

    For example:

    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}


    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character)
    within the buffer via the 'scan_lines' function.
    It maintains the position of the last read position to ensure
    that previous bytes are not exposed again.

    For more details see:
    https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream: Any) -> None:
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self) -> "ChatLineIterator":
        return self

    def __next__(self) -> Any:
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
                if "PayloadPart" in chunk:
                    self.buffer.seek(0, io.SEEK_END)
                    self.buffer.write(chunk["PayloadPart"]["Bytes"])
                    continue
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    remaining = self.buffer.getvalue()[self.read_pos :]
                    self.read_pos = self.buffer.getbuffer().nbytes
                    return remaining
                raise
            if line:
                self.read_pos += len(line)
                return line[:-1] if line[-1] == ord("\n") else line
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


class ChatModelContentHandler(ContentHandlerBase[List[Dict[str, Any]], BaseMessage]):
    """Content handler for ChatSagemakerEndpoint class."""


class ChatSagemakerEndpoint(BaseChatModel):
    """A chat model that uses a HuggingFace TGI compatible SageMaker Endpoint.

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
    Key Args:

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        client: boto3 client for Sagemaker Endpoint
        
        endpoint_name: The name of the endpoint from the deployed Sagemaker model.

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

    region_name: Optional[str] = ""
    """The aws region, e.g., `us-west-2`. 

    Falls back to AWS_REGION or AWS_DEFAULT_REGION env variable or region specified in 
    ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files.

    Profile should either have access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used. 
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id. 

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.
    """

    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key. 

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token. 

    If provided, aws_access_key_id and aws_secret_access_key must 
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    endpoint_url: Optional[str] = Field(default=None, alias="base_url")
    """Needed if you don't want to default to us-east-1 endpoint"""

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

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
        """Skip creating new client if passed in constructor"""
        if self.client is None:
            self.client = create_aws_client(
                region_name=self.region_name,
                credentials_profile_name=self.credentials_profile_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                endpoint_url=self.endpoint_url,
                config=self.config,
                service_name="sagemaker-runtime",
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

    def _format_messages_request(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
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

        # If inference_component_name is specified, append it to invocation_params
        if self.inference_component_name:
            invocation_params["InferenceComponentName"] = self.inference_component_name
        return invocation_params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        invocation_params = self._format_messages_request(messages=messages, **kwargs)
        try:
            resp = self.client.invoke_endpoint_with_response_stream(**invocation_params)
            iterator = ChatLineIterator(resp["Body"])

            for line in iterator:
                text = self.content_handler.transform_output(line)
                if stop is not None:
                    text = enforce_stop_tokens(text, stop)

                if text:
                    generation_chunk = ChatGenerationChunk(message=text)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    yield generation_chunk

        except Exception as e:
            logger.exception("Error raised by streaming inference endpoint")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        invocation_params = self._format_messages_request(messages=messages, **kwargs)
        try:
            response = self.client.invoke_endpoint(**invocation_params)
        except Exception as e:
            logging.error(f"Error raised by inference endpoint: {e}")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e
        logger.debug(f"The message received from SageMaker: {response['Body']}")

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
