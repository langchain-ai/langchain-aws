"""Amazon Nova Sonic bidirectional streaming chat model."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LangSmithParams
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Default audio configuration constants
DEFAULT_INPUT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_SAMPLE_SIZE_BITS = 16
DEFAULT_CHANNEL_COUNT = 1
DEFAULT_AUDIO_MEDIA_TYPE = "audio/lpcm"
DEFAULT_TEXT_MEDIA_TYPE = "text/plain"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Voice IDs available in Nova Sonic v1
_V1_VOICE_IDS = frozenset(
    {
        "tiffany",
        "matthew",
        "amy",
        "ambre",
        "florian",
        "beatrice",
        "lorenzo",
        "greta",
        "lennart",
        "lupe",
        "carlos",
    }
)

# Voice IDs added in Nova 2 Sonic
_V2_VOICE_IDS = frozenset(
    {
        "olivia",
        "kiara",
        "arjun",
        "tina",
        "carolina",
        "leo",
    }
)

# All supported voice IDs
_ALL_VOICE_IDS = _V1_VOICE_IDS | _V2_VOICE_IDS


def _is_v2_model(model_id: str) -> bool:
    """Return True if the model ID refers to Nova 2 Sonic."""
    return "nova-2-sonic" in model_id


def _check_nova_sonic_deps() -> None:
    """Verify that the Nova Sonic SDK dependencies are installed.

    Raises:
        ImportError: If ``aws_sdk_bedrock_runtime`` is not installed.
    """
    try:
        import aws_sdk_bedrock_runtime  # noqa: F401
    except ImportError as exc:
        msg = (
            "Could not import aws_sdk_bedrock_runtime. "
            'Please install it with: pip install "langchain-aws[nova-sonic]"'
        )
        raise ImportError(msg) from exc


class NovaSonicSession:
    """Manages a single bidirectional streaming session with Nova Sonic.

    This class handles the event protocol for sending and receiving audio/text
    over the ``InvokeModelWithBidirectionalStream`` API. Sessions are created
    via :meth:`ChatBedrockNovaSonic.create_session` and should be used as an
    async context manager.

    Example::

        async with model.create_session(system_prompt="Be helpful.") as session:
            await session.send_audio_chunk(audio_bytes)
            async for event in session.receive_events():
                handle(event)
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        *,
        system_prompt: Optional[str] = None,
        voice_id: str = "matthew",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE,
        output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
        audio_media_type: str = DEFAULT_AUDIO_MEDIA_TYPE,
        endpointing_sensitivity: Optional[str] = None,
    ) -> None:
        """Initialize a Nova Sonic session.

        Args:
            client: The ``BedrockRuntimeClient`` instance.
            model_id: The Nova Sonic model identifier.
            system_prompt: Optional system prompt for the conversation.
            voice_id: Voice identifier for audio output.
            max_tokens: Maximum tokens for inference.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            input_sample_rate: Sample rate for input audio in Hz.
            output_sample_rate: Sample rate for output audio in Hz.
            audio_media_type: Media type for audio data.
            endpointing_sensitivity: Turn-detection sensitivity
                (HIGH/MEDIUM/LOW). Nova 2 Sonic only.
        """
        self._client = client
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._voice_id = voice_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._audio_media_type = audio_media_type
        self._endpointing_sensitivity = endpointing_sensitivity

        self._stream: Any = None
        self._is_active = False
        self._input_closed = False
        self._prompt_name = str(uuid.uuid4())
        self._content_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())
        self._role: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Whether the session is currently active."""
        return self._is_active

    async def _send_event(self, event_json: str) -> None:
        """Send a JSON event to the bidirectional stream.

        Args:
            event_json: JSON-encoded event string.
        """
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    async def start(self) -> None:
        """Start the bidirectional streaming session.

        Sends the session start, prompt start, and system prompt events
        in the required order.

        Raises:
            RuntimeError: If the session is already active.
        """
        if self._is_active:
            msg = "Session is already active."
            raise RuntimeError(msg)

        from aws_sdk_bedrock_runtime.client import (
            InvokeModelWithBidirectionalStreamOperationInput,
        )

        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model_id)
        )
        self._is_active = True

        # Session start
        session_start_payload: Dict[str, Any] = {
            "inferenceConfiguration": {
                "maxTokens": self._max_tokens,
                "topP": self._top_p,
                "temperature": self._temperature,
            }
        }
        if self._endpointing_sensitivity is not None:
            session_start_payload["turnDetectionConfiguration"] = {
                "endpointingSensitivity": self._endpointing_sensitivity,
            }
        session_start = json.dumps({"event": {"sessionStart": session_start_payload}})
        await self._send_event(session_start)

        # Prompt start
        prompt_start = json.dumps(
            {
                "event": {
                    "promptStart": {
                        "promptName": self._prompt_name,
                        "textOutputConfiguration": {
                            "mediaType": DEFAULT_TEXT_MEDIA_TYPE,
                        },
                        "audioOutputConfiguration": {
                            "mediaType": self._audio_media_type,
                            "sampleRateHertz": self._output_sample_rate,
                            "sampleSizeBits": DEFAULT_SAMPLE_SIZE_BITS,
                            "channelCount": DEFAULT_CHANNEL_COUNT,
                            "voiceId": self._voice_id,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }
        )
        await self._send_event(prompt_start)

        # System prompt — always required as the first content by Nova Sonic.
        # Falls back to a default if none was provided.
        system_prompt = self._system_prompt or DEFAULT_SYSTEM_PROMPT
        await self._send_system_prompt(system_prompt)

    async def _send_system_prompt(self, prompt: str) -> None:
        """Send a system prompt to the session.

        Args:
            prompt: The system prompt text.
        """
        content_name = str(uuid.uuid4())

        content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {
                            "mediaType": DEFAULT_TEXT_MEDIA_TYPE,
                        },
                    }
                }
            }
        )
        await self._send_event(content_start)

        text_input = json.dumps(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "content": prompt,
                    }
                }
            }
        )
        await self._send_event(text_input)

        content_end = json.dumps(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )
        await self._send_event(content_end)

    async def send_text(self, text: str, *, interactive: bool = True) -> None:
        """Send a text message as user input.

        Args:
            text: The user text to send.
            interactive: If ``True`` (default), the text is part of an
                ongoing conversation turn that relies on audio turn
                detection.  If ``False``, the text is treated as a
                complete turn and the model will respond immediately
                without requiring audio input.

        Raises:
            RuntimeError: If the session is not active.
        """
        if not self._is_active:
            msg = "Session is not active. Call start() first."
            raise RuntimeError(msg)

        content_name = str(uuid.uuid4())

        content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": interactive,
                        "role": "USER",
                        "textInputConfiguration": {
                            "mediaType": DEFAULT_TEXT_MEDIA_TYPE,
                        },
                    }
                }
            }
        )
        await self._send_event(content_start)

        text_input = json.dumps(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "content": text,
                    }
                }
            }
        )
        await self._send_event(text_input)

        content_end = json.dumps(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )
        await self._send_event(content_end)

    async def start_audio_input(self) -> None:
        """Start an audio input stream.

        Call this before sending audio chunks. When done, call
        :meth:`end_audio_input`.

        Raises:
            RuntimeError: If the session is not active.
        """
        if not self._is_active:
            msg = "Session is not active. Call start() first."
            raise RuntimeError(msg)

        self._audio_content_name = str(uuid.uuid4())

        audio_content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": self._audio_media_type,
                            "sampleRateHertz": self._input_sample_rate,
                            "sampleSizeBits": DEFAULT_SAMPLE_SIZE_BITS,
                            "channelCount": DEFAULT_CHANNEL_COUNT,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )
        await self._send_event(audio_content_start)

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Send a chunk of audio data to the stream.

        Args:
            audio_bytes: Raw audio bytes (PCM format expected).

        Raises:
            RuntimeError: If the session is not active.
        """
        if not self._is_active:
            return

        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        audio_event = json.dumps(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "content": encoded,
                    }
                }
            }
        )
        await self._send_event(audio_event)

    async def end_audio_input(self) -> None:
        """End the current audio input stream."""
        if not self._is_active:
            return

        audio_content_end = json.dumps(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                    }
                }
            }
        )
        await self._send_event(audio_content_end)

    async def receive_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Receive and yield parsed events from the model.

        Yields dictionaries with the following possible structures:

        - ``{"type": "text", "role": str, "text": str}`` for text output
        - ``{"type": "audio", "audio": bytes}`` for audio output
        - ``{"type": "content_start", "role": str, ...}`` for content start
        - ``{"type": "content_end"}`` for content end

        Yields:
            Parsed event dictionaries.
        """
        try:
            while True:
                output = await self._stream.await_output()
                result = await output[1].receive()

                if result is None or not (result.value and result.value.bytes_):
                    continue

                response_data = result.value.bytes_.decode("utf-8")
                json_data = json.loads(response_data)

                if "event" not in json_data:
                    continue

                event = json_data["event"]

                if "contentStart" in event:
                    content_start = event["contentStart"]
                    self._role = content_start.get("role")
                    yield {
                        "type": "content_start",
                        "role": self._role,
                        "content_start": content_start,
                    }

                elif "textOutput" in event:
                    text = event["textOutput"]["content"]
                    yield {
                        "type": "text",
                        "role": self._role or "ASSISTANT",
                        "text": text,
                    }

                elif "audioOutput" in event:
                    audio_content = event["audioOutput"]["content"]
                    audio_bytes = base64.b64decode(audio_content)
                    yield {
                        "type": "audio",
                        "audio": audio_bytes,
                    }

                elif "contentEnd" in event:
                    yield {"type": "content_end"}

                elif "usageEvent" in event:
                    yield {
                        "type": "usage",
                        "usage": event["usageEvent"],
                    }

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error processing Nova Sonic responses: %s", exc)
            raise

    async def close_input(self) -> None:
        """Signal end of input to the model.

        Sends ``promptEnd`` and ``sessionEnd`` events and closes the
        input stream. The output stream remains readable so that
        ``receive_events`` can continue to yield responses.
        """
        if self._input_closed:
            return
        self._input_closed = True

        try:
            prompt_end = json.dumps(
                {
                    "event": {
                        "promptEnd": {
                            "promptName": self._prompt_name,
                        }
                    }
                }
            )
            await self._send_event(prompt_end)

            session_end = json.dumps({"event": {"sessionEnd": {}}})
            await self._send_event(session_end)

            await self._stream.input_stream.close()
        except Exception as exc:
            logger.warning("Error closing Nova Sonic input: %s", exc)

    async def end(self) -> None:
        """End the session and close the stream.

        Sends prompt end and session end events (if not already sent),
        then marks the session as inactive.
        """
        if not self._is_active:
            return

        self._is_active = False

        await self.close_input()


class ChatBedrockNovaSonic(BaseChatModel):
    """Chat model for Amazon Nova Sonic bidirectional streaming.

    This provides a LangChain integration for Amazon Nova Sonic's
    bidirectional streaming API (``InvokeModelWithBidirectionalStream``).
    Nova Sonic enables real-time speech-to-speech conversations over a
    persistent, full-duplex streaming connection. Unlike the Converse API
    used by :class:`ChatBedrockConverse`, this maintains a persistent
    full-duplex connection for continuous audio streaming.

    !!! warning "Experimental"
        This integration requires the ``aws-sdk-bedrock-runtime`` package
        which is under active development. Install with:
        ``pip install "langchain-aws[nova-sonic]"``

    For simple text interactions, use :meth:`ainvoke` or :meth:`astream`.
    For full audio streaming, use :meth:`create_session` to get a
    :class:`NovaSonicSession`.

    Quick start::

        import asyncio
        from langchain_aws.chat_models.bedrock_nova_sonic import ChatBedrockNovaSonic

        model = ChatBedrockNovaSonic(
            model_id="amazon.nova-sonic-v1:0",
            region_name="us-east-1",
        )

        # Text-only conversation
        response = asyncio.run(
            model.ainvoke("Hello, how are you?")
        )
        print(response.content)

    Audio streaming::

        import asyncio
        from langchain_aws.chat_models.bedrock_nova_sonic import (
            ChatBedrockNovaSonic,
            NovaSonicSession,
        )

        async def stream_audio():
            model = ChatBedrockNovaSonic(
                model_id="amazon.nova-sonic-v1:0",
                region_name="us-east-1",
                voice_id="matthew",
            )

            async with model.create_session() as session:
                # Send audio chunks
                await session.send_audio_chunk(audio_bytes)

                # Receive responses
                async for event in session.receive_events():
                    if event["type"] == "audio":
                        play(event["audio"])
                    elif event["type"] == "text":
                        print(event["text"])

        asyncio.run(stream_audio())

    Args:
        model_id: The Nova Sonic model identifier.
        region_name: AWS region for the Bedrock endpoint.
        voice_id: Voice identifier for audio output.
        system_prompt: Default system prompt for sessions.
        max_tokens: Maximum tokens for inference.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        input_sample_rate: Sample rate for input audio in Hz.
        output_sample_rate: Sample rate for output audio in Hz.
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    model_id: str = Field(
        default="amazon.nova-sonic-v1:0",
        alias="model",
        description="The Nova Sonic model identifier.",
    )
    region_name: Optional[str] = Field(
        default=None,
        description="AWS region for the Bedrock endpoint.",
    )
    credentials_profile_name: Optional[str] = Field(
        default=None,
        exclude=True,
        description=(
            "The name of the profile in the ~/.aws/credentials or ~/.aws/config files."
        ),
    )
    aws_access_key_id: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None),
        description="AWS access key ID.",
    )
    aws_secret_access_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None),
        description="AWS secret access key.",
    )
    aws_session_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None),
        description="AWS session token.",
    )
    endpoint_url: Optional[str] = Field(
        default=None,
        alias="base_url",
        description="Custom Bedrock endpoint URL.",
    )
    bedrock_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AWS_BEARER_TOKEN_BEDROCK", default=None),
        alias="api_key",
        description=(
            "Bedrock API key for bearer-token authentication. "
            "Warning: sets the AWS_BEARER_TOKEN_BEDROCK environment variable "
            "at the process level, so it is not compatible with multi-tenant "
            "deployments using different API keys in the same process."
        ),
    )
    voice_id: str = Field(
        default="matthew",
        description="Voice identifier for audio output.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Default system prompt for sessions.",
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for inference.",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p sampling parameter.",
    )
    input_sample_rate: int = Field(
        default=DEFAULT_INPUT_SAMPLE_RATE,
        description="Sample rate for input audio in Hz.",
    )
    output_sample_rate: int = Field(
        default=DEFAULT_OUTPUT_SAMPLE_RATE,
        description="Sample rate for output audio in Hz.",
    )
    endpointing_sensitivity: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = Field(
        default=None,
        description=(
            "Turn-detection sensitivity controlling how quickly the model "
            "detects end of speech. Only supported on Nova 2 Sonic."
        ),
    )

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the Nova Sonic SDK is installed and create client."""
        _check_nova_sonic_deps()

        is_v2 = _is_v2_model(self.model_id)

        # Validate voice_id
        if self.voice_id not in _ALL_VOICE_IDS:
            raise ValueError(
                f"Unknown voice_id '{self.voice_id}'. "
                f"Supported voices: {sorted(_ALL_VOICE_IDS)}"
            )
        if not is_v2 and self.voice_id in _V2_VOICE_IDS:
            raise ValueError(
                f"Voice '{self.voice_id}' is only available on Nova 2 Sonic. "
                f"Use model 'amazon.nova-2-sonic-v1:0' or choose from: "
                f"{sorted(_V1_VOICE_IDS)}"
            )

        # Validate endpointing_sensitivity (v2 only)
        if self.endpointing_sensitivity is not None and not is_v2:
            raise ValueError(
                "endpointing_sensitivity is only supported on Nova 2 Sonic. "
                "Use model 'amazon.nova-2-sonic-v1:0' or remove this parameter."
            )

        if self._client is None:
            from langchain_aws.utils import create_aws_bedrock_runtime_client

            self._client = create_aws_bedrock_runtime_client(
                region_name=self.region_name,
                credentials_profile_name=self.credentials_profile_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                endpoint_url=self.endpoint_url,
                api_key=self.bedrock_api_key,
            )
        return self

    @asynccontextmanager
    async def create_session(
        self,
        *,
        system_prompt: Optional[str] = None,
        voice_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[NovaSonicSession]:
        """Create a bidirectional streaming session.

        Use as an async context manager. The session is automatically
        started and ended.

        Args:
            system_prompt: Override the default system prompt.
            voice_id: Override the default voice ID.
            max_tokens: Override the default max tokens.
            temperature: Override the default temperature.
            top_p: Override the default top-p.

        Yields:
            A started :class:`NovaSonicSession`.

        Example::

            async with model.create_session() as session:
                await session.send_text("Hello!")
                async for event in session.receive_events():
                    if event["type"] == "text":
                        print(event["text"])
        """
        session = NovaSonicSession(
            client=self._client,
            model_id=self.model_id,
            system_prompt=system_prompt or self.system_prompt,
            voice_id=voice_id or self.voice_id,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
            endpointing_sensitivity=self.endpointing_sensitivity,
        )
        await session.start()
        try:
            yield session
        finally:
            await session.end()

    # -- LangChain BaseChatModel interface --

    @property
    def _llm_type(self) -> str:
        return "amazon_bedrock_nova_sonic"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="amazon_bedrock",
            ls_model_name=self.model_id,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for tracing."""
        return {
            "model_id": self.model_id,
            "voice_id": self.voice_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "endpointing_sensitivity": self.endpointing_sensitivity,
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_aws", "chat_models"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
            "bedrock_api_key": "AWS_BEARER_TOKEN_BEDROCK",
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation — delegates to async via event loop.

        Nova Sonic is inherently async. This method enables sync callers
        (e.g. ``invoke``) by running ``_agenerate`` in a new event loop.
        Prefer ``ainvoke`` / ``astream`` for async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self._agenerate(messages, stop=stop, run_manager=None, **kwargs),
                ).result()
        return asyncio.run(
            self._agenerate(messages, stop=stop, run_manager=None, **kwargs)
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from text messages via Nova Sonic.

        Extracts system and user messages, sends them through a
        bidirectional session, and collects the text response.

        Nova Sonic requires an active audio input stream to trigger
        response generation via its turn-detection mechanism.  This
        method sends user text with ``interactive=True``, opens an
        audio stream, feeds silence, and collects the assistant's
        text reply.

        Note:
            Only ``SystemMessage`` and ``HumanMessage`` are processed from
            input. ``AIMessage`` and other message types are ignored because
            Nova Sonic maintains conversation state within the bidirectional
            stream. Previous assistant responses do not need to be sent back
            as context.

        Args:
            messages: Input messages.
            stop: Stop sequences (not supported by Nova Sonic).
            run_manager: Callback manager.
            **kwargs: Additional keyword arguments.

        Returns:
            A ChatResult containing the model response.
        """
        system_prompt = self.system_prompt
        user_texts: List[str] = []

        if stop:
            logger.warning(
                "stop sequences are not supported by Nova Sonic and will be ignored."
            )

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                user_texts.append(str(msg.content))

        collected_text: List[str] = []
        usage_event: Optional[Dict[str, Any]] = None

        async with self.create_session(
            system_prompt=system_prompt, **kwargs
        ) as session:
            for text in user_texts:
                await session.send_text(text)

            # Nova Sonic needs an active audio input stream to trigger
            # response generation.  Start audio, feed silence in the
            # background, and collect the text response.
            await session.start_audio_input()

            silence_task = asyncio.ensure_future(self._feed_silence(session))

            try:
                collected_text, usage_event = await self._collect_text(session)
            finally:
                silence_task.cancel()
                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass
                await session.end_audio_input()
                await session.close_input()

        response_text = "".join(collected_text)

        ai_kwargs: Dict[str, Any] = {}
        if usage_event:
            ai_kwargs["usage_metadata"] = self._parse_usage(usage_event)
        ai_kwargs["response_metadata"] = {"model_name": self.model_id}

        ai_message = AIMessage(content=response_text, **ai_kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @staticmethod
    async def _collect_text(
        session: "NovaSonicSession",
    ) -> tuple[List[str], Optional[Dict[str, Any]]]:
        """Collect assistant text and usage from a session's event stream.

        Waits until audio output has been received and a ``content_end``
        event fires after it, ensuring the model has finished its turn.

        Returns:
            A tuple of (text_chunks, usage_event_or_None).
        """
        collected: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        got_audio = False
        async for event in session.receive_events():
            if event["type"] == "text" and event.get("role") == "ASSISTANT":
                collected.append(event["text"])
            elif event["type"] == "audio":
                got_audio = True
            elif event["type"] == "usage":
                usage = event["usage"]
            elif event["type"] == "content_end" and got_audio:
                break
        return collected, usage

    @staticmethod
    def _parse_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Nova Sonic ``usageEvent`` into LangChain usage metadata."""
        return {
            "input_tokens": usage.get("totalInputTokens", 0),
            "output_tokens": usage.get("totalOutputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        }

    @staticmethod
    async def _feed_silence(session: "NovaSonicSession") -> None:
        """Send silence audio chunks until cancelled.

        Nova Sonic requires an active audio input stream to trigger
        response generation via turn detection.  This coroutine sends
        silent PCM frames (~32 ms each) continuously until it is
        cancelled by the caller.
        """
        chunk = b"\x00\x00" * 512  # 512 samples of 16-bit silence
        try:
            while session.is_active:
                await session.send_audio_chunk(chunk)
                await asyncio.sleep(0.032)
        except (asyncio.CancelledError, Exception):
            pass

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream text responses from Nova Sonic.

        Uses the same audio-based turn-detection approach as
        ``_agenerate``: sends user text, opens an audio stream with
        silence, and yields text chunks as they arrive.

        Note:
            Only ``SystemMessage`` and ``HumanMessage`` are processed from
            input. ``AIMessage`` and other message types are ignored because
            Nova Sonic maintains conversation state within the bidirectional
            stream. Previous assistant responses do not need to be sent back
            as context.

        Args:
            messages: Input messages.
            stop: Stop sequences (not supported by Nova Sonic).
            run_manager: Callback manager.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk for each text fragment.
        """
        system_prompt = self.system_prompt
        user_texts: List[str] = []

        if stop:
            logger.warning(
                "stop sequences are not supported by Nova Sonic and will be ignored."
            )

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                user_texts.append(str(msg.content))

        async with self.create_session(
            system_prompt=system_prompt, **kwargs
        ) as session:
            for text in user_texts:
                await session.send_text(text)

            await session.start_audio_input()
            silence_task = asyncio.ensure_future(self._feed_silence(session))

            got_audio = False
            usage_event: Optional[Dict[str, Any]] = None
            try:
                async for event in session.receive_events():
                    if event["type"] == "text" and event.get("role") == "ASSISTANT":
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=event["text"])
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                event["text"], chunk=chunk
                            )
                        yield chunk
                    elif event["type"] == "audio":
                        got_audio = True
                    elif event["type"] == "usage":
                        usage_event = event["usage"]
                    elif event["type"] == "content_end" and got_audio:
                        # Emit final chunk with usage metadata.
                        final_kwargs: Dict[str, Any] = {
                            "chunk_position": "last",
                        }
                        if usage_event:
                            final_kwargs["usage_metadata"] = self._parse_usage(
                                usage_event
                            )
                        final_kwargs["response_metadata"] = {
                            "model_name": self.model_id,
                        }
                        final = ChatGenerationChunk(
                            message=AIMessageChunk(content="", **final_kwargs)
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token("", chunk=final)
                        yield final
                        break
            finally:
                silence_task.cancel()
                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass
                await session.end_audio_input()
                await session.close_input()

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming — collects chunks from ``_astream``.

        Nova Sonic is inherently async.  This runs ``_astream`` in a
        separate thread so that sync callers (e.g. ``stream()``) get
        proper ``ChatGenerationChunk`` objects with ``chunk_position``
        signaling.
        """
        import concurrent.futures
        import queue as _queue

        q: _queue.Queue[Optional[ChatGenerationChunk]] = _queue.Queue()

        async def _pump() -> None:
            async for chunk in self._astream(
                messages, stop=stop, run_manager=None, **kwargs
            ):
                q.put(chunk)
            q.put(None)  # sentinel

        def _run() -> None:
            asyncio.run(_pump())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            while True:
                chunk = q.get()
                if chunk is None:
                    break
                yield chunk
            fut.result()  # propagate exceptions
