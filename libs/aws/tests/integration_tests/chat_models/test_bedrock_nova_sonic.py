"""Integration tests for ChatBedrockNovaSonic."""

import asyncio
import sys
from typing import Any, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Nova Sonic SDK requires Python >= 3.12",
)

pytest.importorskip(
    "aws_sdk_bedrock_runtime",
    reason=(
        "Nova Sonic SDK not installed. "
        'Install with: pip install "langchain-aws[nova-sonic]"'
    ),
)

from langchain_aws.chat_models.bedrock_nova_sonic import (  # noqa: E402
    ChatBedrockNovaSonic,
)


class TestBedrockNovaSonicStandard(ChatModelIntegrationTests):
    """Standard integration tests for ChatBedrockNovaSonic (Nova 2 Sonic).

    Nova Sonic is a speech-to-speech model that uses a bidirectional streaming
    API. It does not support image/PDF/audio content blocks or multi-turn
    conversation history via AIMessage.

    Nova 2 Sonic supports tool calling at the protocol level, but the
    LangChain integration has not yet implemented ``bind_tools``.
    Tool calling tests are skipped until that is added.

    The standard ``invoke``/``stream`` path sends text with
    ``interactive=True`` and feeds silence audio to trigger the model's
    turn-detection. This works on Nova 2 Sonic but the model does not
    return ``usage_metadata`` or ``response_metadata.model_name``.
    """

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockNovaSonic

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "amazon.nova-2-sonic-v1:0",
            "region_name": "us-east-1",
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {"temperature": 0.7, "max_tokens": 1024}

    @property
    def has_tool_calling(self) -> bool:
        # Nova 2 Sonic supports tool calling at the API level, but
        # bind_tools is not yet implemented in the LangChain integration.
        return False

    @property
    def has_tool_choice(self) -> bool:
        # Nova 2 Sonic supports toolChoice (auto/any/tool) in the
        # protocol, but bind_tools is not yet implemented.
        return False

    @property
    def has_structured_output(self) -> bool:
        # Structured output relies on bind_tools / with_structured_output,
        # which are not yet implemented.  Once tool calling support is
        # added, structured output should work via the tool-calling path.
        return False

    @property
    def supports_json_mode(self) -> bool:
        # Nova Sonic is a speech model; no JSON mode output option.
        return False

    @property
    def supports_image_inputs(self) -> bool:
        # Nova Sonic is a speech-to-speech model.  The bidirectional
        # streaming API only accepts audio and text content — there is
        # no image input event type in the protocol.
        return False

    @property
    def supports_image_urls(self) -> bool:
        # No image support in the Nova Sonic protocol.
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        # Anthropic-style tool_use / tool_result content blocks are not
        # part of the Nova Sonic bidirectional streaming protocol.
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        # Nova Sonic *does* support audio input — it is fundamentally a
        # speech-to-speech model.  However, it accepts audio as a real-time
        # stream of raw PCM chunks via the bidirectional streaming API
        # (session.start_audio_input → session.send_audio_chunk), not as
        # base64-encoded content blocks inside HumanMessage (the format
        # the standard test_audio_inputs test uses).
        #
        # Supporting the content-block format would require _agenerate to
        # detect {"type": "audio", ...} blocks in HumanMessage.content,
        # decode the base64 payload, convert from wav/mp3 to raw 16 kHz
        # 16-bit mono PCM, and stream the result through send_audio_chunk
        # in appropriately sized frames.
        #
        # Audio input via the streaming session API is tested in
        # TestBedrockNovaSonicTextToSpeech.
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        # No document/PDF support in the Nova Sonic protocol.
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_model_override(self) -> bool:
        # Model ID is baked into the bidirectional stream connection;
        # it cannot be overridden per-invoke.
        return False

    # Nova Sonic uses a bidirectional streaming API with audio-based
    # turn detection.  The text-only invoke/stream path now feeds
    # silence audio to trigger responses.


class TestBedrockNovaSonicTextToSpeech:
    """Integration tests for text-to-speech via Nova 2 Sonic.

    Nova Sonic requires an active audio input stream to trigger response
    generation. These tests send text context, open an audio stream with
    silence, and collect the audio/text responses. A timeout guards
    against the stream staying open indefinitely.
    """

    SILENCE_INTERVAL = 0.032  # ~32ms per chunk
    COLLECT_TIMEOUT = 30.0  # seconds to wait for a response

    def _make_model(self, **kwargs: Any) -> ChatBedrockNovaSonic:
        return ChatBedrockNovaSonic(
            model_id="amazon.nova-2-sonic-v1:0",  # type: ignore[call-arg]
            region_name="us-east-1",
            **kwargs,
        )

    @staticmethod
    async def _feed_silence(session: Any) -> None:
        """Send silence chunks until cancelled."""
        chunk = b"\x00\x00" * 512
        try:
            while session.is_active:
                await session.send_audio_chunk(chunk)
                await asyncio.sleep(0.032)
        except (asyncio.CancelledError, Exception):
            pass

    async def _run_tts(
        self,
        session: Any,
        text: str,
    ) -> tuple[list[str], list[bytes]]:
        """Send text, feed silence, and collect response events.

        Returns (text_chunks, audio_chunks).
        """
        await session.send_text(text)
        await session.start_audio_input()

        silence_task = asyncio.ensure_future(self._feed_silence(session))
        text_chunks: list[str] = []
        audio_chunks: list[bytes] = []
        try:
            deadline = asyncio.get_event_loop().time() + self.COLLECT_TIMEOUT
            async for event in session.receive_events():
                if asyncio.get_event_loop().time() > deadline:
                    break
                if event["type"] == "text" and event.get("role") == "ASSISTANT":
                    text_chunks.append(event["text"])
                elif event["type"] == "audio":
                    audio_chunks.append(event["audio"])
                elif event["type"] == "content_end" and audio_chunks:
                    # Only break after we've received audio output
                    break
        finally:
            silence_task.cancel()
            try:
                await silence_task
            except asyncio.CancelledError:
                pass
            await session.end_audio_input()
            await session.close_input()

        return text_chunks, audio_chunks

    @pytest.mark.asyncio
    async def test_text_to_speech_returns_audio(self) -> None:
        """Test that sending text produces audio output events."""
        model = self._make_model()

        async with model.create_session(
            system_prompt="You are a helpful assistant. Keep answers very short.",
        ) as session:
            _, audio_chunks = await self._run_tts(session, "Say hello.")

        assert len(audio_chunks) > 0, "Expected audio output but received none."
        total_bytes = sum(len(c) for c in audio_chunks)
        assert total_bytes > 0, "Audio output was empty."

    @pytest.mark.asyncio
    async def test_text_to_speech_returns_text_and_audio(self) -> None:
        """Test that text input produces both text and audio responses."""
        model = self._make_model()

        async with model.create_session(
            system_prompt="You are a helpful assistant. Keep answers very short.",
        ) as session:
            text_chunks, audio_chunks = await self._run_tts(
                session, "What is 2 plus 2?"
            )

        assert len(audio_chunks) > 0, "Expected audio output but received none."
        assert len(text_chunks) > 0, "Expected text output but received none."
        full_text = "".join(text_chunks)
        assert len(full_text) > 0, "Text output was empty."

    @pytest.mark.asyncio
    async def test_text_to_speech_with_custom_voice(self) -> None:
        """Test text-to-speech with a non-default voice ID."""
        model = self._make_model(voice_id="tiffany")

        async with model.create_session() as session:
            _, audio_chunks = await self._run_tts(session, "Hello there.")

        assert len(audio_chunks) > 0, "Expected audio output with tiffany voice."

    @pytest.mark.asyncio
    async def test_text_to_speech_with_system_prompt(self) -> None:
        """Test that system prompt influences audio response generation."""
        model = self._make_model()

        async with model.create_session(
            system_prompt="You are a pirate. Always respond in pirate speak.",
        ) as session:
            text_chunks, audio_chunks = await self._run_tts(
                session, "How are you today?"
            )

        assert len(audio_chunks) > 0, "Expected audio output but received none."

    @pytest.mark.asyncio
    async def test_text_to_speech_audio_is_valid_pcm(self) -> None:
        """Test that returned audio bytes are valid 16-bit PCM data."""
        model = self._make_model()

        async with model.create_session(
            system_prompt="You are a helpful assistant. Keep answers very short.",
        ) as session:
            _, audio_chunks = await self._run_tts(session, "Say yes.")

        assert len(audio_chunks) > 0
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) % 2 == 0, (
                f"Audio chunk length {len(chunk)} is not even — "
                "expected 16-bit PCM (2 bytes per sample)."
            )

    @pytest.mark.asyncio
    async def test_text_to_speech_multiple_texts(self) -> None:
        """Test sending multiple text messages before collecting audio."""
        model = self._make_model()

        async with model.create_session(
            system_prompt="You are a helpful assistant. Keep answers very short.",
        ) as session:
            await session.send_text("My name is Alice.")
            _, audio_chunks = await self._run_tts(session, "What is my name?")

        assert len(audio_chunks) > 0, "Expected audio output for multi-text input."
