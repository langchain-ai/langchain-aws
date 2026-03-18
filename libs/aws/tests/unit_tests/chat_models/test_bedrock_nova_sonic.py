"""Unit tests for ChatBedrockNovaSonic and NovaSonicSession."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the aws_sdk_bedrock_runtime before importing the module under test
mock_sdk_module = MagicMock()
mock_sdk_models = MagicMock()
mock_sdk_config = MagicMock()
mock_sdk_client = MagicMock()
mock_smithy = MagicMock()


@pytest.fixture(autouse=True)
def _mock_nova_sonic_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the Nova Sonic SDK modules so imports succeed without install."""
    monkeypatch.setitem(
        __import__("sys").modules, "aws_sdk_bedrock_runtime", mock_sdk_module
    )
    monkeypatch.setitem(
        __import__("sys").modules, "aws_sdk_bedrock_runtime.client", mock_sdk_client
    )
    monkeypatch.setitem(
        __import__("sys").modules, "aws_sdk_bedrock_runtime.models", mock_sdk_models
    )
    monkeypatch.setitem(
        __import__("sys").modules, "aws_sdk_bedrock_runtime.config", mock_sdk_config
    )
    monkeypatch.setitem(__import__("sys").modules, "smithy_aws_core", mock_smithy)
    monkeypatch.setitem(
        __import__("sys").modules,
        "smithy_aws_core.credentials_resolvers",
        mock_smithy,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "smithy_aws_core.credentials_resolvers.environment",
        mock_smithy,
    )


def _import_module() -> Any:
    """Import the module under test (after mocks are in place)."""
    import importlib

    import langchain_aws.chat_models.bedrock_nova_sonic as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# ChatBedrockNovaSonic init & config tests
# ---------------------------------------------------------------------------


class TestChatBedrockNovaSonicInit:
    """Test initialization and configuration."""

    def test_default_values(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic()
        assert model.model_id == "amazon.nova-sonic-v1:0"
        assert model.region_name is None
        assert model.voice_id == "matthew"
        assert model.max_tokens == 1024
        assert model.temperature == 0.7
        assert model.top_p == 0.9
        assert model.endpointing_sensitivity is None

    def test_custom_values(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-2-sonic-v1:0",
            region_name="us-west-2",
            voice_id="tiffany",
            max_tokens=2048,
            temperature=0.5,
            top_p=0.8,
            system_prompt="Be concise.",
        )
        assert model.model_id == "amazon.nova-2-sonic-v1:0"
        assert model.region_name == "us-west-2"
        assert model.voice_id == "tiffany"
        assert model.max_tokens == 2048
        assert model.temperature == 0.5
        assert model.top_p == 0.8
        assert model.system_prompt == "Be concise."

    def test_llm_type(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic()
        assert model._llm_type == "amazon_bedrock_nova_sonic"


# ---------------------------------------------------------------------------
# Voice validation tests
# ---------------------------------------------------------------------------


class TestVoiceValidation:
    """Test voice_id validation against model version."""

    def test_v1_voice_on_v1_model(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-sonic-v1:0",
            voice_id="matthew",
        )
        assert model.voice_id == "matthew"

    def test_all_v1_voices_accepted(self) -> None:
        mod = _import_module()
        for voice in mod._V1_VOICE_IDS:
            model = mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-sonic-v1:0",
                voice_id=voice,
            )
            assert model.voice_id == voice

    def test_v2_voice_on_v2_model(self) -> None:
        mod = _import_module()
        for voice in mod._V2_VOICE_IDS:
            model = mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-2-sonic-v1:0",
                voice_id=voice,
            )
            assert model.voice_id == voice

    def test_v1_voice_on_v2_model(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-2-sonic-v1:0",
            voice_id="amy",
        )
        assert model.voice_id == "amy"

    def test_v2_voice_on_v1_model_raises(self) -> None:
        mod = _import_module()
        with pytest.raises(ValueError, match="only available on Nova 2 Sonic"):
            mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-sonic-v1:0",
                voice_id="olivia",
            )

    def test_unknown_voice_raises(self) -> None:
        mod = _import_module()
        with pytest.raises(ValueError, match="Unknown voice_id"):
            mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-sonic-v1:0",
                voice_id="nonexistent",
            )


# ---------------------------------------------------------------------------
# Endpointing sensitivity validation tests
# ---------------------------------------------------------------------------


class TestEndpointingSensitivityValidation:
    """Test endpointing_sensitivity validation against model version."""

    def test_sensitivity_on_v2_model(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-2-sonic-v1:0",
            endpointing_sensitivity="HIGH",
        )
        assert model.endpointing_sensitivity == "HIGH"

    def test_sensitivity_on_v1_model_raises(self) -> None:
        mod = _import_module()
        with pytest.raises(ValueError, match="only supported on Nova 2 Sonic"):
            mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-sonic-v1:0",
                endpointing_sensitivity="MEDIUM",
            )

    def test_no_sensitivity_on_v1_model_ok(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-sonic-v1:0",
        )
        assert model.endpointing_sensitivity is None

    def test_all_sensitivity_values(self) -> None:
        mod = _import_module()
        for value in ("HIGH", "MEDIUM", "LOW"):
            model = mod.ChatBedrockNovaSonic(
                model_id="amazon.nova-2-sonic-v1:0",
                endpointing_sensitivity=value,
            )
            assert model.endpointing_sensitivity == value


# ---------------------------------------------------------------------------
# LangChain BaseChatModel interface tests
# ---------------------------------------------------------------------------


class TestBaseChatModelInterface:
    """Test alignment with BaseChatModel contract."""

    def test_get_ls_params(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic()
        params = model._get_ls_params()
        assert params["ls_provider"] == "amazon_bedrock"
        assert params["ls_model_name"] == "amazon.nova-sonic-v1:0"
        assert params["ls_model_type"] == "chat"
        assert params["ls_temperature"] == 0.7
        assert params["ls_max_tokens"] == 1024

    def test_get_ls_params_with_stop(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic()
        params = model._get_ls_params(stop=["END"])
        assert params["ls_stop"] == ["END"]

    def test_identifying_params(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic(
            model_id="amazon.nova-2-sonic-v1:0",
            voice_id="tiffany",
            endpointing_sensitivity="HIGH",
        )
        params = model._identifying_params
        assert params["model_id"] == "amazon.nova-2-sonic-v1:0"
        assert params["voice_id"] == "tiffany"
        assert params["max_tokens"] == 1024
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["endpointing_sensitivity"] == "HIGH"

    def test_is_lc_serializable(self) -> None:
        mod = _import_module()
        assert mod.ChatBedrockNovaSonic.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        mod = _import_module()
        assert mod.ChatBedrockNovaSonic.get_lc_namespace() == [
            "langchain_aws",
            "chat_models",
        ]

    def test_lc_secrets(self) -> None:
        mod = _import_module()
        model = mod.ChatBedrockNovaSonic()
        secrets = model.lc_secrets
        assert secrets == {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
            "bedrock_api_key": "AWS_BEARER_TOKEN_BEDROCK",
        }


# ---------------------------------------------------------------------------
# Model version detection tests
# ---------------------------------------------------------------------------


class TestModelVersionDetection:
    """Test _is_v2_model helper."""

    def test_v1_model(self) -> None:
        mod = _import_module()
        assert mod._is_v2_model("amazon.nova-sonic-v1:0") is False

    def test_v2_model(self) -> None:
        mod = _import_module()
        assert mod._is_v2_model("amazon.nova-2-sonic-v1:0") is True

    def test_arbitrary_string(self) -> None:
        mod = _import_module()
        assert mod._is_v2_model("some-other-model") is False


# ---------------------------------------------------------------------------
# NovaSonicSession tests
# ---------------------------------------------------------------------------


class TestNovaSonicSession:
    """Test the session event protocol."""

    def _make_session(self, **kwargs: Any) -> Any:
        mod = _import_module()
        defaults = {
            "client": AsyncMock(),
            "model_id": "amazon.nova-sonic-v1:0",
            "system_prompt": "You are helpful.",
            "voice_id": "matthew",
        }
        defaults.update(kwargs)
        return mod.NovaSonicSession(**defaults)

    @pytest.mark.asyncio
    async def test_start_sends_events_in_order(self) -> None:
        session = self._make_session()

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )

        await session.start()

        assert session.is_active
        # session start + prompt start + system content start + text input
        # + content end = 5 events
        assert mock_stream.input_stream.send.call_count == 5

    @pytest.mark.asyncio
    async def test_start_without_system_prompt(self) -> None:
        session = self._make_session(system_prompt=None)

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )

        await session.start()

        assert session.is_active
        # session start + prompt start + default system prompt (content start
        # + text input + content end) = 5 events
        assert mock_stream.input_stream.send.call_count == 5

    @pytest.mark.asyncio
    async def test_start_raises_if_already_active(self) -> None:
        session = self._make_session()

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )

        await session.start()

        with pytest.raises(RuntimeError, match="already active"):
            await session.start()

    @pytest.mark.asyncio
    async def test_send_text_raises_if_not_active(self) -> None:
        session = self._make_session()

        with pytest.raises(RuntimeError, match="not active"):
            await session.send_text("hello")

    @pytest.mark.asyncio
    async def test_send_text_sends_three_events(self) -> None:
        session = self._make_session()

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )
        await session.start()

        initial_count = mock_stream.input_stream.send.call_count
        await session.send_text("Hello!")

        # content start + text input + content end = 3 events
        assert mock_stream.input_stream.send.call_count - initial_count == 3

    @pytest.mark.asyncio
    async def test_start_audio_input_raises_if_not_active(self) -> None:
        session = self._make_session()

        with pytest.raises(RuntimeError, match="not active"):
            await session.start_audio_input()

    @pytest.mark.asyncio
    async def test_send_audio_chunk_noop_if_not_active(self) -> None:
        session = self._make_session()
        # Should not raise, just return
        await session.send_audio_chunk(b"\x00\x01\x02")

    @pytest.mark.asyncio
    async def test_end_session(self) -> None:
        session = self._make_session()

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )
        await session.start()

        initial_count = mock_stream.input_stream.send.call_count
        await session.end()

        assert not session.is_active
        # prompt end + session end = 2 events
        assert mock_stream.input_stream.send.call_count - initial_count == 2
        mock_stream.input_stream.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_end_noop_if_not_active(self) -> None:
        session = self._make_session()
        # Should not raise
        await session.end()

    @pytest.mark.asyncio
    async def test_endpointing_sensitivity_in_session_start(self) -> None:
        """Verify turnDetectionConfiguration is sent when sensitivity is set."""
        session = self._make_session(endpointing_sensitivity="HIGH")

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )

        await session.start()
        assert session.is_active

    @pytest.mark.asyncio
    async def test_no_endpointing_sensitivity_in_session_start(self) -> None:
        """Verify turnDetectionConfiguration is NOT sent when sensitivity is None."""
        session = self._make_session(endpointing_sensitivity=None)

        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        session._client.invoke_model_with_bidirectional_stream = AsyncMock(
            return_value=mock_stream
        )

        await session.start()
        assert session.is_active


# ---------------------------------------------------------------------------
# Import guard test
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Test that missing SDK raises a clear error."""

    def test_check_nova_sonic_deps_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_module()

        import sys

        saved = sys.modules.get("aws_sdk_bedrock_runtime")
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", None)

        with pytest.raises(ImportError, match="nova-sonic"):
            mod._check_nova_sonic_deps()

        if saved is not None:
            monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", saved)
