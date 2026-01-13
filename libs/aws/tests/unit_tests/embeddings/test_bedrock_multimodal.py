"""Test multimodal embedding support for BedrockEmbeddings."""

import base64
import json
from unittest.mock import Mock, patch

import pytest

from langchain_aws.embeddings.bedrock import BedrockEmbeddings

# Minimal valid JPEG bytes (1x1 red pixel)
JPEG_BYTES = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xDB,
        0x00,
        0x43,
        0x00,
        0x08,
        0x06,
        0x06,
        0x07,
        0x06,
        0x05,
        0x08,
        0x07,
        0x07,
        0x07,
        0x09,
        0x09,
        0x08,
        0x0A,
        0x0C,
        0x14,
        0x0D,
        0x0C,
        0x0B,
        0x0B,
        0x0C,
        0x19,
        0x12,
    ]
)

# Minimal valid PNG bytes
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


class TestModelDetectionProperties:
    """Test model detection properties for multimodal support."""

    def test_is_titan_multimodal_true(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        assert embeddings._is_titan_multimodal is True

    def test_is_titan_multimodal_false_text_model(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        assert embeddings._is_titan_multimodal is False

    def test_is_marengo_true(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        assert embeddings._is_marengo is True

    def test_is_marengo_false(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
        assert embeddings._is_marengo is False

    def test_supports_image_titan(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        assert embeddings._supports_image() is True

    def test_supports_image_nova(self) -> None:
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        assert embeddings._supports_image() is True

    def test_supports_image_cohere_v3(self) -> None:
        embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3")
        assert embeddings._supports_image() is True

    def test_supports_image_cohere_v4(self) -> None:
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        assert embeddings._supports_image() is True

    def test_supports_image_marengo(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        assert embeddings._supports_image() is True

    def test_supports_image_text_model_false(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
        assert embeddings._supports_image() is False

    def test_supports_audio_nova_only(self) -> None:
        nova = BedrockEmbeddings(model_id="amazon.nova-2-multimodal-embeddings-v1:0")
        titan = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        marengo = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")

        assert nova._supports_audio() is True
        assert titan._supports_audio() is False
        assert marengo._supports_audio() is False

    def test_supports_video_nova_only(self) -> None:
        nova = BedrockEmbeddings(model_id="amazon.nova-2-multimodal-embeddings-v1:0")
        titan = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        marengo = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")

        assert nova._supports_video() is True
        assert titan._supports_video() is False
        assert marengo._supports_video() is False

    def test_supports_s3_input(self) -> None:
        nova = BedrockEmbeddings(model_id="amazon.nova-2-multimodal-embeddings-v1:0")
        marengo = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        titan = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        cohere = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")

        assert nova._supports_s3_input() is True
        assert marengo._supports_s3_input() is True
        assert titan._supports_s3_input() is False
        assert cohere._supports_s3_input() is False


class TestMediaFormatDetection:
    """Test media format detection from magic bytes."""

    def test_detect_jpeg(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        assert embeddings._detect_media_format(JPEG_BYTES) == "jpeg"

    def test_detect_png(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        assert embeddings._detect_media_format(PNG_BYTES) == "png"

    def test_detect_gif(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        gif_bytes = b"GIF89a" + b"\x00" * 20
        assert embeddings._detect_media_format(gif_bytes) == "gif"

    def test_detect_webp(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
        assert embeddings._detect_media_format(webp_bytes) == "webp"

    def test_detect_mp3_id3(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        mp3_bytes = b"ID3" + b"\x00" * 20
        assert embeddings._detect_media_format(mp3_bytes) == "mp3"

    def test_detect_wav(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 10
        assert embeddings._detect_media_format(wav_bytes) == "wav"

    def test_detect_fallback_jpeg(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        unknown_bytes = b"\x00\x01\x02\x03" * 5
        assert embeddings._detect_media_format(unknown_bytes) == "jpeg"


class TestMediaInputNormalization:
    """Test _load_media input normalization."""

    def test_load_bytes_jpeg(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        payload, fmt, source_kind = embeddings._load_media(JPEG_BYTES, "image")

        assert fmt == "jpeg"
        assert source_kind == "inline"
        assert base64.b64decode(payload) == JPEG_BYTES

    def test_load_data_uri(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        b64 = base64.b64encode(JPEG_BYTES).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"

        payload, fmt, source_kind = embeddings._load_media(data_uri, "image")

        assert fmt == "jpeg"
        assert source_kind == "inline"
        assert payload == b64

    def test_load_raw_base64(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        b64 = base64.b64encode(JPEG_BYTES).decode("utf-8")

        payload, fmt, source_kind = embeddings._load_media(b64, "image")

        assert fmt == "jpeg"
        assert source_kind == "inline"
        assert payload == b64

    def test_load_s3_uri_nova(self) -> None:
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        s3_uri = "s3://my-bucket/images/photo.jpg"

        payload, fmt, source_kind = embeddings._load_media(s3_uri, "image")

        assert payload == s3_uri
        assert fmt == "jpeg"
        assert source_kind == "s3"

    def test_load_s3_uri_marengo(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        s3_uri = "s3://my-bucket/images/photo.png"

        payload, fmt, source_kind = embeddings._load_media(s3_uri, "image")

        assert payload == s3_uri
        assert fmt == "png"
        assert source_kind == "s3"

    def test_load_s3_uri_unsupported_model_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        s3_uri = "s3://my-bucket/images/photo.jpg"

        with pytest.raises(ValueError, match="S3 URIs not supported"):
            embeddings._load_media(s3_uri, "image")

    def test_load_invalid_input_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")

        with pytest.raises(ValueError, match="Could not interpret input"):
            embeddings._load_media("not-valid-base64-!!!", "image")


class TestImageRequestBuilders:
    """Test _build_image_request for each provider."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_titan_image_request(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-image-v1", dimensions=256
        )
        embeddings.embed_image(JPEG_BYTES)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert "inputImage" in body
        assert body["embeddingConfig"]["outputEmbeddingLength"] == 256

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_image_request_inline(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        nova_response = {
            "embeddings": [{"embeddingType": "IMAGE", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0", dimensions=256
        )
        embeddings.embed_image(JPEG_BYTES)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["taskType"] == "SINGLE_EMBEDDING"
        params = body["singleEmbeddingParams"]
        assert params["embeddingPurpose"] == "GENERIC_INDEX"
        assert params["embeddingDimension"] == 256
        assert "image" in params
        assert params["image"]["format"] == "jpeg"
        assert "bytes" in params["image"]["source"]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_image_request_s3(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        nova_response = {
            "embeddings": [{"embeddingType": "IMAGE", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        embeddings.embed_image("s3://bucket/image.png")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        params = body["singleEmbeddingParams"]
        assert params["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.png"

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_cohere_v3_image_request(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": {"float": [[0.1, 0.2, 0.3]]}}')
        }

        embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3")
        embeddings.embed_image(JPEG_BYTES)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["input_type"] == "image"
        assert body["embedding_types"] == ["float"]
        assert len(body["images"]) == 1
        assert body["images"][0].startswith("data:image/jpeg;base64,")

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_cohere_v4_image_request(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": {"float": [[0.1, 0.2]]}}')
        }

        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0", dimensions=512)
        embeddings.embed_image(JPEG_BYTES)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["input_type"] == "search_document"
        assert body["output_dimension"] == 512
        assert len(body["images"]) == 1
        assert body["images"][0].startswith("data:image/jpeg;base64,")

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_marengo_image_request_inline(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"data": [{"embedding": [0.1, 0.2, 0.3]}]}')
        }

        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        embeddings.embed_image(JPEG_BYTES)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["inputType"] == "image"
        assert "base64String" in body["image"]["mediaSource"]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_marengo_image_request_s3(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"data": [{"embedding": [0.1, 0.2, 0.3]}]}')
        }

        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        embeddings.embed_image("s3://bucket/image.jpg")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["inputType"] == "image"
        assert (
            body["image"]["mediaSource"]["s3Location"]["uri"] == "s3://bucket/image.jpg"
        )


class TestAudioVideoNova:
    """Test audio/video embedding for Nova models."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_audio_request(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        nova_response = {
            "embeddings": [{"embeddingType": "AUDIO", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        mp3_bytes = b"ID3" + b"\x00" * 20
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        embeddings.embed_audio(mp3_bytes)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["taskType"] == "SINGLE_EMBEDDING"
        params = body["singleEmbeddingParams"]
        assert "audio" in params
        assert params["audio"]["format"] == "mp3"

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_video_request(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        nova_response = {
            "embeddings": [{"embeddingType": "VIDEO", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        # MP4 magic bytes
        mp4_bytes = b"\x00\x00\x00\x20ftyp" + b"\x00" * 20
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        embeddings.embed_video(mp4_bytes)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["taskType"] == "SINGLE_EMBEDDING"
        params = body["singleEmbeddingParams"]
        assert "video" in params
        assert params["video"]["format"] == "mp4"
        assert params["video"]["embeddingMode"] == "AUDIO_VIDEO_COMBINED"

    def test_audio_unsupported_model_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")

        with pytest.raises(ValueError, match="Audio embeddings not supported"):
            embeddings.embed_audio(b"audio data")

    def test_video_unsupported_model_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")

        with pytest.raises(ValueError, match="Video embeddings not supported"):
            embeddings.embed_video(b"video data")

    def test_marengo_audio_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")

        with pytest.raises(ValueError, match="Audio embeddings not supported"):
            embeddings.embed_audio(b"audio data")

    def test_marengo_video_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")

        with pytest.raises(ValueError, match="Video embeddings not supported"):
            embeddings.embed_video(b"video data")


class TestResponseExtraction:
    """Test _extract_media_embedding response parsing."""

    def test_extract_titan_response(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        response = {"embedding": [0.1, 0.2, 0.3]}
        result = embeddings._extract_media_embedding(response)
        assert result == [0.1, 0.2, 0.3]

    def test_extract_nova_response(self) -> None:
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        response = {"embeddings": [{"embeddingType": "IMAGE", "embedding": [0.1, 0.2]}]}
        result = embeddings._extract_media_embedding(response)
        assert result == [0.1, 0.2]

    def test_extract_cohere_v4_response(self) -> None:
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        response = {"embeddings": {"float": [[0.1, 0.2, 0.3]]}}
        result = embeddings._extract_media_embedding(response)
        assert result == [0.1, 0.2, 0.3]

    def test_extract_marengo_response(self) -> None:
        embeddings = BedrockEmbeddings(model_id="twelvelabs.marengo-embed-3-0-v1:0")
        response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        result = embeddings._extract_media_embedding(response)
        assert result == [0.1, 0.2, 0.3]

    def test_extract_missing_embedding_raises(self) -> None:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        response = {"unexpected": "data"}

        with pytest.raises(ValueError, match="No embedding found"):
            embeddings._extract_media_embedding(response)


class TestEmbedImages:
    """Test embed_images batch method."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_images_batch(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }

        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1")
        result = embeddings.embed_images([JPEG_BYTES, PNG_BYTES])

        assert len(result) == 2
        assert mock_client.invoke_model.call_count == 2


class TestNormalization:
    """Test normalization with multimodal embeddings."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_image_normalized(self, mock_create_client: Mock) -> None:
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [3.0, 4.0]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-image-v1", normalize=True
        )
        result = embeddings.embed_image(JPEG_BYTES)

        # 3/5 = 0.6, 4/5 = 0.8 (unit vector)
        assert abs(result[0] - 0.6) < 0.001
        assert abs(result[1] - 0.8) < 0.001
