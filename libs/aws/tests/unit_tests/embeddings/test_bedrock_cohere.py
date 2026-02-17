"""Test Cohere v4 embedding fixes."""

from unittest.mock import Mock, patch

import pytest

from langchain_aws.embeddings.bedrock import (
    BedrockEmbeddings,
    _batch_cohere_embedding_texts,
)


class TestCohereV4Fixes:
    """Test fixes for Cohere v4 embedding support."""

    def test_is_cohere_v4_property_v4_model(self) -> None:
        """Test that _is_cohere_v4 returns True for v4 models."""
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        assert embeddings._is_cohere_v4 is True

    def test_is_cohere_v4_property_v3_model(self) -> None:
        """Test that _is_cohere_v4 returns False for v3 models."""
        embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3")
        assert embeddings._is_cohere_v4 is False

    def test_is_cohere_v4_property_non_cohere_model(self) -> None:
        """Test that _is_cohere_v4 returns False for non-Cohere models."""
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
        assert embeddings._is_cohere_v4 is False

    def test_batch_cohere_v3_limits(self) -> None:
        """Test that v3 batching respects 2048 character limit."""
        # Test text under limit
        short_texts = ["hello"] * 10
        batches = list(_batch_cohere_embedding_texts(short_texts, is_v4=False))
        assert len(batches) == 1
        assert len(batches[0]) == 10

        # Test text over limit
        long_text = "x" * 2049
        with pytest.raises(ValueError) as exc_info:
            list(_batch_cohere_embedding_texts([long_text], is_v4=False))
        assert "2048 characters" in str(exc_info.value)

    def test_batch_cohere_v4_limits(self) -> None:
        """Test that v4 batching respects higher character limit."""
        # Test text that would fail on v3 but pass on v4
        medium_text = "x" * 10000  # > 2048 but << 512k
        batches = list(_batch_cohere_embedding_texts([medium_text], is_v4=True))
        assert len(batches) == 1
        assert len(batches[0]) == 1

        # Test text over v4 limit
        huge_text = "x" * 600000  # > 512k chars
        with pytest.raises(ValueError) as exc_info:
            list(_batch_cohere_embedding_texts([huge_text], is_v4=True))
        assert "128K tokens" in str(exc_info.value)

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embedding_func_cohere_v3_schema(self, mock_create_client: Mock) -> None:
        """Test that _embedding_func handles v3 schema correctly."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock v3 response (direct array)
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": [[0.1, 0.2, 0.3]]}')
        }

        embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3")
        result = embeddings._embedding_func("test text")

        assert result == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embedding_func_cohere_v4_schema(self, mock_create_client: Mock) -> None:
        """Test that _embedding_func handles v4 schema correctly."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Mock v4 response (dict with "float" key)
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": {"float": [[0.1, 0.2, 0.3]]}}')
        }

        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        result = embeddings._embedding_func("test text")

        assert result == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_cohere_multi_embedding_uses_v4_batching(
        self, mock_create_client: Mock
    ) -> None:
        """Test that _cohere_multi_embedding passes v4 flag to batching function."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: '{"embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]}}'
            )
        }

        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")

        # Use a text that would fail v3 limits but pass v4 limits
        medium_texts = ["x" * 3000, "y" * 3000]

        with patch(
            "langchain_aws.embeddings.bedrock._batch_cohere_embedding_texts"
        ) as mock_batch:
            mock_batch.return_value = [medium_texts]  # Single batch

            result = embeddings._cohere_multi_embedding(medium_texts)

            # Verify the batching function was called with is_v4=True
            mock_batch.assert_called_once_with(medium_texts, is_v4=True)
            assert len(result) == 2


class TestCohereEmbeddings:
    """Test Cohere embeddings."""

    @pytest.mark.asyncio
    async def test_aembed_documents_uses_document_embedding(self) -> None:
        """Test that aembed_documents uses document embedding."""
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        texts = ["test-text-1", "test-text-2"]
        with patch.object(BedrockEmbeddings, "_invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "embeddings": {"float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
            }
            result = await embeddings.aembed_documents(texts)
            mock_invoke_model.assert_called_once()
            call_args = mock_invoke_model.call_args
            input_body = call_args.kwargs["input_body"]
            assert input_body["input_type"] == "search_document"
            assert input_body["texts"] == texts

        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
