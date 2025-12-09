"""Test Nova embedding support."""

import json
from unittest.mock import Mock, patch

from langchain_aws.embeddings.bedrock import BedrockEmbeddings


class TestNovaEmbeddings:
    """Test Nova embedding support."""

    def test_is_nova_embed_property_nova_model(self) -> None:
        """Test that _is_nova_embed returns True for Nova models."""
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        assert embeddings._is_nova_embed is True

    def test_is_nova_embed_property_titan_model(self) -> None:
        """Test that _is_nova_embed returns False for Titan models."""
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        assert embeddings._is_nova_embed is False

    def test_is_nova_embed_property_cohere_model(self) -> None:
        """Test that _is_nova_embed returns False for Cohere models."""
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0")
        assert embeddings._is_nova_embed is False

    def test_is_nova_embed_property_nova_llm(self) -> None:
        """Test that _is_nova_embed returns False for Nova LLMs."""
        embeddings = BedrockEmbeddings(model_id="amazon.nova-pro-v1:0")
        assert embeddings._is_nova_embed is False

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embedding_func_nova_schema(self, mock_create_client: Mock) -> None:
        """Test that _embedding_func sends correct Nova schema."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: '{"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3]}]}'
            )
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        result = embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["taskType"] == "SINGLE_EMBEDDING"
        assert body["singleEmbeddingParams"]["embeddingPurpose"] == "GENERIC_INDEX"
        assert body["singleEmbeddingParams"]["text"]["truncationMode"] == "END"
        assert body["singleEmbeddingParams"]["text"]["value"] == "test text"
        assert result == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embedding_func_titan_schema_unchanged(
        self, mock_create_client: Mock
    ) -> None:
        """Test that _embedding_func still sends correct Titan schema."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2, 0.3]}')
        }

        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        result = embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body == {"inputText": "test text"}
        assert result == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_query_nova(self, mock_create_client: Mock) -> None:
        """Test embed_query works with Nova models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: '{"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3]}]}'
            )
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        result = embeddings.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_documents_nova(self, mock_create_client: Mock) -> None:
        """Test embed_documents works with Nova models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: '{"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3]}]}'
            )
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        result = embeddings.embed_documents(["doc1", "doc2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.1, 0.2, 0.3]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_query_nova_normalized(self, mock_create_client: Mock) -> None:
        """Test embed_query with normalization works for Nova models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(
                read=lambda: '{"embeddings": [{"embeddingType": "TEXT", "embedding": [3.0, 4.0]}]}'
            )
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0", normalize=True
        )
        result = embeddings.embed_query("test query")

        # 3/5 = 0.6, 4/5 = 0.8 (unit vector)
        assert abs(result[0] - 0.6) < 0.001
        assert abs(result[1] - 0.8) < 0.001
