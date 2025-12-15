"""Test dimensions parameter support for BedrockEmbeddings."""

import json
from unittest.mock import Mock, patch

from langchain_aws.embeddings.bedrock import BedrockEmbeddings


class TestDimensionsParameter:
    """Test dimensions parameter and provider-specific translation."""

    def test_get_dimensions_params_none(self) -> None:
        """Test that _get_dimensions_params returns empty dict when not set."""
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        assert embeddings._get_dimensions_params() == {}

    def test_get_dimensions_params_titan(self) -> None:
        """Test that Titan models use 'dimensions' key."""
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", dimensions=256
        )
        assert embeddings._get_dimensions_params() == {"dimensions": 256}

    def test_get_dimensions_params_cohere(self) -> None:
        """Test that Cohere models use 'output_dimension' key."""
        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", dimensions=512
        )
        assert embeddings._get_dimensions_params() == {"output_dimension": 512}

    def test_get_dimensions_params_cohere_v4(self) -> None:
        """Test that Cohere v4 models use 'output_dimension' key."""
        embeddings = BedrockEmbeddings(model_id="us.cohere.embed-v4:0", dimensions=1024)
        assert embeddings._get_dimensions_params() == {"output_dimension": 1024}

    def test_get_dimensions_params_nova(self) -> None:
        """Test that Nova models use 'embeddingDimension' key."""
        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0", dimensions=384
        )
        assert embeddings._get_dimensions_params() == {"embeddingDimension": 384}


class TestDimensionsInRequests:
    """Test that dimensions are correctly included in API request bodies."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_titan_request_includes_dimensions(self, mock_create_client: Mock) -> None:
        """Test that Titan requests include dimensions in body."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", dimensions=256
        )
        embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["inputText"] == "test text"
        assert body["dimensions"] == 256

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_titan_request_without_dimensions(self, mock_create_client: Mock) -> None:
        """Test that Titan requests omit dimensions when not set."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2]}')
        }

        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body == {"inputText": "test text"}
        assert "dimensions" not in body

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_cohere_request_includes_dimensions(self, mock_create_client: Mock) -> None:
        """Test that Cohere requests include output_dimension in body."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": [[0.1, 0.2]]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", dimensions=512
        )
        embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["texts"] == ["test text"]
        assert body["output_dimension"] == 512

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_cohere_multi_embedding_includes_dimensions(
        self, mock_create_client: Mock
    ) -> None:
        """Test that Cohere batch embedding includes output_dimension."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", dimensions=1024
        )
        embeddings._cohere_multi_embedding(["text1", "text2"])

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["output_dimension"] == 1024

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_request_includes_dimensions(self, mock_create_client: Mock) -> None:
        """Test that Nova requests include embeddingDimension in body."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        nova_response = {
            "embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0", dimensions=384
        )
        embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["taskType"] == "SINGLE_EMBEDDING"
        assert body["embeddingDimension"] == 384

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_nova_request_without_dimensions(self, mock_create_client: Mock) -> None:
        """Test that Nova requests omit embeddingDimension when not set."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        nova_response = {
            "embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2]}]
        }
        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: json.dumps(nova_response))
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.nova-2-multimodal-embeddings-v1:0"
        )
        embeddings._embedding_func("test text")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert "embeddingDimension" not in body


class TestDimensionsWithPublicMethods:
    """Test dimensions work correctly with embed_query and embed_documents."""

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_query_with_dimensions(self, mock_create_client: Mock) -> None:
        """Test embed_query passes dimensions to underlying call."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", dimensions=512
        )
        result = embeddings.embed_query("test query")

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["dimensions"] == 512
        assert result == [0.1, 0.2]

    @patch("langchain_aws.embeddings.bedrock.create_aws_client")
    def test_embed_documents_with_dimensions(self, mock_create_client: Mock) -> None:
        """Test embed_documents passes dimensions to underlying calls."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_client.invoke_model.return_value = {
            "body": Mock(read=lambda: '{"embedding": [0.1, 0.2]}')
        }

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", dimensions=1024
        )
        result = embeddings.embed_documents(["doc1", "doc2"])

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])

        assert body["dimensions"] == 1024
        assert len(result) == 2
