"""Integration tests for the native async Bedrock transport.

These hit real Bedrock. They use the cheapest widely available models with
one-token prompts, so a full run costs a fraction of a cent.
"""

import json
from typing import Any, Dict, List

import boto3
import pytest
from botocore.exceptions import ClientError

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_aws.async_client import BedrockAsyncClient

REGION = "us-east-1"
CHAT_MODEL = "us.amazon.nova-micro-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
HELLO: List[Dict[str, Any]] = [{"role": "user", "content": [{"text": "hi"}]}]
TINY = {"maxTokens": 5, "temperature": 0}


@pytest.fixture
async def client() -> Any:
    async with BedrockAsyncClient(region_name=REGION) as async_client:
        yield async_client


async def test_converse_is_accepted_by_aws(client: Any) -> None:
    """Proves the SigV4 signature and payload are accepted by the service."""
    response = await client.converse(
        modelId=CHAT_MODEL, messages=HELLO, inferenceConfig=TINY
    )

    assert response["output"]["message"]["content"][0]["text"]
    assert response["usage"]["totalTokens"] > 0
    assert response["ResponseMetadata"]["RequestId"]
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


async def test_converse_matches_boto3_response_shape(client: Any) -> None:
    sync_response = boto3.client("bedrock-runtime", region_name=REGION).converse(
        modelId=CHAT_MODEL, messages=HELLO, inferenceConfig=TINY
    )
    async_response = await client.converse(
        modelId=CHAT_MODEL, messages=HELLO, inferenceConfig=TINY
    )

    assert set(sync_response) == set(async_response)
    assert set(sync_response["output"]["message"]) == set(
        async_response["output"]["message"]
    )
    # The async client returns the raw JSON, so newer unmodeled members that
    # botocore would drop may also be present.
    assert set(sync_response["usage"]) <= set(async_response["usage"])


async def test_converse_stream_parses_real_event_frames(client: Any) -> None:
    response = await client.converse_stream(
        modelId=CHAT_MODEL,
        messages=[{"role": "user", "content": [{"text": "count to three"}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    events = [event async for event in response["stream"]]

    names = {name for event in events for name in event}
    assert {"messageStart", "messageStop"} <= names
    text = "".join(
        event["contentBlockDelta"]["delta"].get("text", "")
        for event in events
        if "contentBlockDelta" in event
    )
    assert text.strip()


async def test_invoke_model_is_accepted_by_aws(client: Any) -> None:
    response = await client.invoke_model(
        modelId=EMBED_MODEL, body=json.dumps({"inputText": "hi"})
    )

    assert len(json.loads(response["body"].read())["embedding"]) > 100


async def test_image_bytes_are_base64_encoded_for_the_wire(client: Any) -> None:
    """Raw `bytes` must be encoded the way botocore's serializer would."""
    # 1x1 red PNG
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xda"
        b"c\xfc\xcf\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x8c!\x00\x00\x00\x00"
        b"IEND\xaeB`\x82"
    )
    response = await client.converse(
        modelId="us.amazon.nova-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": png}}},
                    {"text": "one word"},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 8, "temperature": 0},
    )

    assert response["output"]["message"]["content"][0]["text"]


async def test_service_errors_become_client_error(client: Any) -> None:
    with pytest.raises(ClientError) as excinfo:
        await client.converse(
            modelId="amazon.definitely-not-a-model-v9", messages=HELLO
        )

    assert excinfo.value.response["Error"]["Code"] == "ValidationException"
    assert excinfo.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400


async def test_chat_bedrock_converse_async_round_trip() -> None:
    llm = ChatBedrockConverse(
        model=CHAT_MODEL,
        region_name=REGION,
        use_async_transport=True,
        max_tokens=8,
        temperature=0,
    )
    try:
        message = await llm.ainvoke("hi")
        assert message.text.strip()
        assert message.usage_metadata is not None
        assert message.usage_metadata["total_tokens"] > 0

        chunks = [chunk async for chunk in llm.astream("count to three")]
        assert "".join(chunk.text for chunk in chunks).strip()
        assert any(chunk.usage_metadata for chunk in chunks)
    finally:
        await llm.aclose()


async def test_chat_bedrock_converse_closes_via_context_manager() -> None:
    async with ChatBedrockConverse(
        model=CHAT_MODEL,
        region_name=REGION,
        use_async_transport=True,
        max_tokens=4,
        temperature=0,
    ) as llm:
        assert (await llm.ainvoke("hi")).text is not None

    assert llm.async_client is None


async def test_embeddings_async_matches_sync() -> None:
    embeddings = BedrockEmbeddings(
        model_id=EMBED_MODEL, region_name=REGION, use_async_transport=True
    )
    try:
        vector = await embeddings.aembed_query("hi")
        assert len(vector) > 100

        vectors = await embeddings.aembed_documents(["a", "b", "c"])
        assert len(vectors) == 3

        sync_vector = BedrockEmbeddings(
            model_id=EMBED_MODEL, region_name=REGION
        ).embed_query("hi")
        assert vector == sync_vector
    finally:
        await embeddings.aclose()
