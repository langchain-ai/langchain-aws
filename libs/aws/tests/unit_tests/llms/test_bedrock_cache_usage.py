"""`ChatBedrock` must report cache tokens in `input_tokens`, like Converse.

`UsageMetadata` documents `input_tokens` as the sum of all input token types
and `total_tokens` as `input_tokens` + `output_tokens`. Bedrock reports only
the *uncached* input in `x-amzn-bedrock-input-token-count` /
`inputTokenCount`, so the cache counts have to be added in.
`ChatBedrockConverse` has done this since #1023.
"""

from langchain_aws.llms.bedrock import _get_invocation_metrics_chunk


def _usage(input_tokens: int, read: int, write: int, output_tokens: int) -> dict:
    chunk = _get_invocation_metrics_chunk(
        {
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": input_tokens,
                "outputTokenCount": output_tokens,
                "cacheReadInputTokenCount": read,
                "cacheWriteInputTokenCount": write,
            }
        }
    )
    assert chunk.generation_info is not None
    usage: dict = chunk.generation_info["usage_metadata"]
    return usage


def test_streaming_usage_includes_cache_reads() -> None:
    usage = _usage(726, 31426, 0, 114)

    assert usage["input_tokens"] == 726 + 31426
    assert usage["total_tokens"] == 726 + 31426 + 114
    assert usage["input_token_details"]["cache_read"] == 31426


def test_streaming_usage_includes_cache_writes() -> None:
    usage = _usage(726, 0, 2048, 114)

    assert usage["input_tokens"] == 726 + 2048
    assert usage["total_tokens"] == 726 + 2048 + 114
    assert usage["input_token_details"]["cache_creation"] == 2048


def test_streaming_usage_without_cache_is_unchanged() -> None:
    usage = _usage(726, 0, 0, 114)

    assert usage["input_tokens"] == 726
    assert usage["total_tokens"] == 840


def test_streaming_total_is_always_input_plus_output() -> None:
    for read, write in ((0, 0), (31426, 0), (0, 2048), (31426, 2048)):
        usage = _usage(726, read, write, 114)

        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
