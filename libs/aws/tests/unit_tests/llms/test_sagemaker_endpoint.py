from langchain_aws.llms.sagemaker_endpoint import enforce_stop_tokens


def test_enforce_stop_tokens_treats_stop_sequence_as_literal_text() -> None:
    """Stop sequences containing regex metacharacters must not be treated as regex.

    Special tokens like Llama 3's ``<|eot_id|>`` contain a literal ``|``, which is
    the regex alternation operator. If the stop sequence isn't escaped before being
    joined into the split pattern, text is truncated wherever any *character* of the
    stop sequence appears, instead of only where the exact stop sequence occurs.
    """
    text = "Here is <content> you asked for, spanning multiple sentences."
    result = enforce_stop_tokens(text, ["<|eot_id|>"])
    assert result == text


def test_enforce_stop_tokens_does_not_raise_on_regex_metacharacters() -> None:
    """A stop sequence like ``AI)`` must not be interpreted as unbalanced regex."""
    result = enforce_stop_tokens("Hello AI) world", ["AI)"])
    assert result == "Hello "


def test_enforce_stop_tokens_still_cuts_on_exact_match() -> None:
    text = "Hello<|eot_id|>World"
    result = enforce_stop_tokens(text, ["<|eot_id|>"])
    assert result == "Hello"
