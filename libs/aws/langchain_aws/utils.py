import re
from typing import Any, List


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def _get_anthropic_client() -> Any:
    try:
        import anthropic
        from packaging import version

        max_supported_version = version.parse("0.38.0")
        anthropic_version = version.parse(anthropic.__version__)

        if anthropic_version > max_supported_version:
            raise NotImplementedError(
                "Currently installed anthropic version {anthropic_version} is not "
                "supported. Please use ChatAnthropic.get_num_tokens_from_messages "
                "instead."
            )

    except ImportError:
        raise ImportError(
            "Could not import anthropic python package. "
            "This is needed in order to accurately tokenize the text "
            "for anthropic models. Please install it with `pip install anthropic`."
        )
    return anthropic.Anthropic()


def get_num_tokens_anthropic(text: str) -> int:
    """Get the number of tokens in a string of text."""
    client = _get_anthropic_client()
    return client.count_tokens(text=text)


def get_token_ids_anthropic(text: str) -> List[int]:
    """Get the token ids for a string of text."""
    client = _get_anthropic_client()
    tokenizer = client.get_tokenizer()
    encoded_text = tokenizer.encode(text)
    return encoded_text.ids
