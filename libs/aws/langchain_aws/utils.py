import re
import sys
from packaging import version
from typing import Any, List


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def check_anthropic_tokens_dependencies() -> bool:
    """Check if we have all requirements for Anthropic count_tokens() and get_tokenizer()."""
    try:
        import anthropic
        import httpx
    except ImportError:
        return False

    anthropic_version = version.parse(anthropic.__version__)
    httpx_version = version.parse(httpx.__version__)
    python_version = sys.version_info
    if (
        anthropic_version > version.parse("0.38.0") or
        httpx_version > version.parse("0.27.2") or
        python_version > (3, 12)
    ):
        return False

    return True


def _get_anthropic_client() -> Any:
    import anthropic
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
