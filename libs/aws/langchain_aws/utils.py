import re
import sys
from typing import Any, List

from packaging import version


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def check_anthropic_tokens_dependencies() -> List[str]:
    """Check if we have all requirements for Anthropic count_tokens() and get_tokenizer()."""
    bad_deps = []

    python_version = sys.version_info
    if python_version > (3, 12):
        bad_deps.append(f"Python 3.12 or earlier required, found {'.'.join(map(str, python_version[:3]))})")

    bad_anthropic = None
    try:
        import anthropic
        anthropic_version = version.parse(anthropic.__version__)
        if anthropic_version > version.parse("0.38.0"):
            bad_anthropic = anthropic_version
    except ImportError:
        bad_anthropic = "none installed"

    bad_httpx = None
    try:
        import httpx
        httpx_version = version.parse(httpx.__version__)
        if httpx_version > version.parse("0.27.2"):
            bad_httpx = httpx_version
    except ImportError:
        bad_httpx = "none installed"

    if bad_anthropic:
        bad_deps.append(f"anthropic<=0.38.0 required, found {bad_anthropic}.")
    if bad_httpx:
        bad_deps.append(f"httpx<=0.27.2 required, found {bad_httpx}.")

    return bad_deps


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
