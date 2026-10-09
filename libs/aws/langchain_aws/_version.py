"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from langchain_core.language_models import BaseLanguageModel

try:
    __version__ = version("langchain-aws")
except PackageNotFoundError:
    __version__ = "0.0.0"

#: Source marker appended to the wire ``User-Agent`` of non-boto SDK transports
#: (the ``anthropic`` and ``openai`` SDKs used by the Mantle/Bedrock chat classes).
#: It matches the token the boto-based classes already carry in their botocore
#: ``user_agent_extra`` via the monkey-patch in ``__init__.py``, so the Bedrock and
#: Mantle backend services see one consistent marker across every transport and can
#: attribute usage to langchain-aws regardless of which chat class produced it.
FRAMEWORK_UA_TOKEN = "x-client-framework:langchain-aws"


def _tag_user_agent(
    default_headers: dict[str, Any] | None, sdk_user_agent: str
) -> dict[str, Any]:
    """Return ``default_headers`` with a langchain-aws source tag on ``User-Agent``.

    The SDK's own ``User-Agent`` (e.g. ``"OpenAI/Python 3.22.1"`` or
    ``"AnthropicBedrockMantle/Python 1.9.0"``) already distinguishes the transport
    and, for the anthropic clients, the endpoint family. We only append a shared
    ``langchain-aws`` token so the call is attributable to this package -- no
    per-class tag, since the SDK's own UA prefix and the target operation already
    tell the classes apart on the wire.

    The token is appended to whatever ``User-Agent`` is in play -- the SDK's own
    base UA, or a caller-supplied one -- so attribution survives even when a caller
    sets their own ``User-Agent``. It is appended at most once (idempotent): if the
    ``User-Agent`` already carries the token as a distinct space-delimited part, it
    is left unchanged.

    Args:
        default_headers: Existing headers passed to the SDK client, or ``None``.
        sdk_user_agent: The SDK client's own ``user_agent`` string to build on.

    Returns:
        A new headers dict carrying the tagged ``User-Agent``.
    """
    headers = dict(default_headers or {})
    base_ua = headers.get("User-Agent", sdk_user_agent)
    # Exact-part match, not a substring test: a caller UA that merely contains the
    # token (e.g. ``my-x-client-framework:langchain-aws-proxy/2.0``) must still be
    # tagged.
    if not any(part == FRAMEWORK_UA_TOKEN for part in base_ua.split()):
        base_ua = f"{base_ua} {FRAMEWORK_UA_TOKEN}".strip()
    headers["User-Agent"] = base_ua
    return headers


def _add_langchain_aws_version(model: BaseLanguageModel) -> None:
    """Record the langchain-aws version in the model's tracing metadata.

    Appends to ``metadata['lc_versions']`` alongside the entries seeded by
    langchain-core, so every trace carries the package versions that produced it.

    Args:
        model: The model instance to tag.
    """
    model._add_version("langchain-aws", __version__)
