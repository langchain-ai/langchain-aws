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
#: The boto-based classes already carry ``x-client-framework:langchain-aws`` in
#: their botocore ``user_agent_extra`` via the monkey-patch in ``__init__.py``;
#: this is the equivalent attribution tag for the transports botocore never sees.
#: It lands on the wire ``User-Agent``, so the Bedrock/Mantle backend services can
#: attribute usage to langchain-aws regardless of which chat class produced it.
FRAMEWORK_UA_TOKEN = "langchain-aws"


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

    A caller-supplied ``User-Agent`` is respected and left untouched; otherwise the
    SDK's base UA is used and the token appended once (idempotent).

    Args:
        default_headers: Existing headers passed to the SDK client, or ``None``.
        sdk_user_agent: The SDK client's own ``user_agent`` string to build on.

    Returns:
        A new headers dict carrying the tagged ``User-Agent``.
    """
    headers = dict(default_headers or {})
    base_ua = headers.get("User-Agent", sdk_user_agent)
    if FRAMEWORK_UA_TOKEN not in base_ua:
        headers["User-Agent"] = f"{base_ua} {FRAMEWORK_UA_TOKEN}".strip()
    else:
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
