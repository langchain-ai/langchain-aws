"""Package version and user agent string."""

from importlib.metadata import version

try:
    __version__ = version("langgraph-checkpoint-aws")
except Exception:
    __version__ = "1.0.0"

SDK_USER_AGENT = f"LangGraphCheckpointAWS#{__version__}"
