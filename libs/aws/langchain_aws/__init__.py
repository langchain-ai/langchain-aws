from typing import TYPE_CHECKING, Any

from langchain_aws.chat_models import ChatBedrock, ChatBedrockConverse
from langchain_aws.llms import BedrockLLM, SagemakerEndpoint
from langchain_aws.retrievers import (
    AmazonKendraRetriever,
    AmazonKnowledgeBasesRetriever,
    AmazonS3VectorsRetriever,
)

if TYPE_CHECKING:
    from langchain_aws.chains import (
        create_neptune_opencypher_qa_chain,
        create_neptune_sparql_qa_chain,
    )
    from langchain_aws.chat_models import ChatAnthropicBedrock, ChatBedrockNovaSonic
    from langchain_aws.document_compressors.rerank import BedrockRerank
    from langchain_aws.embeddings import BedrockEmbeddings
    from langchain_aws.graphs import NeptuneAnalyticsGraph, NeptuneGraph
    from langchain_aws.vectorstores.inmemorydb import (
        InMemorySemanticCache,
        InMemoryVectorStore,
    )
    from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors
    from langchain_aws.vectorstores.valkey import ValkeyVectorStore


def setup_logging() -> None:
    import logging
    import os

    if os.environ.get("LANGCHAIN_AWS_DEBUG", "FALSE").lower() in ["true", "1"]:
        DEFAULT_LOG_FORMAT = (
            "%(asctime)s %(levelname)s | [%(filename)s:%(lineno)s]"
            "| %(name)s - %(message)s"
        )
        log_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(log_formatter)
        logging.getLogger("langchain_aws").addHandler(log_handler)
        logging.getLogger("langchain_aws").setLevel(logging.DEBUG)


setup_logging()


try:
    import boto3
    from botocore.config import Config

    FRAMEWORK_UA = "x-client-framework:langchain-aws"
    ORIGINAL_BOTO3_CLIENT = boto3.client
    ORIGINAL_SESSION_CLIENT = boto3.session.Session.client

    def _ensure_framework_ua(cfg: Config | None) -> Config:
        """Return a Config guaranteed to contain our framework UA tag."""
        if cfg is None:
            return Config(user_agent_extra=FRAMEWORK_UA)
        existing = getattr(cfg, "user_agent_extra", "") or ""
        if FRAMEWORK_UA in existing:
            return cfg
        merged_extra = f"{existing} {FRAMEWORK_UA}".strip()
        return cfg.merge(Config(user_agent_extra=merged_extra))

    def _patched_boto3_client(*args: Any, **kwargs: Any) -> Any:
        kwargs["config"] = _ensure_framework_ua(kwargs.get("config"))
        return ORIGINAL_BOTO3_CLIENT(*args, **kwargs)

    def _patched_session_client(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["config"] = _ensure_framework_ua(kwargs.get("config"))
        return ORIGINAL_SESSION_CLIENT(self, *args, **kwargs)

    boto3.client = _patched_boto3_client
    # Monkey-patch boto3 session client method to inject framework user-agent
    # mypy complains about assigning to method, but this is intentional monkey-patching
    boto3.session.Session.client = _patched_session_client  # type: ignore[method-assign]
except Exception:
    pass


__all__ = [
    "BedrockEmbeddings",
    "BedrockLLM",
    "ChatAnthropicBedrock",
    "ChatBedrock",
    "ChatBedrockConverse",
    "ChatBedrockNovaSonic",
    "SagemakerEndpoint",
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "AmazonS3VectorsRetriever",
    "create_neptune_opencypher_qa_chain",
    "create_neptune_sparql_qa_chain",
    "NeptuneAnalyticsGraph",
    "NeptuneGraph",
    "InMemoryVectorStore",
    "InMemorySemanticCache",
    "AmazonS3Vectors",
    "BedrockRerank",
    "ValkeyVectorStore",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional and heavyweight dependencies.

    Classes that pull in large transitive dependencies (e.g. numpy via
    BedrockEmbeddings, or neptune libs via graph classes) are loaded on
    first access rather than at module import time. This keeps
    ``import langchain_aws`` fast for the common case where only chat
    models or retrievers are needed.
    """
    import importlib

    # Modules that require extra pip installs
    _optional_imports: dict[str, tuple[str, str]] = {
        "ChatAnthropicBedrock": (
            "langchain_aws.chat_models",
            "pip install langchain-aws[anthropic]",
        ),
        "ChatBedrockNovaSonic": (
            "langchain_aws.chat_models",
            'pip install "langchain-aws[nova-sonic]"',
        ),
    }

    # Modules deferred to avoid importing heavyweight transitive deps
    # (e.g. numpy, neptune connector) at package import time
    _deferred_imports: dict[str, str] = {
        "BedrockEmbeddings": "langchain_aws.embeddings",
        "BedrockRerank": "langchain_aws.document_compressors.rerank",
        "InMemorySemanticCache": "langchain_aws.vectorstores.inmemorydb",
        "InMemoryVectorStore": "langchain_aws.vectorstores.inmemorydb",
        "AmazonS3Vectors": "langchain_aws.vectorstores.s3_vectors",
        "ValkeyVectorStore": "langchain_aws.vectorstores.valkey",
        "NeptuneAnalyticsGraph": "langchain_aws.graphs",
        "NeptuneGraph": "langchain_aws.graphs",
        "create_neptune_opencypher_qa_chain": "langchain_aws.chains",
        "create_neptune_sparql_qa_chain": "langchain_aws.chains",
    }

    if name in _optional_imports:
        module_path, install_hint = _optional_imports[name]
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, name)
        except ImportError as e:
            msg = f"Cannot import {name}. Please install it with `{install_hint}`."
            raise ImportError(msg) from e

    if name in _deferred_imports:
        mod = importlib.import_module(_deferred_imports[name])
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
