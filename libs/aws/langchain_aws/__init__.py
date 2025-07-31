from langchain_aws.chains import (
    create_neptune_opencypher_qa_chain,
    create_neptune_sparql_qa_chain,
)
from langchain_aws.chat_models import ChatBedrock, ChatBedrockConverse
from langchain_aws.document_compressors.rerank import BedrockRerank
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.graphs import NeptuneAnalyticsGraph, NeptuneGraph
from langchain_aws.llms import BedrockLLM, SagemakerEndpoint
from langchain_aws.retrievers import (
    AmazonKendraRetriever,
    AmazonKnowledgeBasesRetriever,
)
from langchain_aws.vectorstores.inmemorydb import (
    InMemorySemanticCache,
    InMemoryVectorStore,
)


def setup_logging():
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

    def _patched_boto3_client(*args, **kwargs):
        kwargs["config"] = _ensure_framework_ua(kwargs.get("config"))
        return ORIGINAL_BOTO3_CLIENT(*args, **kwargs)

    def _patched_session_client(self, *args, **kwargs):
        kwargs["config"] = _ensure_framework_ua(kwargs.get("config"))
        return ORIGINAL_SESSION_CLIENT(self, *args, **kwargs)

    boto3.client = _patched_boto3_client
    boto3.session.Session.client = _patched_session_client
except Exception:
    pass


__all__ = [
    "BedrockEmbeddings",
    "BedrockLLM",
    "ChatBedrock",
    "ChatBedrockConverse",
    "SagemakerEndpoint",
    "AmazonKendraRetriever",
    "AmazonKnowledgeBasesRetriever",
    "create_neptune_opencypher_qa_chain",
    "create_neptune_sparql_qa_chain",
    "NeptuneAnalyticsGraph",
    "NeptuneGraph",
    "InMemoryVectorStore",
    "InMemorySemanticCache",
    "BedrockRerank",
]
