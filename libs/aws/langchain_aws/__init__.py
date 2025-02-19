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
