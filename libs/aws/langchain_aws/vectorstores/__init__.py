import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_aws.vectorstores.inmemorydb import (
        InMemorySemanticCache,
        InMemoryVectorStore,
    )
    from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors
    from langchain_aws.vectorstores.valkey import ValkeyVectorStore

__all__ = [
    "InMemoryVectorStore",
    "InMemorySemanticCache",
    "AmazonS3Vectors",
    "ValkeyVectorStore",
]

_module_lookup = {
    "InMemoryVectorStore": "langchain_aws.vectorstores.inmemorydb",
    "InMemorySemanticCache": "langchain_aws.vectorstores.inmemorydb",
    "AmazonS3Vectors": "langchain_aws.vectorstores.s3_vectors",
    "ValkeyVectorStore": "langchain_aws.vectorstores.valkey",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
