import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_aws.vectorstores.documentdb import MongoDBVectorStore
    from langchain_aws.vectorstores.inmemorydb import (
        InMemorySemanticCache,
        InMemoryVectorStore,
    )
__all__ = [
  "MongoDBVectorStore",
    "InMemoryVectorStore",
    "InMemorySemanticCache",

]

_module_lookup = {

    "MongoDBVectorStore": "langchain_aws.vectorstores.documentdb",
    "InMemorySemanticCache": "langchain_aws.vectorstores.inmemorydb",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
