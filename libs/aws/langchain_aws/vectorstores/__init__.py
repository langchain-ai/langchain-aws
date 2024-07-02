import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.vectorstores import (
        VectorStore,
    )

    from langchain_aws.vectorstores.awsinmemorydb import (
        AWSInMemoryDBVectorSearch,
    )

__all__ = [
    "AWSInMemoryDBVectorSearch",
]

_module_lookup = {
    "AWSInMemoryDBVectorSearch": "langchain_aws.vectorstores.awsinmemorydb",
}

def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
