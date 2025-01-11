import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_aws.chains.graph_qa.neptune_cypher import (
        create_neptune_opencypher_qa_chain
    )
    from langchain_aws.chains.graph_qa.neptune_sparql import (
        create_neptune_sparql_qa_chain
    )

__all__ = [
    "create_neptune_opencypher_qa_chain",
    "create_neptune_sparql_qa_chain"
]

_module_lookup = {
    "create_neptune_opencypher_qa_chain": "langchain_aws.chains.graph_qa.neptune_cypher",
    "create_neptune_sparql_qa_chain": "langchain_aws.chains.graph_qa.neptune_sparql",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")