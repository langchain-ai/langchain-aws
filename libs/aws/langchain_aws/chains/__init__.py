import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_aws.chains.graph_qa.neptune_cypher import (
        create_neptune_opencypher_qa_chain
    )
    from langchain_aws.chains.graph_qa.neptune_sparql import (
        NeptuneSparqlQAChain
    )

__all__ = [
    "create_neptune_opencypher_qa_chain",
    "NeptuneSparqlQAChain"
]

_module_lookup = {
    "create_neptune_opencypher_qa_chain": "langchain_aws.chains.graph_qa.neptune_cypher",
    "NeptuneSparqlQAChain": "langchain_aws.chains.graph_qa.neptune_sparql",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")