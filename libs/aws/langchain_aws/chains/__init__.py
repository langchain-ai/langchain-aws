import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_aws.chains.graph_qa.neptune_cypher import (
        NeptuneOpenCypherQAChain
    )
    from langchain_aws.chains.graph_qa.neptune_sparql import (
        NeptuneSparqlQAChain
    )

__all__ = [
    "NeptuneOpenCypherQAChain",
    "NeptuneSparqlQAChain"
]

_module_lookup = {
    "NeptuneOpenCypherQAChain": "langchain_aws.chains.graph_qa.neptune_cypher",
    "NeptuneSparqlQAChain": "langchain_aws.chains.graph_qa.neptune_sparql",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")