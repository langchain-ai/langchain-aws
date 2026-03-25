import glob
import importlib
from pathlib import Path


def test_importable_all() -> None:
    for path in glob.glob("../langchain_aws/*"):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module = importlib.import_module("langchain_aws." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


def test_lazy_imports() -> None:
    from langchain_aws import (
        AmazonS3Vectors,
        BedrockEmbeddings,
        BedrockRerank,
        InMemorySemanticCache,
        InMemoryVectorStore,
        NeptuneAnalyticsGraph,
        NeptuneGraph,
        ValkeyVectorStore,
        create_neptune_opencypher_qa_chain,
        create_neptune_sparql_qa_chain,
    )

    assert BedrockEmbeddings is not None
    assert BedrockRerank is not None
    assert InMemoryVectorStore is not None
    assert InMemorySemanticCache is not None
    assert AmazonS3Vectors is not None
    assert ValkeyVectorStore is not None
    assert NeptuneAnalyticsGraph is not None
    assert NeptuneGraph is not None
    assert create_neptune_opencypher_qa_chain is not None
    assert create_neptune_sparql_qa_chain is not None
