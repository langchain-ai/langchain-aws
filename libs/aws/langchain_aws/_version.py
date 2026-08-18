"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version

from langchain_core.language_models import BaseLanguageModel

try:
    __version__ = version("langchain-aws")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _add_langchain_aws_version(model: BaseLanguageModel) -> None:
    """Record the langchain-aws version in the model's tracing metadata.

    Appends to ``metadata['lc_versions']`` alongside the entries seeded by
    langchain-core, so every trace carries the package versions that produced it.

    Args:
        model: The model instance to tag.
    """
    model._add_version("langchain-aws", __version__)
