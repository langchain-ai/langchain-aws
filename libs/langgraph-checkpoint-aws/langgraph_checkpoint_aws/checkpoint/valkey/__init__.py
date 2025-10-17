"""Valkey checkpoint implementation for LangGraph checkpoint AWS."""

from typing import Any

# Store the import error for later use
_import_error: ImportError | None = None

# Conditional imports for optional dependencies
try:
    from .async_saver import AsyncValkeySaver
    from .saver import ValkeySaver

    __all__ = ["ValkeySaver", "AsyncValkeySaver"]
except ImportError as e:
    # Store the error for later use
    _import_error = e

    # If dependencies are not available, provide helpful error message
    def _missing_dependencies_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Valkey checkpoint functionality requires optional dependencies. "
            "Install them with: pip install 'langgraph-checkpoint-aws[valkey]'"
        ) from _import_error

    # Create placeholder classes that raise helpful errors
    # Use type: ignore to suppress mypy errors for this intentional pattern
    ValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]
    AsyncValkeySaver: type[Any] = _missing_dependencies_error  # type: ignore[assignment,no-redef]

    __all__ = ["ValkeySaver", "AsyncValkeySaver"]
