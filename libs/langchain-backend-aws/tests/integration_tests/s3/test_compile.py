import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Placeholder test to verify the package can be imported."""
    from langchain_backend_aws import S3Backend  # noqa: F401
