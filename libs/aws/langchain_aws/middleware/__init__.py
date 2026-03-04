"""Middleware for LangChain AWS integrations."""

from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware

__all__ = [
    "BedrockPromptCachingMiddleware",
]
