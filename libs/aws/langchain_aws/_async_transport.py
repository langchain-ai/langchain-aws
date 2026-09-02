"""Shared wiring for the models that can issue native async Bedrock calls.

`ChatBedrockConverse`, `BedrockBase` and `BedrockEmbeddings` each expose the same
two fields (`use_async_transport` and `async_client`) and need identical setup and
teardown for them. The behavior lives here so the three copies cannot drift apart.

This is a plain mixin rather than a pydantic base class: the fields stay declared
on each model, where their documentation belongs, and only the behavior is shared.
"""

from types import TracebackType
from typing import Any, Optional, Type

from typing_extensions import Self

from langchain_aws.utils import create_bedrock_async_client, validate_async_client


class AsyncTransportMixin:
    """Setup, access and teardown for a model's async Bedrock client.

    Expects the host class to declare `use_async_transport: bool` and
    `async_client: Any` fields.
    """

    use_async_transport: bool
    async_client: Any

    def _setup_async_client(self, *, required_method: str, config: Any) -> None:
        """Validate a supplied async client, or build one when opted in.

        Args:
            required_method: The client operation this model calls, used to
                reject an unentered context manager early.
            config: The botocore config to size the built client's pool and
                retries from.
        """
        self.async_client = validate_async_client(
            self.async_client, required_method=required_method
        )
        if self.async_client is None and self.use_async_transport:
            self._reject_unsupported_api_key()
            self.async_client = create_bedrock_async_client(
                region_name=getattr(self, "region_name", None),
                credentials_profile_name=getattr(
                    self, "credentials_profile_name", None
                ),
                aws_access_key_id=getattr(self, "aws_access_key_id", None),
                aws_secret_access_key=getattr(self, "aws_secret_access_key", None),
                aws_session_token=getattr(self, "aws_session_token", None),
                endpoint_url=getattr(self, "endpoint_url", None),
                config=config,
            )

    def _reject_unsupported_api_key(self) -> None:
        """Refuse to build a transport that would ignore a Bedrock API key.

        The built client signs with SigV4 only. Silently falling back to the
        ambient credentials would authenticate as a different identity than the
        synchronous path, so this is rejected rather than ignored.

        Raises:
            ValueError: If a Bedrock API key is configured.
        """
        api_key = getattr(self, "bedrock_api_key", None)
        if api_key is None or not api_key.get_secret_value():
            return
        msg = (
            "`use_async_transport=True` does not support bearer-token "
            "authentication, and a Bedrock API key is configured. Remove the "
            "API key to sign with SigV4, or pass an `async_client` that "
            "authenticates the way you need."
        )
        raise ValueError(msg)

    def _require_async_client(self) -> Any:
        """Return the configured async client.

        Raises:
            ValueError: If no async client is configured.
        """
        if self.async_client is None:
            msg = (
                "No async client is configured. Pass `use_async_transport=True` "
                "or an explicit `async_client`."
            )
            raise ValueError(msg)
        return self.async_client

    async def aclose(self) -> None:
        """Close the async client, if this model built it.

        A client supplied via `async_client` is left alone; its owner closes it.
        """
        if self.use_async_transport and self.async_client is not None:
            close = getattr(self.async_client, "close", None)
            if close is not None:
                await close()
            self.async_client = None

    async def __aenter__(self) -> Self:
        """Enter a context that closes the async client on exit."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Close the async client, if this model built it."""
        await self.aclose()
