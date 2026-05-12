"""Unit tests for S3Backend construction / client build / config validation.

Covers ``S3Backend`` constructor, the async-API delegation,
``build_client`` kwarg forwarding, ``endpoint_url`` SSRF allow-list
(scheme + private/loopback/IMDS literals + IPv6 corner cases + control
characters + wildcard-DNS suffixes), ``extra_boto_config`` validation
and immutability, and ``proxies`` / ``proxies_config`` allow-list +
audit logging.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_backend_aws import S3Backend, S3BackendConfig
from langchain_backend_aws.s3._config import build_client

from ._helpers import _make_backend, _s3_object_response

# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    """Tests for S3Backend initialization."""

    def test_keyword_only_init(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(bucket="b", client=mock_client)
        assert backend._bucket == "b"
        assert backend._prefix == ""

    def test_prefix_normalization(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", prefix="foo/bar/", client=mock_client
        )
        assert backend._prefix == "foo/bar/"

    def test_prefix_normalization_no_trailing_slash(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b", prefix="foo/bar", client=mock_client
        )
        assert backend._prefix == "foo/bar/"

    def test_config_normalized_prefix_property(self) -> None:
        # ``S3BackendConfig.normalized_prefix`` is the single source of
        # truth for the slash-shape contract; ``S3Backend.__init__``
        # consumes it directly so the boundary string cannot drift
        # between callers.
        from langchain_backend_aws.s3._config import S3BackendConfig

        assert S3BackendConfig(bucket="b", prefix="").normalized_prefix == ""
        assert (
            S3BackendConfig(bucket="b", prefix="foo/bar").normalized_prefix
            == "foo/bar/"
        )
        assert (
            S3BackendConfig(bucket="b", prefix="/foo/bar/").normalized_prefix
            == "foo/bar/"
        )

    def test_empty_prefix(self) -> None:
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(bucket="b", prefix="", client=mock_client)
        assert backend._prefix == ""

    @patch("langchain_backend_aws.s3._config.boto3")
    def test_creates_client_with_credentials(self, mock_boto3: MagicMock) -> None:
        S3Backend.from_kwargs(
            bucket="b",
            region_name="us-west-2",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        mock_boto3.client.assert_called_once()
        call = mock_boto3.client.call_args
        assert call.args == ("s3",)
        assert call.kwargs["region_name"] == "us-west-2"
        assert call.kwargs["aws_access_key_id"] == "AKID"
        assert call.kwargs["aws_secret_access_key"] == "SECRET"
        assert "config" in call.kwargs

    def test_accepts_config_object(self) -> None:
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(bucket="bkt", prefix="px")
        mock_client = MagicMock()
        backend = S3Backend(config, client=mock_client)
        assert backend._bucket == "bkt"
        assert backend._prefix == "px/"

    def test_constructor_requires_config(self) -> None:
        # ``S3Backend()`` without a config raises ``TypeError`` from the
        # signature itself (no default), which is the expected Python
        # behavior for a required positional argument.
        with pytest.raises(TypeError):
            S3Backend()  # type: ignore[call-arg]

    def test_repr_does_not_leak_credentials(self) -> None:
        """Secret-bearing fields are excluded from the dataclass repr."""
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(
            bucket="bkt",
            aws_access_key_id="AKID-SHOULD-NOT-LEAK",
            aws_secret_access_key="SUPER-SECRET",
            aws_session_token="SESSION-TOKEN",
        )
        rendered = repr(config)
        assert "AKID-SHOULD-NOT-LEAK" not in rendered
        assert "SUPER-SECRET" not in rendered
        assert "SESSION-TOKEN" not in rendered

    @patch("langchain_backend_aws.s3._config.boto3")
    def test_extra_boto_config_explicit_keys_win(self, mock_boto3: MagicMock) -> None:
        """Explicit config attributes override colliding ``extra_boto_config`` keys."""
        from langchain_backend_aws import S3BackendConfig

        S3Backend(
            S3BackendConfig(
                bucket="b",
                max_retries=7,
                read_timeout=99.0,
                extra_boto_config={
                    # Should be silently filtered out — explicit fields win.
                    "retries": {"max_attempts": 1, "mode": "standard"},
                    "read_timeout": 1.0,
                    # An orthogonal option should still be forwarded.
                    "signature_version": "s3v4",
                },
            )
        )
        mock_boto3.client.assert_called_once()
        boto_config = mock_boto3.client.call_args.kwargs["config"]
        assert boto_config.retries["max_attempts"] == 7
        assert boto_config.read_timeout == 99.0
        assert boto_config.signature_version == "s3v4"

    def test_extra_boto_config_user_agent_full_override_rejected(self) -> None:
        """``user_agent`` full override is excluded from the allowlist.

        The parent ``langchain-aws`` package patches the boto3 client
        with a framework user-agent header for AWS attribution; a full
        ``user_agent`` override would silently strip that identifier.
        Only ``user_agent_extra`` is permitted.
        """
        from langchain_backend_aws import S3BackendConfig

        with pytest.raises(ValueError, match="unsupported keys"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"user_agent": "evil/1.0"},
            )

    def test_extra_boto_config_user_agent_extra_accepted(self) -> None:
        from langchain_backend_aws import S3BackendConfig

        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"user_agent_extra": "myapp/1.0"},
        )
        assert config.extra_boto_config["user_agent_extra"] == "myapp/1.0"


# ------------------------------------------------------------------
# Async API (inherited from BackendProtocol)
# ------------------------------------------------------------------


class TestAsyncAPI:
    """Tests for the async methods inherited from BackendProtocol.

    BackendProtocol provides default async implementations that delegate
    to the sync methods via ``asyncio.to_thread``. These tests verify the
    delegation works through S3Backend without override.
    """

    async def test_aread_delegates_to_read(self) -> None:
        backend, mock = _make_backend()
        mock.get_object.return_value = _s3_object_response(b"hello\nworld")

        result = await backend.aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "hello\nworld"

    async def test_awrite_delegates_to_write(self) -> None:
        backend, mock = _make_backend()
        result = await backend.awrite("/file.txt", "content")
        assert result.error is None
        assert result.path == "/file.txt"
        mock.put_object.assert_called_once()

    async def test_als_delegates_to_ls(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [], "CommonPrefixes": []}]
        mock.get_paginator.return_value = paginator

        result = await backend.als("/")
        assert result.error is None
        assert result.entries == []


# ------------------------------------------------------------------
# ``build_client`` kwarg forwarding.
# ------------------------------------------------------------------


def _capture_boto3_client() -> tuple[Any, dict[str, Any]]:
    """Patch :func:`boto3.client` and capture the kwargs forwarded to it."""
    captured: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> str:
        captured["service"] = service
        captured["kwargs"] = kwargs
        return "fake-client"

    return fake_client, captured


class TestBuildClient:
    def test_forwards_explicit_fields(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            region_name="us-east-1",
            aws_access_key_id="AKIA",
            aws_secret_access_key="secret",
            aws_session_token="token",
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            client = build_client(config)
        assert client == "fake-client"
        assert captured["service"] == "s3"
        kwargs = captured["kwargs"]
        assert kwargs["region_name"] == "us-east-1"
        assert kwargs["aws_access_key_id"] == "AKIA"
        assert kwargs["aws_secret_access_key"] == "secret"
        assert kwargs["aws_session_token"] == "token"
        assert "config" in kwargs

    def test_drops_extra_boto_keys_overlapping_explicit_fields(self) -> None:
        # ``retries`` etc. are dropped by ``build_client`` so botocore
        # never sees a duplicated kwarg. Use ``extra_boto_config`` keys
        # that pass ``_validate_extra_boto_keys`` to confirm the drop is
        # silent rather than rejected.
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "retries": {"max_attempts": 99, "mode": "standard"},
                "connect_timeout": 999.0,
                "user_agent_extra": "test/1.0",
            },
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            build_client(config)
        boto_config = captured["kwargs"]["config"]
        # Explicit field wins for retries.
        assert boto_config.retries["max_attempts"] == config.max_retries
        assert boto_config.retries["mode"] == "adaptive"
        # The allow-listed extra (not in ``_EXPLICIT_BOTO_KEYS``) is forwarded.
        assert boto_config.user_agent_extra == "test/1.0"

    def test_endpoint_url_forwarded_when_set(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="https://s3.example.com",
        )
        fake, captured = _capture_boto3_client()
        with patch("langchain_backend_aws.s3._config.boto3.client", fake):
            build_client(config)
        assert captured["kwargs"]["endpoint_url"] == "https://s3.example.com"


# ------------------------------------------------------------------
# ``endpoint_url`` SSRF allow-list: scheme + literal IP/loopback/IMDS +
# IPv6 zone-id and IPv4-mapped corner cases + metadata DNS aliases +
# control characters.
# ------------------------------------------------------------------


class TestEndpointUrlValidation:
    def test_https_endpoint_accepted(self) -> None:
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.com")
        assert config.endpoint_url == "https://s3.example.com"

    def test_http_endpoint_accepted(self) -> None:
        # Public hostname over http is accepted without opt-in. MinIO/
        # LocalStack on localhost requires ``allow_private_endpoints``;
        # see ``test_localhost_requires_opt_in`` below.
        config = S3BackendConfig(
            bucket="b", endpoint_url="http://s3.example.com:9000"
        )
        assert config.endpoint_url == "http://s3.example.com:9000"

    def test_localhost_rejected_by_default(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://localhost:9000")

    def test_localhost_accepted_with_opt_in(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://localhost:9000",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://localhost:9000"

    def test_imds_address_rejected(self) -> None:
        # 169.254.169.254 is the IMDS endpoint on EC2; mistakenly
        # pointing the client at it is the canonical SSRF case.
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b", endpoint_url="http://169.254.169.254/latest/meta-data"
            )

    def test_loopback_ip_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://127.0.0.1:9000")

    def test_rfc1918_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://10.0.0.1:9000")

    def test_ipv6_loopback_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[::1]:9000")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="file:///etc/passwd")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="ftp://example.com")

    def test_empty_scheme_rejected(self) -> None:
        # ``urlparse('localhost:9000').scheme`` is ``''`` — no implicit
        # http; reject so callers cannot omit the scheme by accident.
        with pytest.raises(ValueError, match="endpoint_url scheme"):
            S3BackendConfig(bucket="b", endpoint_url="//example.com/")

    def test_endpoint_url_none_is_fine(self) -> None:
        config = S3BackendConfig(bucket="b")
        assert config.endpoint_url is None

    # ------------------------------------------------------------------
    # Metadata-service DNS aliases. ``_is_private_host`` cannot resolve
    # DNS names safely at construction time, so the literal-name
    # allowlist is the only line of defense against pointing the client
    # at IMDS via its hostname (instead of the raw ``169.254.169.254``).
    # These tests pin that allowlist as a regression guard — adding a
    # new metadata alias to a cloud provider should also land here.
    # ------------------------------------------------------------------

    def test_gce_metadata_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b", endpoint_url="http://metadata.google.internal/"
            )

    def test_gce_metadata_short_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata/")

    def test_gce_metadata_goog_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata.goog/")

    def test_ec2_metadata_dns_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://metadata.ec2.internal/")

    def test_metadata_dns_accepted_with_opt_in(self) -> None:
        # ``allow_private_endpoints=True`` is the documented escape
        # hatch; if an operator opts in we trust the destination.
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://metadata.google.internal/",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://metadata.google.internal/"

    # ------------------------------------------------------------------
    # IPv6 corner cases. ``urlparse(...).hostname`` strips bracket
    # syntax, but zone identifiers (``%eth0``) and IPv4-mapped IPv6
    # (``::ffff:127.0.0.1``) used to slip past ``ipaddress.ip_address``
    # / ``is_loopback``. Pin the defenses so a regression surfaces.
    # ------------------------------------------------------------------

    def test_ipv6_zone_id_loopback_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[::1%25eth0]:9000")

    def test_ipv6_zone_id_link_local_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url="http://[fe80::1%25eth0]:9000")

    def test_ipv4_mapped_ipv6_loopback_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b", endpoint_url="http://[::ffff:127.0.0.1]:9000"
            )

    def test_ipv4_mapped_ipv6_imds_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                endpoint_url="http://[::ffff:169.254.169.254]/latest/meta-data",
            )

    @pytest.mark.parametrize(
        "url",
        [
            "https://s3.example.com\r\nHost: evil.example.com",
            "https://s3.example.com\nX-Injected: true",
            "https://s3.example.com\rfoo",
            "https://s3.example.com/\x00",
        ],
    )
    def test_control_characters_rejected(self, url: str) -> None:
        # Defense-in-depth: CR/LF/NUL in a URL string is the splitting
        # primitive for HTTP request smuggling and log injection.
        with pytest.raises(ValueError, match="control character"):
            S3BackendConfig(bucket="b", endpoint_url=url)

    def test_unknown_dns_name_passes(self) -> None:
        # We deliberately do not resolve arbitrary DNS names; a public
        # hostname slips through. Operators pointing at a private DNS
        # name must opt in via ``allow_private_endpoints=True``.
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.com")
        assert config.endpoint_url == "https://s3.example.com"


# ------------------------------------------------------------------
# Wildcard-DNS SSRF guard (nip.io / sslip.io / xip.io / traefik.me /
# local-ip.sh). These suffixes resolve any IPv4 literal embedded in the
# subdomain, so the literal-IP allow-list cannot see through them.
# ------------------------------------------------------------------


class TestWildcardDnsSsrf:
    @pytest.mark.parametrize(
        "host",
        [
            "127-0-0-1.nip.io",
            "169-254-169-254.nip.io",
            "192-168-1-1.sslip.io",
            "10-0-0-1.xip.io",
            "anything.traefik.me",
            "evil.local-ip.sh",
        ],
    )
    def test_wildcard_dns_endpoint_rejected(self, host: str) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(bucket="b", endpoint_url=f"http://{host}/")

    @pytest.mark.parametrize(
        "host",
        [
            "127-0-0-1.nip.io",
            "192-168-1-1.sslip.io",
        ],
    )
    def test_wildcard_dns_proxy_rejected(self, host: str) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {"http": f"http://{host}/"}},
            )

    def test_wildcard_dns_allowed_with_opt_in(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            endpoint_url="http://127-0-0-1.nip.io/",
            allow_private_endpoints=True,
        )
        assert config.endpoint_url == "http://127-0-0-1.nip.io/"

    def test_unrelated_public_domain_still_accepted(self) -> None:
        # The suffix list must not over-match — a public hostname that
        # merely contains ``.io`` or ``.me`` somewhere is unaffected.
        config = S3BackendConfig(bucket="b", endpoint_url="https://s3.example.io/")
        assert config.endpoint_url == "https://s3.example.io/"


# ------------------------------------------------------------------
# ``extra_boto_config['proxies']`` SSRF validation. Proxy URLs are
# forwarded to botocore's transport layer, so a proxy pointed at IMDS
# would re-introduce the SSRF surface that ``endpoint_url`` blocks.
# ------------------------------------------------------------------


class TestProxiesSsrf:
    def test_public_proxy_accepted(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies": {
                    "http": "http://proxy.example.com:3128",
                    "https": "https://proxy.example.com:3129",
                }
            },
        )
        assert config.extra_boto_config["proxies"]["http"].startswith("http://")

    def test_imds_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://169.254.169.254"},
                },
            )

    def test_localhost_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"https": "https://localhost:3128"},
                },
            )

    def test_rfc1918_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://10.0.0.1:3128"},
                },
            )

    def test_metadata_alias_proxy_rejected(self) -> None:
        with pytest.raises(ValueError, match="private, loopback"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "http://metadata.google.internal"},
                },
            )

    def test_non_http_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies": {"http": "file:///etc/passwd"},
                },
            )

    def test_private_proxy_allowed_with_opt_in(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            allow_private_endpoints=True,
            extra_boto_config={
                "proxies": {"http": "http://localhost:3128"},
            },
        )
        assert config.extra_boto_config["proxies"]["http"] == "http://localhost:3128"

    def test_proxies_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": "http://proxy.example.com"},
            )

    def test_proxy_value_must_be_string(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {"http": 12345}},
            )

    def test_proxy_key_must_be_string(self) -> None:
        # botocore expects scheme-string keys; a non-str key would
        # otherwise propagate to the transport layer and surface as an
        # opaque error.
        with pytest.raises(TypeError, match="keys must be str"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {1: "http://proxy.example.com"}},
            )

    def test_proxies_config_metadata_passthrough(self) -> None:
        # ``proxies_config`` carries non-URL metadata (CA bundle paths,
        # client cert, forwarding flag) and is intentionally not
        # validated against the SSRF allow-list.
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies_config": {
                    "proxy_ca_bundle": "/etc/ssl/ca.pem",
                    "proxy_use_forwarding_for_https": True,
                },
            },
        )
        assert "proxies_config" in config.extra_boto_config


# ------------------------------------------------------------------
# ``extra_boto_config['proxies_config']`` allow-list + audit logging.
# ------------------------------------------------------------------


class TestProxiesConfigKeyAllowList:
    """Unknown ``proxies_config`` keys are rejected with the allow-list."""

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported keys"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {"unknown_option": "value"},
                },
            )

    def test_known_keys_accepted(self) -> None:
        # All three documented botocore keys must pass through cleanly.
        cfg = S3BackendConfig(
            bucket="b",
            extra_boto_config={
                "proxies_config": {
                    "proxy_ca_bundle": "/etc/ssl/ca.pem",
                    "proxy_use_forwarding_for_https": True,
                    "proxy_client_cert": ("/tmp/cert.pem", "/tmp/key.pem"),
                },
            },
        )
        assert "proxies_config" in cfg.extra_boto_config


class TestProxiesConfigTupleShape:
    """Only ``proxy_client_cert`` may carry a tuple value in ``proxies_config``."""

    def test_non_client_cert_tuple_rejected(self) -> None:
        # ``proxy_ca_bundle`` is in the allow-list but only accepts
        # str/None — passing a tuple must fail closed.
        with pytest.raises(TypeError, match="does not accept a tuple"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {"proxy_ca_bundle": ("a", "b")},
                },
            )


class TestClientCertAuditLog:
    """Configured ``client_cert`` paths must be logged at WARNING."""

    def test_client_cert_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config_logger = "langchain_backend_aws.s3._config"
        with caplog.at_level(logging.WARNING, logger=config_logger):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={
                    "proxies_config": {
                        "proxy_client_cert": (
                            "/etc/ssl/cert.pem",
                            "/etc/ssl/key.pem",
                        ),
                    },
                },
            )
        records = [
            rec for rec in caplog.records if "proxy_client_cert" in rec.message
        ]
        assert records, "expected an audit warning for configured proxy_client_cert"
        # Raw filesystem paths must not appear in the audit log; only
        # SHA-256 fingerprints (12-char hex) are emitted so log
        # aggregators cannot recover the operator's private-key path.
        for rec in records:
            assert "/etc/ssl/cert.pem" not in rec.getMessage()
            assert "/etc/ssl/key.pem" not in rec.getMessage()
            assert "sha256:" in rec.getMessage()

    def test_no_client_cert_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config_logger = "langchain_backend_aws.s3._config"
        with caplog.at_level(logging.WARNING, logger=config_logger):
            S3BackendConfig(bucket="b")
        assert not any(
            "proxy_client_cert" in rec.message for rec in caplog.records
        )


# ------------------------------------------------------------------
# ``extra_boto_config`` validation: deepcopy normalization, s3 shape,
# and explicit-key drop warning.
# ------------------------------------------------------------------


class TestDeepcopyFailureNormalization:
    """A non-copyable ``extra_boto_config`` value surfaces as ``ValueError``."""

    def test_lock_value_surfaces_value_error(self) -> None:
        # ``threading.Lock`` rejects ``deepcopy`` with a ``TypeError``.
        # The dataclass must normalize that into ``ValueError`` so callers
        # see a consistent diagnostic shape across all ``_validate_*``
        # checks.
        with pytest.raises(ValueError, match="non-copyable"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"_lock": threading.Lock()},
            )


class TestExtraBotoConfigS3Shape:
    """``extra_boto_config['s3']`` must be a dict."""

    def test_non_dict_s3_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"extra_boto_config\['s3'\]"):
            S3BackendConfig(bucket="b", extra_boto_config={"s3": "virtual"})

    def test_dict_s3_accepted(self) -> None:
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"s3": {"addressing_style": "virtual"}},
        )
        assert config.extra_boto_config["s3"] == {"addressing_style": "virtual"}


class TestExplicitKeyDroppedWarning:
    """Explicit boto keys passed via ``extra_boto_config`` warn at WARNING."""

    def test_dropped_key_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config_logger = "langchain_backend_aws.s3._config"
        config = S3BackendConfig(
            bucket="b",
            extra_boto_config={"retries": {"max_attempts": 99}},
        )
        with (
            caplog.at_level(logging.WARNING, logger=config_logger),
            patch(
                "langchain_backend_aws.s3._config.boto3.client",
                return_value=MagicMock(),
            ),
        ):
            build_client(config)
        assert any(
            "retries" in rec.getMessage() and rec.levelname == "WARNING"
            for rec in caplog.records
        )


# ------------------------------------------------------------------
# Defense-in-depth: ``extra_boto_config`` is frozen post-init. Without a
# defensive copy a caller holding the original dict could mutate it
# afterwards (e.g. inserting a ``proxies`` entry that resolves to IMDS)
# and bypass the validation.
# ------------------------------------------------------------------


class TestExtraBotoConfigImmutability:
    def test_external_mutation_does_not_leak_into_config(self) -> None:
        original: dict[str, object] = {"signature_version": "s3v4"}
        config = S3BackendConfig(bucket="b", extra_boto_config=original)
        original["proxies"] = {"http": "http://169.254.169.254"}
        assert "proxies" not in config.extra_boto_config

    def test_nested_dict_mutation_does_not_leak(self) -> None:
        # ``proxies`` is itself a dict; aliasing must be broken at every
        # depth so a caller cannot mutate the inner mapping after
        # validation either.
        nested = {"http": "http://proxy.example.com"}
        original: dict[str, object] = {"proxies": nested}
        config = S3BackendConfig(bucket="b", extra_boto_config=original)
        nested["http"] = "http://169.254.169.254"
        assert config.extra_boto_config["proxies"] == {
            "http": "http://proxy.example.com"
        }

    def test_post_init_proxy_mutation_is_isolated_from_caller(self) -> None:
        # Symmetric to the above: a caller that later mutates
        # ``config.extra_boto_config`` should not be able to influence a
        # second backend constructed from the same source dict.
        source: dict[str, object] = {"signature_version": "s3v4"}
        config_a = S3BackendConfig(bucket="a", extra_boto_config=source)
        config_b = S3BackendConfig(bucket="b", extra_boto_config=source)
        config_a.extra_boto_config["signature_version"] = "tampered"
        assert config_b.extra_boto_config["signature_version"] == "s3v4"

    def test_validation_still_runs_on_proxies(self) -> None:
        # Sanity: the deepcopy happens after validation, so an SSRF
        # proxy URL is still rejected at construction.
        with pytest.raises(ValueError, match="link-local|loopback|RFC1918"):
            S3BackendConfig(
                bucket="b",
                extra_boto_config={"proxies": {"http": "http://169.254.169.254"}},
            )
