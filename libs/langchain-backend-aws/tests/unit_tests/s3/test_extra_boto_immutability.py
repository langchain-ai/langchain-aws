"""Defense-in-depth check: ``extra_boto_config`` is frozen post-init.

The SSRF allow-list in :class:`S3BackendConfig.__post_init__` walks
``extra_boto_config['proxies']`` once at construction. Without a
defensive copy, a caller holding the original dict could mutate it
afterwards (e.g. inserting a ``proxies`` entry that resolves to IMDS)
and bypass the validation. ``__post_init__`` deep-copies the field so
the validated state is the only state used by ``build_client``.
"""

from __future__ import annotations

import pytest

from langchain_backend_aws.s3._config import S3BackendConfig


class TestExtraBotoConfigImmutability:
    def test_external_mutation_does_not_leak_into_config(self) -> None:
        # The caller retains a reference to the dict they passed in.
        # Mutating it after construction must not affect the validated
        # snapshot stored on the config — otherwise a follow-up
        # ``proxies`` insertion could silently re-introduce SSRF.
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
        # second backend constructed from the same source dict, because
        # the source dict was deep-copied at construction time.
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
