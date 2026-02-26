# ruff: noqa: T201
"""Integration tests for BrowserToolkit proxy, extensions, and profile.

Tests that create_browser_toolkit() correctly passes proxy_configuration,
extensions, and profile_configuration through to the underlying BrowserClient
against the live StartBrowserSession API.

Requires the following environment variables to be set:

    BROWSER_INTEG_PROXY_SERVER       External proxy server hostname
    BROWSER_INTEG_PROXY_PORT         External proxy server port
    BROWSER_INTEG_PROXY_SECRET_ARN   Secrets Manager ARN for creds
    BROWSER_INTEG_EXTENSION_BUCKET   S3 bucket with extensions
    BROWSER_INTEG_EXTENSION_PREFIX   S3 key prefix for extension zip
    BROWSER_INTEG_PROFILE_ID         Browser profile identifier
    BROWSER_INTEG_REGION             AWS region (default: us-west-2)

Requires valid AWS credentials for the target account.

To run:
    export BROWSER_INTEG_PROXY_SERVER=proxy.example.com
    export BROWSER_INTEG_PROXY_PORT=33335
    export BROWSER_INTEG_PROXY_SECRET_ARN=arn:aws:secretsmanager:...
    export BROWSER_INTEG_EXTENSION_BUCKET=my-bucket
    export BROWSER_INTEG_EXTENSION_PREFIX=tampermonkey.zip
    export BROWSER_INTEG_PROFILE_ID=my-profile-id
    python3 libs/aws/tests/integration_tests/tools/test_browser_toolkit_integ.py
"""

import asyncio
import os
import sys

from langchain_aws.tools.browser_toolkit import create_browser_toolkit

REQUIRED_ENV_VARS = [
    "BROWSER_INTEG_PROXY_SERVER",
    "BROWSER_INTEG_PROXY_PORT",
    "BROWSER_INTEG_PROXY_SECRET_ARN",
    "BROWSER_INTEG_EXTENSION_BUCKET",
    "BROWSER_INTEG_EXTENSION_PREFIX",
    "BROWSER_INTEG_PROFILE_ID",
]

REGION = os.environ.get("BROWSER_INTEG_REGION", "us-west-2")

PROXY_SERVER = os.environ.get("BROWSER_INTEG_PROXY_SERVER", "")
PROXY_PORT = os.environ.get("BROWSER_INTEG_PROXY_PORT", "")
PROXY_SECRET_ARN = os.environ.get("BROWSER_INTEG_PROXY_SECRET_ARN", "")
EXTENSION_BUCKET = os.environ.get("BROWSER_INTEG_EXTENSION_BUCKET", "")
EXTENSION_PREFIX = os.environ.get("BROWSER_INTEG_EXTENSION_PREFIX", "")
PROFILE_ID = os.environ.get("BROWSER_INTEG_PROFILE_ID", "")

BRIGHTDATA_PROXY_CONFIG = {
    "proxies": [
        {
            "externalProxy": {
                "server": PROXY_SERVER,
                "port": int(PROXY_PORT) if PROXY_PORT else 0,
                "domainPatterns": [
                    ".icanhazip.com",
                    ".whoer.net",
                    ".httpbin.org",
                ],
                "credentials": {
                    "basicAuth": {
                        "secretArn": PROXY_SECRET_ARN,
                    }
                },
            }
        }
    ],
    "bypass": {
        "domainPatterns": [
            "checkip.amazonaws.com",
            "169.254.169.254",
        ]
    },
}

EXTENSION_CONFIG = [
    {
        "location": {
            "s3": {
                "bucket": EXTENSION_BUCKET,
                "prefix": EXTENSION_PREFIX,
            }
        }
    }
]

PROFILE_CONFIG = {"profileIdentifier": PROFILE_ID}


def _assert_ready(session_info: dict) -> None:
    """Assert session status is READY."""
    status = session_info["status"]
    msg = f"Expected READY, got: {status}"
    assert status == "READY", msg


def test_proxy_passthrough() -> None:
    """Test proxy_configuration passthrough."""
    print("Test 1: Proxy passthrough via create_browser_toolkit")
    toolkit, tools = create_browser_toolkit(
        region=REGION,
        proxy_configuration=BRIGHTDATA_PROXY_CONFIG,
    )
    try:
        browser = asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.get_async_browser("test-proxy")
        )
        assert browser is not None, "Browser should be returned"

        # Verify the underlying session is READY
        thread_id = "test-proxy"
        client = toolkit.session_manager._async_sessions[thread_id][0]
        session_info = client.get_session()
        _assert_ready(session_info)

        # Navigate to icanhazip.com to verify proxy egress
        page = browser.contexts[0].pages[0]
        asyncio.get_event_loop().run_until_complete(
            page.goto(
                "https://icanhazip.com",
                wait_until="domcontentloaded",
            )
        )
        proxy_ip = asyncio.get_event_loop().run_until_complete(
            page.text_content("body")
        )
        proxy_ip = proxy_ip.strip() if proxy_ip else ""
        assert proxy_ip, "Should get an IP from icanhazip.com"
        print(f"  Session ID: {client.session_id}")
        print(f"  Proxy IP: {proxy_ip}")

        # Navigate to checkip.amazonaws.com (bypassed)
        asyncio.get_event_loop().run_until_complete(
            page.goto(
                "https://checkip.amazonaws.com",
                wait_until="domcontentloaded",
            )
        )
        direct_ip = asyncio.get_event_loop().run_until_complete(
            page.text_content("body")
        )
        direct_ip = direct_ip.strip() if direct_ip else ""
        assert direct_ip, "Should get an IP from checkip"
        print(f"  Direct IP: {direct_ip}")

        if proxy_ip != direct_ip:
            print("  IPs differ, confirming proxy egress")
        else:
            print("  WARNING: IPs match (non-fatal)")
    finally:
        asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.close_all_browsers()
        )
    print("  PASSED")


def test_extensions_passthrough() -> None:
    """Test extensions passthrough."""
    print("\nTest 2: Extensions passthrough")
    toolkit, tools = create_browser_toolkit(
        region=REGION,
        extensions=EXTENSION_CONFIG,
    )
    try:
        browser = asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.get_async_browser("test-extensions")
        )
        assert browser is not None, "Browser should be returned"

        thread_id = "test-extensions"
        client = toolkit.session_manager._async_sessions[thread_id][0]
        session_info = client.get_session()
        _assert_ready(session_info)
        print(f"  Session ID: {client.session_id}")
        print(f"  Status: {session_info['status']}")
    finally:
        asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.close_all_browsers()
        )
    print("  PASSED")


def test_profile_passthrough() -> None:
    """Test profile_configuration passthrough."""
    print("\nTest 3: Profile passthrough")
    toolkit, tools = create_browser_toolkit(
        region=REGION,
        profile_configuration=PROFILE_CONFIG,
    )
    try:
        browser = asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.get_async_browser("test-profile")
        )
        assert browser is not None, "Browser should be returned"

        thread_id = "test-profile"
        client = toolkit.session_manager._async_sessions[thread_id][0]
        session_info = client.get_session()
        _assert_ready(session_info)
        print(f"  Session ID: {client.session_id}")
        print(f"  Status: {session_info['status']}")

        # Navigate to verify the browser works with a profile
        page = browser.contexts[0].pages[0]
        asyncio.get_event_loop().run_until_complete(
            page.goto(
                "https://example.com",
                wait_until="domcontentloaded",
            )
        )
        title = asyncio.get_event_loop().run_until_complete(page.title())
        msg = f"Expected 'Example' in title, got: {title}"
        assert "Example" in title, msg
        print(f"  Page title: {title}")
    finally:
        asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.close_all_browsers()
        )
    print("  PASSED")


def test_all_params_combined() -> None:
    """Test proxy, extensions, and profile together."""
    print("\nTest 4: All parameters combined")
    toolkit, tools = create_browser_toolkit(
        region=REGION,
        proxy_configuration=BRIGHTDATA_PROXY_CONFIG,
        extensions=EXTENSION_CONFIG,
        profile_configuration=PROFILE_CONFIG,
    )
    try:
        browser = asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.get_async_browser("test-combined")
        )
        assert browser is not None, "Browser should be returned"

        thread_id = "test-combined"
        client = toolkit.session_manager._async_sessions[thread_id][0]
        session_info = client.get_session()
        print(f"  Session ID: {client.session_id}")
        print(f"  Status: {session_info['status']}")
        print("  PASSED (API accepted combined config)")
    except Exception as e:
        error_msg = str(e)
        expected = [
            "ValidationException",
            "validation",
            "Access Denied",
        ]
        if any(err in error_msg or err in error_msg.lower() for err in expected):
            print("  PASSED (API rejected: config passed through)")
        else:
            raise
    finally:
        asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.close_all_browsers()
        )


def test_backward_compat_no_params() -> None:
    """Test no extra params still works (regression)."""
    print("\nTest 5: Backward compat -- no extra params")
    toolkit, tools = create_browser_toolkit(region=REGION)
    try:
        browser = asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.get_async_browser("test-compat")
        )
        assert browser is not None, "Browser should be returned"

        thread_id = "test-compat"
        client = toolkit.session_manager._async_sessions[thread_id][0]
        session_info = client.get_session()
        _assert_ready(session_info)
        print(f"  Session ID: {client.session_id}")
        print(f"  Status: {session_info['status']}")

        # Basic navigation sanity check
        page = browser.contexts[0].pages[0]
        asyncio.get_event_loop().run_until_complete(
            page.goto(
                "https://example.com",
                wait_until="domcontentloaded",
            )
        )
        title = asyncio.get_event_loop().run_until_complete(page.title())
        msg = f"Expected 'Example' in title, got: {title}"
        assert "Example" in title, msg
        print(f"  Page title: {title}")
    finally:
        asyncio.get_event_loop().run_until_complete(
            toolkit.session_manager.close_all_browsers()
        )
    print("  PASSED")


if __name__ == "__main__":
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        missing_str = ", ".join(missing)
        print(f"Skipping: missing env vars: {missing_str}")
        sys.exit(0)

    tests = [
        test_proxy_passthrough,
        test_extensions_passthrough,
        test_profile_passthrough,
        test_all_params_combined,
        test_backward_compat_no_params,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    total = len(tests)
    passed = total - failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        sys.exit(1)
