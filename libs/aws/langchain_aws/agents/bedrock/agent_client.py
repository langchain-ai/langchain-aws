
"""
Clients for bedrock agents
"""


def bedrock_agent_runtime():
    """Return Bedrock agents runtime client.

    Returns:
        bedrock-agent-runtime client
    """
    boto3, botocore = get_imports()
    bedrock_config = botocore.client.Config(
        connect_timeout=120,
        read_timeout=120,
        retries={'max_attempts': 3}
    )
    return boto3.client(
        'bedrock-agent-runtime',
        config=bedrock_config
    )


def bedrock_agent():
    """Return Bedrock agents buildtime client.

    Returns:
        bedrock-agent client
    """
    boto3, botocore = get_imports()
    bedrock_config = botocore.client.Config(
        connect_timeout=120,
        read_timeout=120,
        retries={'max_attempts': 3}
    )
    return boto3.client(
        'bedrock-agent',
        config=bedrock_config
    )


def iam():
    """Return Bedrock agents buildtime client.

    Returns:
        iam client
    """
    boto3, _ = get_imports()
    return boto3.client(
        'iam'
    )


def sts():
    """Return Bedrock agents buildtime client.

    Returns:
        sts client
    """
    boto3, _ = get_imports()
    return boto3.client(
        'sts'
    )


def get_imports() -> []:
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is not installed. Please install it with `pip install boto3`")

    try:
        import botocore
    except ImportError:
        raise ImportError("botocore is not installed. Please install it with `pip install botocore`")
    return boto3, botocore
