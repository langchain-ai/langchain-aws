"""Example demonstrating 1-hour prompt caching with ChatBedrock.

This example shows how to use extended cache TTL (1 hour) with Anthropic models
through AWS Bedrock's legacy API.
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage


def main():
    # Initialize the model with beta header for extended cache TTL
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",  # Update to your region
        additional_model_request_fields={
            "anthropic_beta": ["prompt-caching-2024-07-31"]
        },
    )

    # Create cache points with different TTLs
    cache_1h = ChatBedrock.create_cache_point(ttl="1h")
    cache_5m = ChatBedrock.create_cache_point()  # Default 5-minute cache

    # Example 1: Using cache points in messages
    print("Example 1: Using cache points in message content")
    print("-" * 50)

    # Important: 1-hour cache entries must appear before 5-minute cache entries
    messages = [
        HumanMessage(
            content=[
                # Long system prompt that doesn't change often - cache for 1 hour
                """You are an expert AI assistant with extensive knowledge in multiple domains.
            You always provide helpful, accurate, and detailed responses.
            You follow these guidelines:
            1. Be concise but thorough
            2. Use examples when helpful
            3. Admit when you don't know something
            4. Provide sources when possible""",
                cache_1h,  # Cache the above content for 1 hour
                # Context that might change more frequently - cache for 5 minutes
                "Today's date is January 17, 2025. The user is located in Seattle.",
                cache_5m,  # Cache this for 5 minutes
                # The actual user question (not cached)
                "What's the weather typically like in Seattle in January?",
            ]
        )
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")

    # Check cache usage
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print(f"\nToken usage:")
        print(f"  Input tokens: {response.usage_metadata.input_tokens}")
        print(f"  Output tokens: {response.usage_metadata.output_tokens}")

        if (
            hasattr(response.usage_metadata, "input_token_details")
            and response.usage_metadata.input_token_details
        ):
            details = response.usage_metadata.input_token_details
            print(f"  Cache read tokens: {details.get('cache_read', 0)}")
            print(f"  Cache creation tokens: {details.get('cache_creation', 0)}")

    # Example 2: Using SystemMessage and HumanMessage separately
    print("\n\nExample 2: Using separate SystemMessage with caching")
    print("-" * 50)

    messages2 = [
        SystemMessage(
            content=[
                "You are a helpful coding assistant specialized in Python.",
                cache_1h,
            ]
        ),
        HumanMessage(
            content=[
                "Current Python version is 3.11",
                cache_5m,
                "How do I use type hints in Python?",
            ]
        ),
    ]

    response2 = llm.invoke(messages2)
    print(f"Response: {response2.content[:200]}...")  # Show first 200 chars

    # Example 3: Multiple invocations to see cache benefits
    print("\n\nExample 3: Second invocation (should use cached content)")
    print("-" * 50)

    # Make the same request again - should use cached content
    response3 = llm.invoke(messages)

    if hasattr(response3, "usage_metadata") and response3.usage_metadata:
        print(f"Token usage (2nd call):")
        print(f"  Input tokens: {response3.usage_metadata.input_tokens}")

        if (
            hasattr(response3.usage_metadata, "input_token_details")
            and response3.usage_metadata.input_token_details
        ):
            details = response3.usage_metadata.input_token_details
            cache_read = details.get("cache_read", 0)
            print(f"  Cache read tokens: {cache_read}")
            if cache_read > 0:
                print("  ✓ Successfully used cached content!")


if __name__ == "__main__":
    main()
