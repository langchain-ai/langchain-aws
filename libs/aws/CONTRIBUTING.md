# Contributing to langchain-aws

This markdown file contains some design decisions for this package that could be helpful as you begin contributing.

## Language Models

All language models in this package inherit from `langchain_core.language_models.BaseLanguageModel`
via the `langchain_aws._base.BaseBedrock` class.

This package contains two types of language models:

- Chat models (inherit from `langchain_aws.chat_models._base.BaseChatBedrock`)
- LLMs (inherit from `langchain_aws.llms._base.BaseBedrockLLM`)

And under each of these types, there is an implementation per model provider that
bedrock supports. This is because each of Bedrock's hosted models has quirks in how
the input is formatted.