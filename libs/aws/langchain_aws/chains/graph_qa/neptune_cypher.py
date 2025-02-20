from __future__ import annotations

import re
from typing import Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_aws.graphs import BaseNeptuneGraph

from .prompts import (
    CYPHER_QA_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_PROMPT,
    NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT,
)

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def trim_query(query: str) -> str:
    """Trim the query to only include Cypher keywords."""
    keywords = (
        "CALL",
        "CREATE",
        "DELETE",
        "DETACH",
        "LIMIT",
        "MATCH",
        "MERGE",
        "OPTIONAL",
        "ORDER",
        "REMOVE",
        "RETURN",
        "SET",
        "SKIP",
        "UNWIND",
        "WITH",
        "WHERE",
        "//",
    )

    lines = query.split("\n")
    new_query = ""

    for line in lines:
        if line.strip().upper().startswith(keywords):
            new_query += line + "\n"

    return new_query


def extract_cypher(text: str) -> str:
    """Extract Cypher code from text using Regex."""
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


def use_simple_prompt(llm: BaseLanguageModel) -> bool:
    """Decides whether to use the simple prompt"""
    if llm._llm_type and "anthropic" in llm._llm_type:  # type: ignore
        return True

    # Bedrock anthropic
    if hasattr(llm, "model_id") and "anthropic" in llm.model_id:  # type: ignore
        return True

    return False


def get_prompt(llm: BaseLanguageModel) -> BasePromptTemplate:
    """Selects the final prompt"""
    if use_simple_prompt(llm):
        return NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT
    else:
        return NEPTUNE_OPENCYPHER_GENERATION_PROMPT


def create_neptune_opencypher_qa_chain(
    llm: BaseLanguageModel,
    graph: BaseNeptuneGraph,
    qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
    cypher_prompt: Optional[BasePromptTemplate] = None,
    return_intermediate_steps: bool = False,
    return_direct: bool = False,
    extra_instructions: Optional[str] = None,
    allow_dangerous_requests: bool = False,
) -> Runnable:
    """Chain for question-answering against a Neptune graph
    by generating openCypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.

    Example:
        .. code-block:: python

        chain = create_neptune_opencypher_qa_chain(
            llm=llm,
            graph=graph
        )
        response = chain.invoke({"query": "your_query_here"})
    """

    if allow_dangerous_requests is not True:
        raise ValueError(
            "In order to use this chain, you must acknowledge that it can make "
            "dangerous requests by setting `allow_dangerous_requests` to `True`. "
            "You must narrowly scope the permissions of the database connection "
            "to only include necessary permissions. Failure to do so may result "
            "in data corruption or loss or reading sensitive data if such data is "
            "present in the database. "
            "Only use this chain if you understand the risks and have taken the "
            "necessary precautions. "
            "See https://python.langchain.com/docs/security for more information."
        )

    qa_chain = qa_prompt | llm

    _cypher_prompt = cypher_prompt or get_prompt(llm)
    cypher_generation_chain = _cypher_prompt | llm

    def normalize_input(raw_input: Union[str, dict]) -> dict:
        if isinstance(raw_input, str):
            return {"query": raw_input}
        return raw_input

    def execute_graph_query(cypher_query: str) -> dict:
        return graph.query(cypher_query)

    def get_cypher_inputs(inputs: dict) -> dict:
        return {
            "question": inputs["query"],
            "schema": graph.get_schema,
            "extra_instructions": extra_instructions or "",
        }

    def get_qa_inputs(inputs: dict) -> dict:
        return {
            "question": inputs["query"],
            "context": inputs["context"],
        }

    def format_response(inputs: dict) -> dict:
        intermediate_steps = [{"query": inputs["cypher"]}]

        if return_direct:
            final_response = {"result": inputs["context"]}
        else:
            final_response = {"result": inputs["qa_result"]}
            intermediate_steps.append({"context": inputs["context"]})

        if return_intermediate_steps:
            final_response[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return final_response

    chain_result = (
        normalize_input
        | RunnablePassthrough.assign(cypher_generation_inputs=get_cypher_inputs)
        | {
            "query": lambda x: x["query"],
            "cypher": (lambda x: x["cypher_generation_inputs"])
            | cypher_generation_chain
            | (lambda x: extract_cypher(x.content))
            | trim_query,
        }
        | RunnablePassthrough.assign(context=lambda x: execute_graph_query(x["cypher"]))
        | RunnablePassthrough.assign(qa_result=(lambda x: get_qa_inputs(x)) | qa_chain)
        | format_response
    )

    return chain_result
