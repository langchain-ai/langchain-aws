"""
Question answering over an RDF or OWL graph using SPARQL.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_aws.graphs import NeptuneRdfGraph

from .prompts import (
    NEPTUNE_SPARQL_GENERATION_PROMPT,
    NEPTUNE_SPARQL_GENERATION_TEMPLATE,
    SPARQL_QA_PROMPT,
)

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


def extract_sparql(query: str) -> str:
    """Extract SPARQL code from a text.

    Args:
        query: Text to extract SPARQL code from.

    Returns:
        SPARQL code extracted from the text.
    """
    query = query.strip()
    querytoks = query.split("```")
    if len(querytoks) == 3:
        query = querytoks[1]

        if query.startswith("sparql"):
            query = query[6:]
    elif query.startswith("<sparql>") and query.endswith("</sparql>"):
        query = query[8:-9]
    return query


def get_prompt(examples: str) -> BasePromptTemplate:
    """Selects the final prompt."""
    template_to_use = NEPTUNE_SPARQL_GENERATION_TEMPLATE
    if examples:
        template_to_use = template_to_use.replace("Examples:", "Examples: " + examples)
        return PromptTemplate(
            input_variables=["schema", "prompt"], template=template_to_use
        )
    return NEPTUNE_SPARQL_GENERATION_PROMPT


def create_neptune_sparql_qa_chain(
    llm: BaseLanguageModel,
    graph: NeptuneRdfGraph,
    qa_prompt: BasePromptTemplate = SPARQL_QA_PROMPT,
    sparql_prompt: Optional[BasePromptTemplate] = None,
    return_intermediate_steps: bool = False,
    return_direct: bool = False,
    extra_instructions: Optional[str] = None,
    allow_dangerous_requests: bool = False,
    examples: Optional[str] = None,
) -> Runnable[Any, dict]:
    """Chain for question-answering against a Neptune graph
    by generating SPARQL statements.

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

        chain = create_neptune_sparql_qa_chain(
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

    _sparql_prompt = sparql_prompt or get_prompt(examples)
    sparql_generation_chain = _sparql_prompt | llm

    def normalize_input(raw_input: Union[str, dict]) -> dict:
        if isinstance(raw_input, str):
            return {"query": raw_input}
        return raw_input

    def execute_graph_query(sparql_query: str) -> dict:
        return graph.query(sparql_query)

    def get_sparql_inputs(inputs: dict) -> dict:
        return {
            "prompt": inputs["query"],
            "schema": graph.get_schema,
            "extra_instructions": extra_instructions or "",
        }

    def get_qa_inputs(inputs: dict) -> dict:
        return {
            "prompt": inputs["query"],
            "context": inputs["context"],
        }

    def format_response(inputs: dict) -> dict:
        intermediate_steps = [{"query": inputs["sparql"]}]

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
        | RunnablePassthrough.assign(sparql_generation_inputs=get_sparql_inputs)
        | {
            "query": lambda x: x["query"],
            "sparql": (lambda x: x["sparql_generation_inputs"])
            | sparql_generation_chain
            | (lambda x: extract_sparql(x.content)),
        }
        | RunnablePassthrough.assign(context=lambda x: execute_graph_query(x["sparql"]))
        | RunnablePassthrough.assign(qa_result=(lambda x: get_qa_inputs(x)) | qa_chain)
        | format_response
    )

    return chain_result
