# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

SPARQL_QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
You are an assistant that creates well-written and human understandable answers.
The information part contains the information provided, which you can use to construct an answer.
The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make your response sound like the information is coming from an AI assistant, but don't add any information.
Information:
{context}

Question: {prompt}
Helpful Answer:"""
SPARQL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"], template=SPARQL_QA_TEMPLATE
)

NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS = """
Instructions:
Generate the query in openCypher format and follow these rules:
Do not use `NONE`, `ALL` or `ANY` predicate functions, rather use list comprehensions.
Do not use `REDUCE` function. Rather use a combination of list comprehension and the `UNWIND` clause to achieve similar results.
Do not use `FOREACH` clause. Rather use a combination of `WITH` and `UNWIND` clauses to achieve similar results.{extra_instructions}
\n"""

NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Instructions:", NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS
)

NEPTUNE_OPENCYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "extra_instructions"],
    template=NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE,
)

NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE = """
Write an openCypher query to answer the following question. Do not explain the answer. Only return the query.{extra_instructions}
Question:  "{question}". 
Here is the property graph schema: 
{schema}
\n"""

NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "extra_instructions"],
    template=NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE,
)

NEPTUNE_SPARQL_GENERATION_TEMPLATE = """
Task: Generate a SPARQL SELECT statement for querying a graph database.
For instance, to find all email addresses of John Doe, the following 
query in backticks would be suitable:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {{
    ?person foaf:name "John Doe" .
    ?person foaf:mbox ?email .
}}
```
Instructions:
Use only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Include all necessary prefixes.

Examples:

Schema:
{schema}
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than 
for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.

The question is:
{prompt}"""

NEPTUNE_SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=NEPTUNE_SPARQL_GENERATION_TEMPLATE
)
