from __future__ import annotations

import hashlib
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_aws.vectorstores.inmemorydb import InMemoryVectorStore

logger = logging.getLogger(__file__)


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: Json representing a list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    return json.dumps([generation.dict() for generation in generations])


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """
    Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This function (+ its counterpart `_loads_generations`) rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """
    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            f"Legacy 'Generation' cached blob encountered: '{generations_str}'"
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            f"Malformed/unparsable cached blob encountered: '{generations_str}'"
        )
        return None


class InMemorySemanticCache(BaseCache):
    """Cache that uses MemoryDB as a vector-store backend."""

    # TODO - implement a TTL policy in MemoryDB

    DEFAULT_SCHEMA = {
        "content_key": "prompt",
        "text": [
            {"name": "prompt"},
            {"name": "return_val"},
            {"name": "llm_string"},
        ],
    }

    def __init__(
        self, redis_url: str, embedding: Embeddings, score_threshold: float = 0.2
    ):
        """Initialize by passing in the `init` GPTCache func

        Args:
            redis_url (str): URL to connect to MemoryDB.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):

        Example:

        .. code-block:: python

            from langchain_core.globals import set_llm_cache

            from langchain_aws.cache import InMemorySemanticCache

            set_llm_cache(InMemorySemanticCache(
                redis_url="redis://localhost:6379",
                embedding=OpenAIEmbeddings()
            ))

        """
        self._cache_dict: Dict[str, InMemoryVectorStore] = {}
        self.redis_url = redis_url
        self.embedding = embedding
        self.score_threshold = score_threshold

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> InMemoryVectorStore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        try:
            self._cache_dict[index_name] = InMemoryVectorStore.from_existing_index(
                embedding=self.embedding,
                index_name=index_name,
                redis_url=self.redis_url,
                schema=cast(Dict, self.DEFAULT_SCHEMA),
            )
        except ValueError:
            inmemory = InMemoryVectorStore(
                embedding=self.embedding,
                index_name=index_name,
                redis_url=self.redis_url,
                index_schema=cast(Dict, self.DEFAULT_SCHEMA),
            )
            _embedding = self.embedding.embed_query(text="test")
            inmemory._create_index_if_not_exist(dim=len(_embedding))
            self._cache_dict[index_name] = inmemory

        return self._cache_dict[index_name]

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop_index(
                index_name=index_name, delete_documents=True, redis_url=self.redis_url
            )
            del self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            distance_threshold=0.1,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "InMemorySemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)

        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])
