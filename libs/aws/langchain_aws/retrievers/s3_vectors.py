from langchain_core.vectorstores import VectorStoreRetriever


class AmazonS3VectorsRetriever(VectorStoreRetriever):
    """AmazonS3VectorsRetriever is a retriever for Amazon S3 Vectors.

    Examples:
        ```python
        from langchain_aws.vectorstores import AmazonS3Vectors

        vector_store = AmazonS3Vectors(...)
        retriever = vector_store.as_retriever()
        ```
    """

    allowed_search_types = [
        "similarity",
    ]
