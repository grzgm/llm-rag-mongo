""" Custom MovieRetriever based on BaseRetriever
"""
from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from custom_components.movie_vector_store import MovieVectorStore


class MovieRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    movie_vectorstore: MovieVectorStore
    """VectorStore to use for retrieval."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = list(x for x in self.movie_vectorstore.similarity_search_with_score(
            query, **self.search_kwargs))
        # if self.search_type == "similarity":
        #     docs = self.vectorstore.similarity_search(
        #         query, **self.search_kwargs)
        # elif self.search_type == "similarity_score_threshold":
        #     docs_and_similarities = (
        #         self.vectorstore.similarity_search_with_relevance_scores(
        #             query, **self.search_kwargs
        #         )
        #     )
        #     docs = [doc for doc, _ in docs_and_similarities]
        # elif self.search_type == "mmr":
        #     docs = self.vectorstore.max_marginal_relevance_search(
        #         query, **self.search_kwargs
        #     )
        # else:
        #     raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
