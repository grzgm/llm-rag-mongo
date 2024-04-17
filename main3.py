import config
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from langchain_core.pydantic_v1 import Field
MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document


client = MongoClient(config.mongo_uri)
dbName = config.db_name
collectionName = config.coll_name
collection = client[dbName][collectionName]

# Define the text embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


class MovieVectorStore():
    def __init__(
        self,
        collection: Collection[MongoDBDocumentType],
        embedding_model: Embeddings,
        *,
        index_name: str = "default",
        embedding_key: str = "embedding",
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to.
            embedding_model: Text embedding model to use.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
                defaults to 'embedding'
            index_name: Name of the Atlas Search index.
                defaults to 'default'
        """
        self._collection = collection
        self._embedding_model = embedding_model
        self._index_name = index_name
        self._embedding_key = embedding_key

    def _similarity_search_with_score(
        self,
        embedded_query: List[float],
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        custom_projection: Optional[Dict] = None
    ) -> List:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            collection:  MongoDB collection to search in. 
            embedded_query: Embedded query to look up documents similar to.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
                defaults to 'embedding'
            index_name: Name of the Atlas Search index.
                defaults to 'default'
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            custom_projection: (Optional) Custom document projection returned from
                the MongoDB collection.

        Returns:
            List of documents most similar to the query and their scores.
        """
        params = {
            "index": self._index_name,
            "path": self._embedding_key,
            "queryVector": embedded_query,
            "numCandidates": k * 10,
            "limit": k,
        }
        if pre_filter:
            params["filter"] = pre_filter
        query = {"$vectorSearch": params}

        pipeline = [
            query,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        if custom_projection:
            pipeline.append(custom_projection)

        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        return self._collection.aggregate(pipeline)  # type: ignore[arg-type]

    @staticmethod
    def similarity_search_with_score_static(
        collection: Collection,
        embedded_query: List[float],
        embedding_key: str = "embedding",
        index_name: str = "default",
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        custom_projection: Optional[Dict] = None
    ) -> List:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            collection:  MongoDB collection to search in. 
            embedded_query: Embedded query to look up documents similar to.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
                defaults to 'embedding'
            index_name: Name of the Atlas Search index.
                defaults to 'default'
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            custom_projection: (Optional) Custom document projection returned from
                the MongoDB collection.

        Returns:
            List of documents most similar to the query and their scores.
        """
        params = {
            "index": index_name,
            "path": embedding_key,
            "queryVector": embedded_query,
            "numCandidates": k * 10,
            "limit": k,
        }
        if pre_filter:
            params["filter"] = pre_filter
        query = {"$vectorSearch": params}

        pipeline = [
            query,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        if custom_projection:
            pipeline.append(custom_projection)

        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        return collection.aggregate(pipeline)  # type: ignore[arg-type]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        custom_projection: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            custom_projection: (Optional) Custom document projection returned from
                the MongoDB collection.

        Returns:
            List of documents most similar to the query and their scores.
        """
        embedded_query = self._embedding_model.embed_query(query)
        docs = self._similarity_search_with_score(
            embedded_query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            custom_projection=custom_projection,
            **kwargs,
        )
        return docs


class MovieRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    movie_vectorstore: MovieVectorStore
    """VectorStore to use for retrieval."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.movie_vectorstore.similarity_search_with_score(
            query, **self.search_kwargs)
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


custom_projection = {'$project': {
    '_id': 0,
    'title': 1,
    'genres': 1,
    'plot': 1,
    'fullplot': 1,
    'year': 1,
    'score': {
        '$meta': 'vectorSearchScore'
    }}}

embedded_query = embedding_model.embed_query(
    "A man lives in his car. He is 40 years old and although he does not have a lot of free time")
output = MovieVectorStore.similarity_search_with_score_static(
    collection,
    embedded_query,
    embedding_key="embedding",
    index_name="movies_vector_index",
    custom_projection=custom_projection)
print(list(x for x in output))

movie_vectorstore = MovieVectorStore(
    collection, embedding_model, embedding_key="embedding", index_name="movies_vector_index")
retriever = MovieRetriever(movie_vectorstore=movie_vectorstore, search_kwargs={
                           "custom_projection": custom_projection})
output = retriever.get_relevant_documents(
    "A man lives in his car. He is 40 years old and although he does not have a lot of free time")
print(list(x for x in output))
