from pymongo import (MongoClient, collection as Collection)

from langchain_community.embeddings import HuggingFaceEmbeddings

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

import config

client = MongoClient(config.mongo_uri)
dbName = config.db_name
collectionName = config.coll_name
collection = client[dbName][collectionName]

# Define the text embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def similarity_search_with_score(
    collection: Collection,
    embedded_query: List[float],
    embedding_key: str = "embedding",
    index_name: str = "default",
    k: int = 4,
    pre_filter: Optional[Dict] = None,
    post_filter_pipeline: Optional[List[Dict]] = None,
    custom_projection: Optional[Dict] = None,
    include_embedding: bool = False
) -> List:
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
        # Exclude the embedding key from the return payload
        if include_embedding:
            custom_projection.update({embedding_key: 0})
        pipeline.append(custom_projection)

    if post_filter_pipeline is not None:
        pipeline.extend(post_filter_pipeline)

    return collection.aggregate(pipeline)  # type: ignore[arg-type]


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
output = similarity_search_with_score(
    collection, embedded_query, embedding_key="embedding", index_name="movies_vector_index", custom_projection=custom_projection)
print(list(x for x in output))
