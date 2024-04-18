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
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from custom_components.movie_vector_store import MovieVectorStore
from custom_components.movie_retriever import MovieRetriever
from custom_components.movie_prompt_template import PROMPT

client = MongoClient(config.mongo_uri)
db_name = config.db_name
collection_name = config.coll_name
collection = client[db_name][collection_name]

# Define the text embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

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

# embedded_query = embedding_model.embed_query(
#     "A man lives in his car. He is 40 years old and although he does not have a lot of free time")
# output = MovieVectorStore.similarity_search_with_score_static(
#     collection,
#     embedded_query,
#     embedding_key="embedding",
#     index_name="movies_vector_index",
#     custom_projection=custom_projection)
# print(list(x for x in output))

movie_vectorstore = MovieVectorStore(
    collection, embedding_model, embedding_key="embedding", index_name="movies_vector_index")
retriever = MovieRetriever(movie_vectorstore=movie_vectorstore, search_kwargs={
                           "custom_projection": custom_projection})
# output = retriever.get_relevant_documents(
#     "A man lives in his car. He is 40 years old and although he does not have a lot of free time")
# print(list(x for x in output))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        print(f"LLM Formated Prompt: \n {prompts}")

model = Ollama(model="llama2", callback_manager=CallbackManager([MyCustomHandler()]))


output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

movie_chain = setup_and_retrieval | PROMPT

response = movie_chain.invoke('Can you finish the plot of movie titled "L" based on given context?')

print(response)
