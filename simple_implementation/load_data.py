from pymongo import MongoClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
import key_param

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents( data, embeddings, collection=collection )