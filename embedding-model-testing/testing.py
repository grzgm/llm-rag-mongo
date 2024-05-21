from sentence_transformers import SentenceTransformer
import time

def embed_collection():
    # define transofrmer model (from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    start_time = time.time()
    for _ in range(ITERATIONS):
        # print(_)

        message = collection[0]
        keywords = ', '.join(collection[1])
        text = f'Message: "{message}" \nKeywords: {keywords}'

        vector = model.encode(text).tolist()

    print(f"Time of execution: {time.time() - start_time}")


ITERATIONS = 1000
collection = ["Discussion about the heat and various unrelated topics", ["key", "words", "meaning"]]

embed_collection()