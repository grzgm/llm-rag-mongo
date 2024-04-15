from sentence_transformers import SentenceTransformer
import pymongo
import config

def embed_collection():
    # define transofrmer model (from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for x in collection.find({"embedding": {"$exists": False}}, {}):
        # checking if vector already computed for this doc
        if "vector" not in x.keys():
            if "title" in x.keys():
                movieid = x["_id"]
                title = x["title"]
                print("computing vector.. title: " + title)
                text = f'Title: "{title}" \n'
                fullplot = None

                # if fullpplot field present, concat it with title
                if "fullplot" in x.keys():
                    fullplot = x["fullplot"]
                    text = text + f'Fullplot: {fullplot}'

                vector = model.encode(text).tolist()

                collection.update_one(
                    {"_id": movieid},
                    {
                        "$set": {
                            "embedding": vector,
                            "title": title,
                            "fullplot": fullplot,
                        }
                    },
                    upsert=True,
                )
                print("vector computed: " + str(x["_id"]))
        else:
            print("vector already computed")



# Clone the collection
def clone_collection(old_coll_name, new_coll_name):

    cloned_collection = db[old_coll_name].aggregate([{"$match": {}}])

    # Insert documents into the new collection
    db[new_coll_name].insert_many(cloned_collection)

    # Close the connection
    connection.close()

if __name__ == "__main__":
    mongo_uri = config.mongo_uri
    db_name = config.db_name
    collection_name = config.coll_name

    # Initialize db connection
    connection = pymongo.MongoClient(mongo_uri)
    db = connection[db_name]
    collection = db[collection_name]

    embed_collection()