import chromadb
import os
print(os.listdir("./chroma_db"))

client = chromadb.PersistentClient(path="./chroma_db")

# List all collections
collections = client.list_collections()
print("Collections:", collections)