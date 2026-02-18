from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DB_DIR = "./chroma_db"

print("Initializing Embeddings and Vector Store...")
# ✅ Load once at startup
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="crawl4ai_collection"
)
print("Vector Store Initialized.")

def clear_vector_store():
    """
    Clears the chroma collection by deleting all documents.
    """
    try:
        # Get all IDs
        collection_data = vector_store.get()
        ids = collection_data.get("ids", [])
        if ids:
            vector_store.delete(ids)
            print(f"Vector store cleared. Deleted {len(ids)} documents.")
        else:
            print("Vector store is already empty.")
    except Exception as e:
        print(f"Error clearing vector store: {e}")
