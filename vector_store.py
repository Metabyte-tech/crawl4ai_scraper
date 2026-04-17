import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "./chroma_db"

print("Initializing Embeddings and Vector Store...", flush=True)
# ✅ Load once at startup
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Warmup call to initialize C++ hooks/buffers
print("Warming up embeddings model...", flush=True)
embeddings.embed_query("warmup")

vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="crawl4ai_collection"
)
print("Vector Store Initialized.", flush=True)

def clear_vector_store():
    """
    Clears the chroma collection by deleting all documents.
    """
    try:
        # Get all IDs
        collection_data = vector_store.get()
        ids = collection_data.get("ids", [])
        if ids:
            # SQLite limit is typically 999 or 32766 variables. We chunk the deletions.
            chunk_size = 500
            for i in range(0, len(ids), chunk_size):
                vector_store.delete(ids[i:i + chunk_size])
            print(f"Vector store cleared. Deleted {len(ids)} documents in chunks.", flush=True)
        else:
            print("Vector store is already empty.", flush=True)
    except Exception as e:
        print(f"Error clearing vector store: {e}", flush=True)
