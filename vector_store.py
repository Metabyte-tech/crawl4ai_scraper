import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# --- Setup Multi-Process Chroma Server ---
import socket
import subprocess
import time
import sys
import chromadb

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if not is_port_in_use(8001):
    print("⚡ Starting ChromaDB HTTP server implicitly on port 8001...", flush=True)
    subprocess.Popen(
        # Notice we use chromadb.cli explicitly to avoid path resolution errors
        [sys.executable, "-m", "chromadb", "run", "--path", DB_DIR, "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Wait for the server to bind and be ready
    for _ in range(30):
        if is_port_in_use(8001):
            time.sleep(1) # Extra buffer for HTTP startup
            break
        time.sleep(0.5)

print("🔗 Connecting to ChromaDB HTTP Client...", flush=True)
client = chromadb.HttpClient(host="127.0.0.1", port=8001)

vector_store = Chroma(
    client=client,
    embedding_function=embeddings,
    collection_name="crawl4ai_collection"
)
print("Vector Store Initialized. (Client/Server Mode)", flush=True)

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
