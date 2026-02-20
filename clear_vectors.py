import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def clear_vector_store():
    persist_directory = "./chroma_db"
    
    # 1. Using Shutil to delete the directory (Cleanest way)
    if os.path.exists(persist_directory):
        print(f"Deleting vector store directory: {persist_directory}")
        shutil.rmtree(persist_directory)
        print("Vector store directory deleted.")
    else:
        print("Vector store directory not found.")
        
    # 2. Re-initialize (optional but good for testing)
    print("Re-initializing empty vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Empty vector store ready.")

if __name__ == "__main__":
    clear_vector_store()
