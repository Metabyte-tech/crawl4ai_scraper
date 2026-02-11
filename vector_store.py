from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

DB_DIR = "./chroma_db"

def get_vector_store():
    """
    Initializes or loads the Chroma vector store.
    """
    # Using Ollama for embeddings. Make sure ollama is running.
    embeddings = OllamaEmbeddings(model="llama3")
    
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="crawl4ai_collection"
    )

def add_content_to_store(content: str, metadata: dict = None):
    """
    Splits content into chunks and adds them to the vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_text(content)
    if not chunks:
        return
        
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} chunks to the vector store.")

if __name__ == "__main__":
    # Test adding content
    test_content = "This is a test document about Crawl4AI and LangChain integration."
    add_content_to_store(test_content, {"source": "test"})
