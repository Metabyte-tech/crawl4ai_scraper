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
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
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
        chunk_size=2000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_text(content)
    if not chunks:
        return
        
    documents = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
    
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} chunks to the vector store.")

def add_multiple_contents_to_store(items: list):
    """
    Ingests multiple pages at once for better performance.
    items is a list of dicts with {"content": str, "url": str}
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    all_documents = []
    for item in items:
        chunks = text_splitter.split_text(item["content"])
        all_documents.extend([Document(page_content=chunk, metadata={"source": item["url"]}) for chunk in chunks])
    
    if not all_documents:
        return
        
    vector_store = get_vector_store()
    # Batch add for performance
    vector_store.add_documents(all_documents)
    print(f"Batch added {len(all_documents)} chunks from {len(items)} pages to the vector store.")

def clear_vector_store():
    """
    Clears the chroma collection.
    """
    vector_store = get_vector_store()
    vector_store.delete_collection()
    print("Vector store collection cleared.")

if __name__ == "__main__":
    # Test adding content
    test_content = "This is a test document about Crawl4AI and LangChain integration."
    add_content_to_store(test_content, {"source": "test"})
