from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import vector_store

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=400,  # ✅ Reduced from 2000
        chunk_overlap=50 # ✅ Reduced from 200
    )

def add_content_to_store(content: str, metadata: dict = None):
    """
    Splits content into chunks and adds them to the vector store.
    """
    text_splitter = get_text_splitter()
    
    chunks = text_splitter.split_text(content)
    if not chunks:
        return
        
    # ✅ Add metadata to Documents
    documents = [
        Document(
            page_content=chunk, 
            metadata=(metadata or {})
        ) for chunk in chunks
    ]
    
    # ✅ Batch add (though for single content it might not be huge, it's still good practice)
    vector_store.add_documents(documents, batch_size=64)
    print(f"Added {len(documents)} chunks to the vector store.")

def add_multiple_contents_to_store(items: list):
    """
    Ingests multiple pages at once for better performance.
    items is a list of dicts with {"content": str, "url": str, "metadata": dict (optional)}
    """
    text_splitter = get_text_splitter()
    
    all_documents = []
    for item in items:
        chunks = text_splitter.split_text(item["content"])
        # ✅ Combine system metadata with custom product metadata
        base_metadata = {
            "source": item["url"],
            "category": "retail",
            "type": "crawl4ai"
        }
        if "metadata" in item:
            base_metadata.update(item["metadata"])

        all_documents.extend([
            Document(
                page_content=chunk, 
                metadata=base_metadata
            ) for chunk in chunks
        ])
    
    if not all_documents:
        return
        
    # ✅ Batch add for performance
    vector_store.add_documents(all_documents, batch_size=64)
    print(f"Batch added {len(all_documents)} product chunks to the vector store.")
