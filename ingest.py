from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from vector_store import vector_store
import os
import asyncio

# Global lock for vector store writes to avoid SQLite concurrency issues
write_lock = asyncio.Lock()
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
async def add_content_to_store(content, metadata):
    """
    Standard ingestion for single pages. 
    Processes images for the entire page first, then applies to chunks.
    """
    from asset_processor import asset_processor
    import re

    # 1. Identify and process the 'Best' image for the entire page
    _, page_image = await asset_processor.process_raw_content(
        content, 
        category=metadata.get("category", "retail"),
        subcategory=metadata.get("subcategory", "general")
    )
    
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_text(content)
    
    all_chunks = []
    for chunk in chunks:
        chunk_metadata = metadata.copy()
        # Use found image if metadata is missing one
        if not chunk_metadata.get("image_url") and page_image:
            chunk_metadata["image_url"] = page_image
            chunk_metadata["s3_image_url"] = page_image

        # Strip remaining Markdown/HTML image tags from the chunk to keep it clean for embedding
        clean_chunk = re.sub(r'!\[.*?\]\(.*?\)', '', chunk)
        clean_chunk = re.sub(r'<img.*?>', '', clean_chunk, flags=re.IGNORECASE)

        if clean_chunk and isinstance(clean_chunk, str):
            all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))

    if all_chunks:
        async with write_lock:
            vector_store.add_documents(all_chunks, batch_size=64)
            print(f"Added {len(all_chunks)} chunks for {metadata.get('source')} with image: {page_image}")
async def add_multiple_contents_to_store(items: list):
    """
    Items: list of {"content": str, "url": str, "metadata": dict}
    Processes images in raw content before ingestion.
    """
    from asset_processor import asset_processor
    import re
    
    text_splitter = get_text_splitter()
    all_chunks = []
    
    for item in items:
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        url = item.get("url", "")
        
        # 1. Process images for the ENTIRE product content first
        _, page_image = await asset_processor.process_raw_content(
            content, 
            category=metadata.get("category", "retail"),
            subcategory=metadata.get("subcategory", "general")
        )

        # 2. Split the content into chunks
        chunks = text_splitter.split_text(content)
        
        for chunk in chunks:
            # 3. Apply metadata for this chunk
            chunk_metadata = {
                "source": url,
                "type": "crawl4ai"
            }
            chunk_metadata.update(metadata)
            
            # Use found image if metadata is missing one
            if not chunk_metadata.get("image_url") and page_image:
                chunk_metadata["image_url"] = page_image
                chunk_metadata["s3_image_url"] = page_image
                
            # 4. Strip remaining Markdown/HTML image tags
            clean_chunk = re.sub(r'!\[.*?\]\)|\)', '', chunk) # Clean messed up MD
            clean_chunk = re.sub(r'!\[.*?\]\(.*?\)', '', clean_chunk)
            clean_chunk = re.sub(r'<img.*?>', '', clean_chunk, flags=re.IGNORECASE)
            
            if clean_chunk and isinstance(clean_chunk, str):
                all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))
    
    if all_chunks:
        print(f"Batch adding {len(all_chunks)} chunks to the vector store...")
        async with write_lock:
            # ChromaDB has a max batch size of 5461. 
            # Using manual loop of 500 to guarantee stability across all library versions.
            batch_size = 500
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                vector_store.add_documents(batch)
                print(f"Added batch of {len(batch)} chunks. Total: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
