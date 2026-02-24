from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from vector_store import vector_store
import os
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
def add_content_to_store(content, metadata):
    """
    Standard ingestion for single pages.
    """
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_text(content)
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    vector_store.add_documents(documents, batch_size=64)
    print(f"Added {len(documents)} chunks to the vector store.")
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
        
        # 1. First split the content into chunks
        chunks = text_splitter.split_text(content)
        
        for chunk in chunks:
            # 2. Process images ONLY for this specific chunk
            processed_chunk, first_image = await asset_processor.process_raw_content(
                chunk, 
                category=metadata.get("category", "retail"),
                subcategory=metadata.get("subcategory", "general")
            )
            
            # 3. Apply metadata for this chunk
            chunk_metadata = {
                "source": url,
                "type": "crawl4ai"
            }
            chunk_metadata.update(metadata)
            
            # Use found image if metadata is missing one
            # Use found image if metadata is missing one AND it's not a logo
            if not chunk_metadata.get("image_url") and first_image:
                chunk_metadata["image_url"] = first_image
                chunk_metadata["s3_image_url"] = first_image
                # Basic check to avoid logos in fallback
                logolike = ["logo", "icon", "social", "facebook", "twitter", "linkedin", "instagram"]
                if not any(kw in first_image.lower() for kw in logolike):
                    chunk_metadata["image_url"] = first_image
                    chunk_metadata["s3_image_url"] = first_image
                
            # 4. Strip remaining Markdown/HTML image tags from the chunk
            clean_chunk = re.sub(r'!\[.*?\]\(.*?\)', '', processed_chunk)
            clean_chunk = re.sub(r'<img.*?>', '', clean_chunk, flags=re.IGNORECASE)
            
            all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))
            if clean_chunk and isinstance(clean_chunk, str):
                all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))
    
    if all_chunks:
        vector_store.add_documents(all_chunks, batch_size=64)
        print(f"Batch added {len(all_chunks)} chunks to the vector store.")
        # ChromaDB has a max batch size of 5461. 
        # Using manual loop of 500 to guarantee stability across all library versions.
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            vector_store.add_documents(batch)
            print(f"Added batch of {len(batch)} chunks. Total: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
