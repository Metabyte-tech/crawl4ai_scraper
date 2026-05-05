import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import vector_store
from product_extractor import product_extractor
import os
import asyncio
import hashlib
import gc
from concurrent.futures import ThreadPoolExecutor

# Dedicated executor for heavy embedding tasks to prevent overloading EC2 RAM/CPU
# We limit this to ONE thread to ensure heavy model calls are serialized globally across the process.
embedding_executor = ThreadPoolExecutor(max_workers=1)

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
        base_url=metadata.get("source"),
        category=metadata.get("category", "retail"),
        subcategory=metadata.get("subcategory", "general")
    )
    
    # 2. Extract structured product data (Price, Rating, Brand)
    extracted = product_extractor.extract_from_html(content, metadata.get("source"))
    for k, v in extracted.items():
        if v and not metadata.get(k):
            metadata[k] = v
    
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_text(content)
    
    all_chunks = []
    for chunk in chunks:
        chunk_metadata = metadata.copy()
        
        # Enforce defaults for search visibility
        if "category" not in chunk_metadata:
            chunk_metadata["category"] = "retail"
            
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
            loop = asyncio.get_running_loop()
            # Use dedicated single-threaded executor for the heavy model call
            await loop.run_in_executor(embedding_executor, lambda: vector_store.add_documents(all_chunks, batch_size=64))
            print(f"Added {len(all_chunks)} chunks for {metadata.get('source')} with image: {page_image}", flush=True)
        
        # Free memory immediately
        del all_chunks
        gc.collect()

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
            base_url=url,
            category=metadata.get("category", "retail"),
            subcategory=metadata.get("subcategory", "general")
        )

        # 1.5 Extract structured product data
        extracted = product_extractor.extract_from_html(content, url)
        for k, v in extracted.items():
            if v and not metadata.get(k):
                metadata[k] = v
        
        if metadata.get("price"):
            print(f"DEBUG: Found price '{metadata['price']}' for {url}", flush=True)
        else:
            print(f"DEBUG: No price found for {url}", flush=True)

        # 2. Split the content into chunks
        chunks = text_splitter.split_text(content)
        
        for chunk in chunks:
            # 3. Apply metadata for this chunk
            chunk_metadata = {
                "source": url,
                "type": "crawl4ai",
                "category": metadata.get("category", "retail") # Enforce search visibility
            }
            chunk_metadata.update(metadata)
            
            # Use found image if metadata is missing one
            if not chunk_metadata.get("image_url") and page_image:
                chunk_metadata["image_url"] = page_image
                chunk_metadata["s3_image_url"] = page_image
                
            # 4. Strip remaining Markdown/HTML image tags
            clean_chunk = re.sub(r'!\[.*?\]\)', '', chunk) # Clean messed up MD
            clean_chunk = re.sub(r'!\[.*?\]\(.*?\)', '', clean_chunk)
            clean_chunk = re.sub(r'<img.*?>', '', clean_chunk, flags=re.IGNORECASE)
            
            if clean_chunk and isinstance(clean_chunk, str):
                all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_metadata))
    
    if all_chunks:
        print(f"DEBUG: Deduping {len(all_chunks)} chunks for the vector store...", flush=True)
        seen_hashes = set()
        unique_chunks = []
        
        for doc in all_chunks:
            # Normalize text for hash (strip extra whitespace)
            norm_text = " ".join(doc.page_content.split())
            content_hash = hashlib.md5(norm_text.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(doc)
        
        reduction = len(all_chunks) - len(unique_chunks)
        print(f"DEBUG: Deduplication complete. Removed {reduction} duplicate chunks. Unique chunks: {len(unique_chunks)}", flush=True)
        
        # Free the original list early to save memory
        del all_chunks
        gc.collect()
        
        if unique_chunks:
            # Use upsert with deterministic IDs so re-runs skip already-stored chunks
            batch_size = 500
            total_unique = len(unique_chunks)
            print(f"DEBUG: Starting idempotent ingestion of {total_unique} unique chunks in batches of {batch_size}...", flush=True)

            for i in range(0, total_unique, batch_size):
                batch = unique_chunks[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_unique - 1) // batch_size + 1

                # Build deterministic IDs from content hash so Chroma skips duplicates on re-run
                batch_ids = []
                for doc in batch:
                    norm_text = " ".join(doc.page_content.split())
                    batch_ids.append(hashlib.md5(norm_text.encode("utf-8")).hexdigest())

                print(f"DEBUG: Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...", flush=True)

                import time
                start_t = time.time()

                # Move lock INSIDE the loop so we don't block the entire event loop for an hour
                async with write_lock:
                    loop = asyncio.get_running_loop()
                    # Pass explicit ids so Chroma upserts rather than blindly inserts
                    await loop.run_in_executor(
                        embedding_executor,
                        lambda b=batch, ids=batch_ids: vector_store.add_documents(b, ids=ids)
                    )

                elapsed = time.time() - start_t
                print(f"Added batch {batch_num} of {len(batch)} chunks in {elapsed:.2f}s. Total: {min(i + batch_size, total_unique)}/{total_unique}", flush=True)

                # Yield to the event loop to allow heartbeat logs and other tasks to run
                await asyncio.sleep(0.1)

            # Final cleanup for this task
            del unique_chunks
            gc.collect()

