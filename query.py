from vector_store import vector_store
from functools import lru_cache
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# ✅ Fast Query Function
# ✅ Fast Query Function
def fast_query(query: str, category: str = None, threshold: float = 2.0, preferred_source: str = None, k: int = 25):
    """
    Returns a list of (document, score) tuples that meet the similarity threshold.
    If preferred_source is provided, it boosts results from that source (lower score).
    """
    import time
    import random
    max_retries = 5
    where_filter = {}
    if category:
        where_filter["category"] = category

    if not where_filter:
        where_filter = None

    results_with_scores = []
    for attempt in range(max_retries):
        try:
            # Use similarity_search_with_score to get distances
            results_with_scores = vector_store.similarity_search_with_score(
                query,
                k=k, # Get more candidates to allow for boosting
                filter=where_filter
            )
            break
        except Exception as e:
            # Check if it's a transient locking/concurrency error
            err_msg = str(e).lower()
            transient_messages = ["error finding id", "database is locked", "timeout", "connection"]
            
            # If we hit "error finding id", it might be a corrupt index.
            # We retry, but also log a hint if it persists.
            is_transient = any(msg in err_msg for msg in transient_messages)
            
            if is_transient and attempt < max_retries - 1:
                # Exponential backoff
                wait_time = (2 ** attempt) * 0.5 + (random.random() * 0.1)
                print(f"⚠️ ChromaDB transient/index error ({err_msg}), retrying in {wait_time:.2f}s... ({attempt+1}/{max_retries})", flush=True)
                time.sleep(wait_time)
                continue
            else:
                if "error finding id" in err_msg:
                    print("❌ FATAL: ChromaDB index appears corrupted (error finding id).", flush=True)
                    print("💡 TIP: Try clearing ./chroma_db directory or call clear_vector_store().", flush=True)
                print(f"❌ ChromaDB fatal error after {attempt+1} attempts: {e}", flush=True)
                raise e

    relevant_results = []
    for doc, score in results_with_scores:
        final_score = score
        
        # Source Affinity: Boost results from the preferred source
        if preferred_source:
            source = doc.metadata.get("source") or doc.metadata.get("source_url") or ""
            if preferred_source.lower() in source.lower():
                # Aggressively boost by subtracting 0.5 from the distance (stronger boost)
                final_score -= 0.5
        
        if final_score < threshold:
            # Image Boost: Prioritize results with visual content
            img = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url") or doc.metadata.get("Image URL")
            if img:
                final_score -= 0.3
            
            relevant_results.append((doc, final_score))

    # Re-sort by final score
    relevant_results.sort(key=lambda x: x[1])
    
    return relevant_results

# ✅ Version with no cache to ensure fresh RAG data after sync
def cached_query(query: str):
    """
    Fresh version of fast_query for dynamic retrieval.
    """
    results = fast_query(query)
    print(f"DEBUG Retrieval for '{query}': Found {len(results)} docs", flush=True)
    for i, res in enumerate(results):
        doc = res[0] if isinstance(res, tuple) else res
        print(f"  Doc {i} Snippet: {doc.page_content[:150]}...", flush=True)
    
    # Return JUST the documents for LangChain compatibility
    return [res[0] if isinstance(res, tuple) else res for res in results]

class CachedRetriever(BaseRetriever):
    """
    Custom retriever that uses the lru_cache.
    """
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return cached_query(query)

def get_cached_retriever():
    return CachedRetriever()
