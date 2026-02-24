from vector_store import vector_store
from functools import lru_cache
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# ✅ Fast Query Function
# ✅ Fast Query Function
def fast_query(query: str, category: str = None, threshold: float = 2.2, preferred_source: str = None, k: int = 25):
    """
    Returns a list of (document, score) tuples that meet the similarity threshold.
    If preferred_source is provided, it boosts results from that source (lower score).
    """
    where_filter = {}
    if category:
        where_filter["category"] = category

    if not where_filter:
        where_filter = None

    # Use similarity_search_with_score to get distances
    results_with_scores = vector_store.similarity_search_with_score(
        query,
        k=k, # Get more candidates to allow for boosting
        filter=where_filter
    )

    relevant_results = []
    for doc, score in results_with_scores:
        final_score = score
        
        # Source Affinity: Boost results from the preferred source
        if preferred_source:
            source = doc.metadata.get("source") or doc.metadata.get("source_url") or ""
            if preferred_source.lower() in source.lower():
                # Aggressively boost by subtracting 0.4 from the distance (similar to 25-30% boost)
                final_score -= 0.4
        
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
    print(f"DEBUG Retrieval for '{query}': Found {len(results)} docs")
    for i, res in enumerate(results):
        doc = res[0] if isinstance(res, tuple) else res
        print(f"  Doc {i} Snippet: {doc.page_content[:150]}...")
    
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
