from vector_store import vector_store
from functools import lru_cache
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# ✅ Fast Query Function
def fast_query(query: str, category: str = None):
    where_filter = {}
    if category:
        where_filter["category"] = category

    if not where_filter:
        where_filter = None

    results = vector_store.similarity_search(
        query,
        k=5,
        filter=where_filter
    )

    return results

# ✅ Version with no cache to ensure fresh RAG data after sync
def cached_query(query: str):
    """
    Fresh version of fast_query for dynamic retrieval.
    """
    results = fast_query(query)
    print(f"DEBUG Retrieval for '{query}': Found {len(results)} docs")
    for i, doc in enumerate(results):
        print(f"  Doc {i} Snippet: {doc.page_content[:150]}...")
    return results

class CachedRetriever(BaseRetriever):
    """
    Custom retriever that uses the lru_cache.
    """
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return cached_query(query)

def get_cached_retriever():
    return CachedRetriever()
