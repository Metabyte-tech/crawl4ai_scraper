from vector_store import vector_store

def check_scores(query):
    print(f"\n--- Checking scores for: '{query}' ---")
    results = vector_store.similarity_search_with_score(query, k=5)
    for i, (doc, score) in enumerate(results):
        print(f"[{i}] Score: {score:.4f} | Snippet: {doc.page_content[:100]}...")

if __name__ == "__main__":
    check_scores("Show me some science toys")
    check_scores("Show me some smartwatches")
    check_scores("What is the capital of France?")
