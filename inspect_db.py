from query import fast_query

def inspect_docs(query):
    print(f"\n--- Inspecting docs for: '{query}' ---")
    results = fast_query(query, threshold=2.0)
    for i, (doc, score) in enumerate(results):
        print(f"\n[Result {i}] Score: {score:.4f}")
        print(f"Content Snippet: {doc.page_content[:200]}")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    inspect_docs("Zig")
