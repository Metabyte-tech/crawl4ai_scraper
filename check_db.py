from vector_store import vector_store
try:
    data = vector_store.get()
    ids = data.get('ids', [])
    metas = data.get('metadatas', [])

    print(f"--- Vector Store Status ---")
    print(f"Total Documents: {len(ids)}")

    query = "children's toys, games, and play sets"
    results = vector_store.similarity_search_with_score(query, k=5)
    
    print(f"--- Top 5 Toy Metadata ---")
    for doc, score in results:
        print(f"Name: {doc.metadata.get('name')}")
        print(f"  Image: {doc.metadata.get('image_url')}")
        print(f"  S3 Image: {doc.metadata.get('s3_image_url')}")
        print(f"  Category: {doc.metadata.get('category')}")

except Exception as e:
    print(f"Error: {e}")
