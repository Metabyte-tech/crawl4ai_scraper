import chromadb
import json
import os
import sys

def view_content():
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print(f"Error: Database directory '{db_path}' not found.")
        return

    # Check for --all or a specific limit
    show_all = "--all" in sys.argv
    limit = 10
    if not show_all:
        for arg in sys.argv:
            if arg.startswith("--limit="):
                try:
                    limit = int(arg.split("=")[1])
                except ValueError:
                    pass

    print(f"Connecting to ChromaDB at: {os.path.abspath(db_path)}")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collections = client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {e}")
        return

    if not collections:
        print("No collections found in ChromaDB.")
        return

    target_collection_name = "crawl4ai_collection"
    collection_names = [col.name if hasattr(col, 'name') else str(col) for col in collections]
    
    if target_collection_name not in collection_names:
        if collection_names:
            target_collection_name = collection_names[0]
        else:
            return

    try:
        collection = client.get_collection(name=target_collection_name)
        count = collection.count()
        print(f"\n--- Content for collection: {target_collection_name} ({count} items) ---")

        if count == 0:
            print("Collection is empty.")
            return

        display_count = count if show_all else min(count, limit)

        if not show_all and count > limit:
            print(f"Showing first {limit} items. Use --all to see everything, or --limit=N for a specific count.\n")

        # Get only the IDs and pieces we need to avoid the "too many SQL variables" error
        results = collection.get(limit=display_count)

        ids = results.get('ids', [])
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])

        for i in range(len(ids)):
            print(f"\n[{i+1}/{count}] ID: {ids[i]}")
            if i < len(metadatas) and metadatas[i]:
                print(f"Metadata: {json.dumps(metadatas[i], indent=2)}")
            
            if i < len(documents):
                content = documents[i]
                print(f"Content:\n{content[:1000]}..." if len(content) > 1000 and not show_all else f"Content:\n{content}")
            print("-" * 50)
            
        if not show_all and count > limit:
            print(f"\n... and {count - limit} more items.")

    except Exception as e:
        print(f"Error accessing collection {target_collection_name}: {e}")

if __name__ == "__main__":
    view_content()
