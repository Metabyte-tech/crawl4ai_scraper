import chromadb
import os

# Initialize ChromaDB
persist_directory = "/home/himanshu/workspace/ai-agent/crawl4AI/chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("crawl4ai_collection")

def clean_placeholders():
    print("Searching for placeholder data...")
    # Get all documents
    results = collection.get()
    ids_to_delete = []
    
    placeholders = ["Check Price", "Verifying...", "Finding the best price", "Finding live information", "Pending Background Check"]
    
    for i in range(len(results['ids'])):
        meta = results['metadatas'][i]
        doc_id = results['ids'][i]
        
        price = str(meta.get('price', ''))
        brand = str(meta.get('brand', ''))
        details = str(meta.get('details', ''))
        
        is_placeholder = any(p.lower() in price.lower() or p.lower() in brand.lower() or p.lower() in details.lower() for p in placeholders)
        
        if is_placeholder:
            print(f"Deleting placeholder doc {doc_id}: {meta.get('name')} (Price: {price})")
            ids_to_delete.append(doc_id)
            
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Successfully deleted {len(ids_to_delete)} placeholder documents.")
    else:
        print("No placeholder documents found.")

if __name__ == "__main__":
    clean_placeholders()
