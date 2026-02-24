import asyncio
import sys
from retail_crawler import retail_crawler

async def run_manual_sync(url, category=None):
    """
    Manually triggers a sync for a given URL.
    Args:
        url (str): The seed URL to start crawling from.
        category (str): The product category to search for (e.g., 'toys', 'shoes').
    """
    print(f"--- MANUAL RETAIL SYNC STARTED ---")
    print(f"Target URL: {url}")
    print(f"Target Category: {category or 'all relevant'}")
    
    try:
        # We use a higher max_pages for manual sync to ensure thorough traversal
        results = await retail_crawler.sync_store(url, max_pages=30, target_category=category or "products")
        
        if results:
            print(f"\n✅ SUCCESS: Synced {len(results)} products.")
            print(f"Images have been uploaded to S3 with categorization.")
            for i, p in enumerate(results[:5]):
                print(f" [{i+1}] {p.get('name')} | Image: {p.get('image_url')}")
            if len(results) > 5:
                print(f" ... and {len(results) - 5} more.")
        else:
            print("\n⚠️  No products were extracted. Check logs for details.")
            
    except Exception as e:
        print(f"\n❌ ERROR during manual sync: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_sync.py <URL> [category]")
        sys.exit(1)
    
    seed_url = sys.argv[1]
    target_cat = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(run_manual_sync(seed_url, target_cat))
