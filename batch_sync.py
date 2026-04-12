import asyncio
import sys
import os
from retail_crawler import retail_crawler

# Set a semaphore to limit concurrent crawls to 5 to avoid resource exhaustion
MAX_CONCURRENT_CRAWLS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)

async def sync_single_url(url, category=None):
    """
    Worker task to sync one URL.
    """
    async with semaphore:
        print(f"📡 [BATCH] Starting sync for: {url}")
        try:
            # max_pages=10 for batch sync to keep it relatively fast but thorough
            results = await retail_crawler.sync_store(url, max_pages=10, target_category=category or "products")
            if results:
                print(f"✅ [BATCH] SUCCESS: {url} (found {len(results)} products)")
            else:
                print(f"⚠️ [BATCH] No products found for: {url}")
            return results
        except Exception as e:
            print(f"❌ [BATCH] ERROR for {url}: {e}")
            return []

async def run_batch_sync(urls, category=None):
    """
    Orchestrates batch syncing of multiple URLs.
    """
    print(f"\n🚀 --- BATCH RETAIL SYNC STARTED ---")
    print(f"Processing {len(urls)} URLs in batches of {MAX_CONCURRENT_CRAWLS}...")
    
    tasks = [sync_single_url(url, category) for url in urls]
    all_results = await asyncio.gather(*tasks)
    
    total_products = sum(len(r) for r in all_results if r)
    print(f"\n✨ --- BATCH SYNC COMPLETE ---")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Total products ingested: {total_products}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_sync.py \"url1,url2,url3\" [category]")
        print("Alternative: python batch_sync.py @urls.txt [category]")
        sys.exit(1)
    
    input_str = sys.argv[1]
    target_cat = sys.argv[2] if len(sys.argv) > 2 else None
    
    urls = []
    if input_str.startswith("@"):
        # Load from file
        file_path = input_str[1:]
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            print(f"Error: File {file_path} not found.")
            sys.exit(1)
    else:
        # Load from comma-separated string
        urls = [u.strip() for u in input_str.split(",") if u.strip()]
    
    if not urls:
        print("No valid URLs found.")
        sys.exit(1)
        
    asyncio.run(run_batch_sync(urls, target_cat))
