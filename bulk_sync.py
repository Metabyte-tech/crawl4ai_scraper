import asyncio
import json
import os
import argparse
from datetime import datetime
from retail_crawler import retail_crawler

STATUS_FILE = "crawl_status.json"

class BulkSyncManager:
    def __init__(self, seeds_file, max_concurrent_sites=3):
        self.seeds_file = seeds_file
        self.max_concurrent_sites = max_concurrent_sites
        self.status = self._load_status()
        self.semaphore = asyncio.Semaphore(max_concurrent_sites)

    def _load_status(self):
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_status(self):
        with open(STATUS_FILE, "w") as f:
            json.dump(self.status, f, indent=2)

    def _get_urls_from_file(self):
        if not os.path.exists(self.seeds_file):
            print(f"Error: Seeds file {self.seeds_file} not found.")
            return []
        with open(self.seeds_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    async def _crawl_site_task(self, url):
        async with self.semaphore:
            print(f"\n[QUEUE] Starting crawl for: {url}")
            self.status[url] = {
                "status": "in-progress",
                "start_time": datetime.now().isoformat()
            }
            self._save_status()

            try:
                # Run the retail crawler sync
                # We can adjust max_pages as needed for bulk. Default is 50.
                results = await retail_crawler.sync_store(url, max_pages=50)
                
                self.status[url] = {
                    "status": "completed",
                    "product_count": len(results or []),
                    "end_time": datetime.now().isoformat()
                }
                print(f"[SUCCESS] Finished {url}: found {len(results or [])} products.")
            except Exception as e:
                self.status[url] = {
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                }
                print(f"[ERROR] Failed {url}: {e}")
            
            self._save_status()

    async def run(self):
        urls = self._get_urls_from_file()
        if not urls:
            return

        # Initialize status for new URLs
        for url in urls:
            if url not in self.status:
                self.status[url] = {"status": "pending"}
        self._save_status()

        # Filter pending or failed URLs
        to_crawl = [url for url in urls if self.status[url]["status"] in ["pending", "failed"]]
        
        if not to_crawl:
            print("All URLs are already completed or in-progress.")
            return

        print(f"Total URLs to crawl: {len(to_crawl)} (Concurrency: {self.max_concurrent_sites})")
        
        tasks = [self._crawl_site_task(url) for url in to_crawl]
        await asyncio.gather(*tasks)
        print("\nAll bulk crawl tasks finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Ingestion Orchestrator for Crawl4AI")
    parser.add_argument("--file", default="seeds.txt", help="Path to the seeds.txt file containing URLs")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent website crawls")
    args = parser.parse_args()

    # Ensure the seeds file exists
    if not os.path.exists(args.file):
        with open(args.file, "w") as f:
            f.write("# Add one URL per line\n")
        print(f"Created empty {args.file}. Please add URLs and run again.")
    else:
        asyncio.run(BulkSyncManager(args.file, args.concurrency).run())
