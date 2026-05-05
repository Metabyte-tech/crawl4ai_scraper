import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import asyncio
from arq import create_pool
from arq.connections import RedisSettings
import os
from dotenv import load_dotenv

load_dotenv()

# Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def cache_products_task(ctx, products, query):
    """
    Background task to process images and store products in the cache.
    """
    print(f"--- [WORKER] Starting cache_products_task for query: {query} ---", flush=True)
    from kimi_service import kimi_service
    try:
        await kimi_service.cache_and_store_products(products, query)
        print(f"--- [WORKER] Finished cache_products_task for query: {query} ---", flush=True)
    except Exception as e:
        print(f"--- [WORKER ERROR] cache_products_task failed: {e} ---", flush=True)

async def deep_crawl_task(ctx, query, fast_products):
    """
    Background task to perform recursive deep crawl for high-quality data.
    """
    print(f"--- [WORKER] Starting deep_crawl_task for query: {query} ---", flush=True)
    from api import background_crawl_and_ingest
    try:
        await background_crawl_and_ingest(query, fast_products)
        print(f"--- [WORKER] Finished deep_crawl_task for query: {query} ---", flush=True)
    except Exception as e:
        print(f"--- [WORKER ERROR] deep_crawl_task failed: {e} ---", flush=True)

async def ingest_url_task(ctx, url, max_pages):
    """
    Background task to perform recursive deep crawl for URL ingestion to RAG DB.
    """
    print(f"--- [WORKER] Starting ingest_url_task for URL: {url} ---", flush=True)
    from api import background_ingest
    try:
        await background_ingest(url, max_pages)
        print(f"--- [WORKER] Finished ingest_url_task for URL: {url} ---", flush=True)
    except Exception as e:
        print(f"--- [WORKER ERROR] ingest_url_task failed: {e} ---", flush=True)

async def admin_ingest_url_task(ctx, batch_id, url, max_pages):
    """
    Tracked background task to perform recursive deep crawl for URL ingestion.
    """
    print(f"--- [WORKER] Starting admin_ingest_url_task for Batch: {batch_id}, URL: {url} ---", flush=True)
    from admin_service import admin_service
    from api import background_ingest
    
    await admin_service.update_url_status(batch_id, url, "running")
    try:
        # We re-run the logic here because background_ingest in api.py 
        # doesn't re-raise exceptions, making it hard to track failures there.
        from crawler import crawl_site, crawl_site_recursive
        from ingest import add_content_to_store, add_multiple_contents_to_store
        
        success = False
        if max_pages <= 1:
            try:
                content_links = await asyncio.wait_for(crawl_site(url), timeout=240)
                content, _ = content_links
                if content and len(content.strip()) > 10:
                    await add_content_to_store(content, {"source": url})
                    success = True
                else:
                    error_reason = "No content extracted (maybe blocked or invalid URL)"
            except asyncio.TimeoutError:
                error_reason = "Timeout (240s) during single page crawl"
        else:
            results = await crawl_site_recursive(url, max_pages=max_pages)
            if results:
                await add_multiple_contents_to_store(results)
                success = True
            else:
                error_reason = "No pages found in recursive crawl"
        
        if success:
            await admin_service.update_url_status(batch_id, url, "success")
            print(f"--- [WORKER] Finished admin_ingest_url_task for URL: {url} ---", flush=True)
        else:
            await admin_service.update_url_status(batch_id, url, f"failed: {error_reason}")
            print(f"--- [WORKER FAILURE] admin_ingest_url_task for URL: {url} -> {error_reason} ---", flush=True)

    except Exception as e:
        print(f"--- [WORKER ERROR] admin_ingest_url_task failed for {url}: {e} ---", flush=True)
        await admin_service.update_url_status(batch_id, url, f"failed: {str(e)}")

class WorkerSettings:
    """
    Arq worker configuration.
    """
    functions = [cache_products_task, deep_crawl_task, ingest_url_task, admin_ingest_url_task]
    # Initialize Redis settings explicitly to allow injecting connection timeouts
    # The default 1-sec timeout is too aggressive for heavy concurrent crawls
    redis_settings = RedisSettings(
        host="localhost",
        port=6379,
        conn_timeout=10,
        conn_retries=5,
        conn_retry_delay=1
    )
    # Increase timeout to 24 hours (86400s) for massive chunk ingestions
    job_timeout = 86400
    # Max concurrent jobs per worker process to avoid OOM on 8GB/16GB EC2
    # Reduced to 3 to leave overhead for Playwright and Torch
    max_jobs = 8
