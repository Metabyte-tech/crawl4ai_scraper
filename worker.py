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

class WorkerSettings:
    """
    Arq worker configuration.
    """
    functions = [cache_products_task, deep_crawl_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    # Increase timeout for heavy deep crawls
    job_timeout = 600 # 10 minutes
    # Max concurrent jobs per worker process
    max_jobs = 10
