import json
import time
import uuid
import os
import logging
import redis.asyncio as redis
from typing import List, Optional, Dict
from db_service import db_service

logger = logging.getLogger(__name__)

class AdminService:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis_client = None

    async def get_redis(self):
        if not self._redis_client:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def start_batch(self, urls: List[str]) -> str:
        """Initializes a new crawl batch and returns its ID."""
        from datetime import datetime
        date_str = datetime.now().strftime("%d-%b-%Y")
        batch_id = f"batch_{date_str}_{uuid.uuid4().hex[:4]}"
        r = await self.get_redis()
        
        # Store batch metadata
        metadata = {
            "batch_id": batch_id,
            "start_time": time.time(),
            "total_urls": len(urls),
            "status": "running"
        }
        await r.hset(f"admin:crawl_batch:{batch_id}", mapping=metadata)
        
        # Initialize URL statuses
        url_statuses = {url: "queued" for url in urls}
        await r.hset(f"admin:crawl_batch:{batch_id}:urls", mapping=url_statuses)
        
        # Add to recent batches list (ZSET by timestamp)
        await r.zadd("admin:crawl_batches", {batch_id: time.time()})
        
        # Persist to PostgreSQL
        await db_service.upsert_batch(batch_id, metadata)
        for url in urls:
            await db_service.upsert_url_result(batch_id, url, "queued")
        
        return batch_id

    async def update_url_status(self, batch_id: str, url: str, status: str):
        """Updates the status of a specific URL within a batch."""
        r = await self.get_redis()
        await r.hset(f"admin:crawl_batch:{batch_id}:urls", url, status)
        
        # Check if batch is finished
        url_statuses = await r.hgetall(f"admin:crawl_batch:{batch_id}:urls")
        is_finished = all(s not in ["queued", "running"] for s in url_statuses.values())
        
        if is_finished:
            await r.hset(f"admin:crawl_batch:{batch_id}", "status", "finished")
            await r.hset(f"admin:crawl_batch:{batch_id}", "end_time", time.time())
            
        # Persist to PostgreSQL
        await db_service.upsert_url_result(batch_id, url, status)
        if is_finished:
            meta = await r.hgetall(f"admin:crawl_batch:{batch_id}")
            await db_service.upsert_batch(batch_id, meta)

    async def get_batch_status(self, batch_id: str) -> Dict:
        """Retrieves the full status and URL details of a batch."""
        r = await self.get_redis()
        metadata = await r.hgetall(f"admin:crawl_batch:{batch_id}")
        if not metadata:
            return None
        
        url_statuses = await r.hgetall(f"admin:crawl_batch:{batch_id}:urls")
        
        # Consistent format for all URLs
        results = []
        total_success = 0
        total_failed = 0
        
        for url, status in url_statuses.items():
            if status == "success":
                results.append({"url": url, "status": "success"})
                total_success += 1
            elif status == "queued" or status == "running":
                results.append({"url": url, "status": status})
            else:
                reason = status.replace("failed: ", "") if status.startswith("failed") else status
                results.append({"url": url, "status": "failed", "reason": reason})
                total_failed += 1
        
        metadata["results"] = results
        metadata["summary"] = {
            "total_success": total_success,
            "total_failed": total_failed,
            "total_urls": len(url_statuses)
        }
        return metadata

    async def list_recent_batches(self, limit: int = 10) -> List[Dict]:
        """Lists recent batch IDs and their high-level status."""
        r = await self.get_redis()
        batch_ids = await r.zrevrange("admin:crawl_batches", 0, limit - 1)
        
        batches = []
        for bid in batch_ids:
            meta = await r.hgetall(f"admin:crawl_batch:{bid}")
            if meta:
                batches.append(meta)
        return batches

    async def delete_batch(self, batch_id: str):
        """Deletes all Redis keys associated with a batch."""
        r = await self.get_redis()
        await r.delete(f"admin:crawl_batch:{batch_id}")
        await r.delete(f"admin:crawl_batch:{batch_id}:urls")
        await r.zrem("admin:crawl_batches", batch_id)
        
        # Also delete from PostgreSQL (non-fatal if DB is unreachable)
        try:
            await db_service.delete_batch(batch_id)
        except Exception as e:
            logger.warning(f"PostgreSQL delete skipped for batch {batch_id}: {e}")

admin_service = AdminService()
