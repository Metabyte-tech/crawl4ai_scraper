import os
import asyncio
import asyncpg
import json
import time
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

class DBService:
    def __init__(self):
        self.pool = None

    async def get_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(DATABASE_URL)
        return self.pool

    async def init_db(self):
        """Initializes the database tables if they don't exist."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS crawl_batches (
                    batch_id TEXT PRIMARY KEY,
                    start_time DOUBLE PRECISION,
                    end_time DOUBLE PRECISION,
                    total_urls INTEGER,
                    status TEXT
                );
                
                CREATE TABLE IF NOT EXISTS crawl_url_results (
                    id SERIAL PRIMARY KEY,
                    batch_id TEXT REFERENCES crawl_batches(batch_id) ON DELETE CASCADE,
                    url TEXT,
                    status TEXT,
                    updated_at DOUBLE PRECISION,
                    UNIQUE(batch_id, url)
                );
                
                CREATE INDEX IF NOT EXISTS idx_crawl_url_results_batch_id ON crawl_url_results(batch_id);
            """)

    async def upsert_batch(self, batch_id: str, metadata: Dict):
        """Inserts or updates a crawl batch record."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO crawl_batches (batch_id, start_time, end_time, total_urls, status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (batch_id) DO UPDATE SET
                    end_time = EXCLUDED.end_time,
                    total_urls = EXCLUDED.total_urls,
                    status = EXCLUDED.status
            """, batch_id, metadata.get('start_time'), metadata.get('end_time'), 
                 metadata.get('total_urls'), metadata.get('status'))

    async def upsert_url_result(self, batch_id: str, url: str, status: str):
        """Inserts or updates a URL status within a batch."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO crawl_url_results (batch_id, url, status, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (batch_id, url) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
            """, batch_id, url, status, time.time())

    async def list_batches(self, limit: int = 20) -> List[Dict]:
        """Lists recent crawl batches."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM crawl_batches 
                ORDER BY start_time DESC 
                LIMIT $1
            """, limit)
            return [dict(row) for row in rows]

    async def get_batch_detail(self, batch_id: str) -> Optional[Dict]:
        """Retrieves full details of a batch and its URL results."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            batch = await conn.fetchrow("SELECT * FROM crawl_batches WHERE batch_id = $1", batch_id)
            if not batch:
                return None
            
            urls = await conn.fetch("SELECT url, status, updated_at FROM crawl_url_results WHERE batch_id = $1", batch_id)
            
            result = dict(batch)
            result['results'] = [dict(u) for u in urls]
            
            # Add summary
            total_success = sum(1 for u in urls if u['status'] == 'success')
            total_failed = sum(1 for u in urls if u['status'].startswith('failed'))
            result['summary'] = {
                "total_success": total_success,
                "total_failed": total_failed,
                "total_urls": len(urls)
            }
            return result

    async def delete_batch(self, batch_id: str):
        """Deletes a batch and its associated URL results from PostgreSQL."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Note: crawl_url_results has ON DELETE CASCADE on batch_id
            await conn.execute("DELETE FROM crawl_batches WHERE batch_id = $1", batch_id)

db_service = DBService()
