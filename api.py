import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import time
import re
import json
import random
import os
from crawler import crawl_site, crawl_site_recursive # Preserved for future use
from ingest import add_content_to_store, add_multiple_contents_to_store # Preserved for future use
from vector_store import clear_vector_store
from query import fast_query
from bot import chat_with_bot
from kimi_service import kimi_service
from arq import create_pool # Preserved for future use
from arq.connections import RedisSettings # Preserved for future use
from urllib.parse import urlparse
from admin_service import admin_service
from db_service import db_service

last_crawled_domain = None


def update_last_domain(url):
    global last_crawled_domain
    try:
        domain = urlparse(url).netloc
        if domain:
            last_crawled_domain = domain
    except Exception:
        pass


def format_response(res):
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        if res.get("type") == "images":
            results = res.get("results", [])
            return f"\n\n<product_grid>{json.dumps(results)}</product_grid>\n\n"
        # If it looks like a single product, wrap it in a grid list
        if any(k in res for k in ["name", "price", "title", "url"]):
            return f"\n\n<product_grid>{json.dumps([res])}</product_grid>\n\n"
        return json.dumps(res, indent=2)
    return str(res)


def rebuild_carousel_with_map(content, lookup_map):
    if not isinstance(content, str) or not lookup_map:
        return content

    def reconstruct(match):
        tags_open, names_str, tags_close = match.group(1), match.group(2).strip(), match.group(3)
        try:
            names = json.loads(names_str)
            if not isinstance(names, list):
                names = [names]
            rebuilt = []
            for name in names:
                nc = str(name).strip().lower()
                data = lookup_map.get(nc) or next(
                    (v for k, v in lookup_map.items() if nc in k or k in nc), None
                )
                if data:
                    rebuilt.append(data)
            if not rebuilt and lookup_map:
                rebuilt = list(lookup_map.values())[:5]
            if not rebuilt:
                return ""
            rebuilt.sort(key=lambda p: 0 if p.get("image_url") else 1)
            return f"{tags_open}{json.dumps(rebuilt, separators=(',', ':'))}{tags_close}"
        except Exception as e:
            print(f"Carousel reconstruct error: {e}", flush=True)
            return match.group(0)

    return re.sub(r'(<product_carousel>)(.*?)(</product_carousel>)', reconstruct, content, flags=re.DOTALL)


# async def background_ingest(url: str, max_pages: int = 1, region="Global", currency="USD", category="General"):
#     try:
#         metadata = {
#             "source": url, 
#             "region": region, 
#             "currency": currency, 
#             "category": category,
#             "type": "crawl4ai"
#         }
#         
#         if max_pages <= 1:
#             content, _ = await crawl_site(url)
#             if content and len(content.strip()) > 10:
#                 await add_content_to_store(content, metadata)
#                 update_last_domain(url)
#         else:
#             results = await crawl_site_recursive(url, max_pages=max_pages)
#             if results:
#                 # Enrich each result with the base metadata
#                 for res in results:
#                     res_meta = metadata.copy()
#                     res_meta["source"] = res.get("url", url)
#                     res["metadata"] = res_meta
#                 await add_multiple_contents_to_store(results)
#     except Exception as e:
#         print(f"Background ingest error for {url}: {e}", flush=True)


# async def background_crawl_and_ingest(query: str, fast_products: list):
#     try:
#         print(f"🔄 BACKGROUND: Deep crawl for '{query}'...", flush=True)
#         deep_results = await kimi_service.run_deep_crawl_process(query, fast_products)
#         if deep_results:
#             await kimi_service.cache_and_store_products(deep_results, query)
#         print(f"✅ BACKGROUND: Done for '{query}'!", flush=True)
#     except Exception as e:
#         print(f"❌ BACKGROUND: Failed for '{query}': {e}", flush=True)
#         import traceback
#         traceback.print_exc()


app = FastAPI(title="Retail AI RAG API")

@app.on_event("startup")
async def startup():
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    print(f"📡 API attempting to connect to Redis at: {redis_url}", flush=True)
    try:
        # Muted ARQ Pool for architecture pivot
        # app.state.arq_pool = await create_pool(RedisSettings.from_dsn(redis_url))
        # await app.state.arq_pool.set('api_health_check', 'ok')
        # print("🚀 ARQ Redis Pool initialized and verified", flush=True)
        
        # Initialize PostgreSQL
        await db_service.init_db()
        print("💾 PostgreSQL History DB initialized", flush=True)
    except Exception as e:
        print(f"❌ Initialization error: {e}", flush=True)
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown():
    # Muted ARQ Pool
    # await app.state.arq_pool.close()
    # print("💤 ARQ Redis Pool closed", flush=True)
    pass


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🔍 {request.method} {request.url.path}", flush=True)
    response = await call_next(request)
    print(f"📉 {request.method} {request.url.path} → {response.status_code}", flush=True)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrawlRequest(BaseModel):
    url: str
    region: Optional[str] = "Global"
    currency: Optional[str] = "USD"
    category: Optional[str] = "General"

class CrawlBatchRequest(BaseModel):
    urls: List[str]
    region: Optional[str] = "Global"
    currency: Optional[str] = "USD"
    category: Optional[str] = "General"



class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    parts: Optional[List[dict]] = None
    id: Optional[str] = None
    model_config = {"extra": "ignore"}


class ChatRequest(BaseModel):
    id: Optional[str] = None
    message: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    selectedChatModel: Optional[str] = None
    selectedVisibilityType: Optional[str] = None
    model_config = {"extra": "ignore"}


@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest, req: Request):
    raise HTTPException(status_code=503, detail="Crawling functionality is currently disabled.")
    # if not request.url.startswith("http"):
    #     raise HTTPException(status_code=400, detail="Invalid URL protocol")
    # await req.app.state.arq_pool.enqueue_job(
    #     'ingest_url_task', 
    #     url=request.url, 
    #     max_pages=1,
    #     region=request.region,
    #     currency=request.currency,
    #     category=request.category
    # )
    # return {"status": "success", "message": f"Ingestion queued for {request.url}"}


@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest, req: Request):
    raise HTTPException(status_code=503, detail="Crawling functionality is currently disabled.")
    # if not request.url.startswith("http"):
    #     raise HTTPException(status_code=400, detail="Invalid URL protocol")
    # await req.app.state.arq_pool.enqueue_job('deep_crawl_task', query=request.url, fast_products=[])
    # return {"status": "success", "message": "Deep ingestion started"}


# ── Batch endpoints (multiple URLs at once) ──────────────────────────────────

@app.post("/crawl/batch")
async def crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    raise HTTPException(status_code=503, detail="Crawling functionality is currently disabled.")
    # """Light crawl for multiple URLs. URLs are processed concurrently by the worker."""
    # valid = [u for u in request.urls if u.startswith("http")]
    # if not valid:
    #     raise HTTPException(status_code=400, detail="No valid URLs provided (must start with http/https)")

    # for url in valid:
    #     await req.app.state.arq_pool.enqueue_job(
    #         'ingest_url_task', 
    #         url=url, 
    #         max_pages=1,
    #         region=request.region,
    #         currency=request.currency,
    #         category=request.category
    #     )

    # return {
    #     "status": "success",
    #     "message": f"Ingestion queued for {len(valid)} URL(s)",
    #     "urls": valid,
    # }


@app.post("/crawl/deep/batch")
async def deep_crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    raise HTTPException(status_code=503, detail="Crawling functionality is currently disabled.")
    # """Deep crawl for multiple URLs. Offloads to background worker for deep recursive crawling."""
    # valid = [u for u in request.urls if u.startswith("http")]
    # if not valid:
    #     raise HTTPException(status_code=400, detail="No valid URLs provided")

    # for url in valid:
    #     await req.app.state.arq_pool.enqueue_job('ingest_url_task', url=url, max_pages=5000)

    # return {"status": "success", "message": f"Deep ingestion queued for {len(valid)} URL(s)", "urls": valid}


# ── Admin endpoints (tracked batch crawls) ───────────────────────────────────

@app.post("/admin/crawl/batch")
async def admin_crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    raise HTTPException(status_code=503, detail="Crawling functionality is currently disabled.")
    # """Deep crawl for multiple URLs with admin-level status tracking."""
    # valid = [u for u in request.urls if u.startswith("http")]
    # if not valid:
    #     raise HTTPException(status_code=400, detail="No valid URLs provided")

    # batch_id = await admin_service.start_batch(valid)
    
    # for url in valid:
    #     # Enqueue with higher priority if needed, or just same pool
    #     await req.app.state.arq_pool.enqueue_job(
    #         'admin_ingest_url_task', 
    #         batch_id=batch_id, 
    #         url=url, 
    #         max_pages=5000
    #     )

    # return {
    #     "status": "success", 
    #     "message": f"Tracked batch crawl started for {len(valid)} URL(s)", 
    #     "batch_id": batch_id,
    #     "urls": valid
    # }

@app.get("/admin/crawl/batches")
async def list_admin_batches():
    """Returns a list of recent admin crawl batches directly from PostgreSQL."""
    return {"batches": []}
    # batches = await db_service.list_batches(limit=50)
    # for b in batches:
    #     b['start_time'] = str(b['start_time'])
    #     b['end_time'] = str(b['end_time']) if b.get('end_time') else None
    # return {"batches": batches}

@app.get("/admin/crawl/batch/{batch_id}")
async def get_admin_batch_status(batch_id: str):
    raise HTTPException(status_code=404, detail="Crawling is disabled")
    # """Returns the detailed status of a specific crawl batch from PostgreSQL."""
    # status = await db_service.get_batch_detail(batch_id)
    # if not status:
    #     # Fallback to Redis if somehow PostgreSQL fails immediately after creating
    #     status = await admin_service.get_batch_status(batch_id)
    # if not status:
    #     raise HTTPException(status_code=404, detail="Batch not found")
    # return status

@app.delete("/admin/crawl/batch/{batch_id}")
async def delete_admin_batch(batch_id: str):
    raise HTTPException(status_code=503, detail="Crawling is disabled")
    # await admin_service.delete_batch(batch_id)
    # return {"status": "deleted", "batch_id": batch_id}

# ── History endpoints (Permanent storage) ────────────────────────────────────

@app.get("/admin/crawl-history")
async def list_crawl_history(limit: int = 20):
    return {"batches": []}
    # """Returns a list of historical crawl batches from PostgreSQL."""
    # batches = await db_service.list_batches(limit)
    # return {"batches": batches}

@app.get("/admin/crawl-history/{batch_id}")
async def get_crawl_history_detail(batch_id: str):
    raise HTTPException(status_code=404, detail="Crawling is disabled")
    # """Returns full detail for a historical batch from PostgreSQL."""
    # detail = await db_service.get_batch_detail(batch_id)
    # if not detail:
    #     raise HTTPException(status_code=404, detail="Historical batch not found")
    # return detail


@app.post("/clear")
async def clear_endpoint():
    clear_vector_store()
    return {"status": "success", "message": "Memory cleared successfully"}

@app.get("/api/categories")
async def get_categories():
    """Returns the universal retail category tree. Live searches are triggered on click."""
    try:
        db_categories = await db_service.get_all_categories()
        
        # Build hierarchy
        top_clusters = [c for c in db_categories if c['parent_id'] is None]
        
        result = []
        for cat in top_clusters:
            display_name = cat['name']
            slug = cat['slug']
            
            # Subcategories
            subs = [c['name'] for c in db_categories if c['parent_id'] == cat['id']]
            
            # Category Redesign: Return empty items list. 
            # The frontend CategoryGrid component will gracefully fallback to showing
            # the static category preview image, and will trigger a live chat search when clicked.
            items = []
            
            result.append({
                "id": cat['id'],
                "name": display_name,
                "slug": slug,
                "subcategories": subs,
                "items": items
            })
            
        return JSONResponse(content={"categories": result})
    except Exception as e:
        print(f"Error in get_categories: {e}")
        return JSONResponse(content={"categories": [], "error": str(e)}, status_code=500)

# async def background_seed_categories():
#     """Proactively populates all categories in parallel with strict tagging."""
#     print("🚀 Starting strict background category seeding...", flush=True)
#     try:
#         db_categories = await db_service.get_all_categories()
#         top_clusters = [c for c in db_categories if c['parent_id'] is None]
#         
#         async def seed_single_category(cat):
#             display_name = cat['name']
#             query_term = f"official retail products for {display_name}"
#             
#             # Check if we already have enough items for THIS category
#             current_docs = fast_query(query_term, category=display_name, k=5)
#             valid_cached = [d for d in current_docs if d[0].metadata.get("image_url") and "placehold" not in d[0].metadata.get("image_url")]
#             
#             if len(valid_cached) < 3:
#                 # Use subcategories for surgical precision
#                 subs = [c['name'] for c in db_categories if c['parent_id'] == cat['id']]
#                 sub_tail = f" {' '.join(subs[:2])}" if subs else ""
#                 
#                 print(f"🔍 Discovery: Seeding '{display_name}' with precision search...", flush=True)
#                 # FORCE RETAIL DOMAINS and subcategory context
#                 specific_query = f"{display_name}{sub_tail} official product site:amazon.com OR site:walmart.com"
#                 live_products = await kimi_service.get_fast_bing_data(specific_query, num_results=15)
#                 
#                 if live_products:
#                     # Deduplicate within category
#                     unique_lives = []
#                     seen = set()
#                     for p in live_products:
#                         if p.get("url") not in seen and p.get("image_url"):
#                             unique_lives.append(p)
#                             seen.add(p.get("url"))
@app.post("/api/categories/seed")
async def trigger_seed(background_tasks: BackgroundTasks):
    raise HTTPException(status_code=503, detail="Category background seeding is currently disabled.")

# ── Session Cache & Semantic Search ──────────────────────────────────────────
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    session_embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to load sentence_transformers: {e}")
    session_embedder = None

_session_cache: dict = {}
MAX_POOL_SIZE = 200
RESULT_THRESHOLD = 10
PAGE_SIZE = 50

def cosine_similarity_np(v1, v2):
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot / (norm_v1 * norm_v2)

def clear_session(conv_id: str):
    if conv_id in _session_cache:
        del _session_cache[conv_id]

def get_session_pool(conv_id: str):
    return _session_cache.get(conv_id)

def set_session_pool(conv_id: str, products: list, topic: str, query: str):
    embeddings = []
    if session_embedder:
        texts = [f"{p.get('name', '')} {p.get('brand', '')} {p.get('details', '')} {p.get('category', '')}" for p in products]
        if texts:
            embeddings = session_embedder.encode(texts).tolist()
    
    _session_cache[conv_id] = {
        "raw_pool": products[:MAX_POOL_SIZE],
        "pool_embeddings": embeddings[:MAX_POOL_SIZE] if embeddings else [],
        "last_topic": topic,
        "last_query": query,
        "display_offset": 0
    }

def extend_session_pool(conv_id: str, new_products: list):
    session = _session_cache.get(conv_id)
    if not session: return
    
    existing_urls = {p.get("url") for p in session["raw_pool"] if p.get("url")}
    unique_new = [p for p in new_products if p.get("url") not in existing_urls]
    
    room_left = MAX_POOL_SIZE - len(session["raw_pool"])
    added = unique_new[:room_left]
    
    if not added: return
    
    session["raw_pool"].extend(added)
    if session_embedder:
        texts = [f"{p.get('name', '')} {p.get('brand', '')} {p.get('details', '')} {p.get('category', '')}" for p in added]
        if texts:
            new_embs = session_embedder.encode(texts).tolist()
            session["pool_embeddings"].extend(new_embs)

def _is_followup(query: str, conv_id: str) -> bool:
    if conv_id not in _session_cache: return False
    
    signals = [
        "the ones", "those", "among them", "from these", "filter", "sort by", 
        "only show", "show only", "cheaper", "more expensive", "best rated", 
        "under", "above", "below", "between", "which one", "which ones", 
        "any", "brand", "color", "size", "more", "next page", "load more"
    ]
    query_lower = query.lower()
    
    # Keyword match
    if any(s in query_lower for s in signals):
        return True
        
    return False

def _filter_pool(conv_id: str, max_price: float, query: str, limit: int = 50, min_price: float = 0.0):
    session = _session_cache.get(conv_id)
    if not session: return []
    
    pool = session["raw_pool"]
    embeddings = session.get("pool_embeddings", [])
    
    # Detect query currency
    query_lower = query.lower()
    query_is_inr = any(sym in query_lower for sym in ['₹', 'rs', 'inr', 'rupee'])
    query_is_usd = any(sym in query_lower for sym in ['$', 'usd', 'dollar'])
    if not query_is_inr and not query_is_usd:
        # Default to USD if max_price < 500, else INR
        query_is_usd = (max_price < 500)
        query_is_inr = not query_is_usd

    # 1. Price Filter
    price_filtered = []
    filtered_embeddings = []
    for i, p in enumerate(pool):
        price_str = str(p.get("price", "")).lower()
        val = kimi_service._parse_price(price_str)
        
        # Detect product currency
        prod_is_inr = any(sym in price_str for sym in ['₹', 'rs', 'inr', 'rupee'])
        prod_is_usd = any(sym in price_str for sym in ['$', 'usd', 'dollar'])
        if not prod_is_inr and not prod_is_usd:
            prod_is_usd = (val < 500)
            prod_is_inr = not prod_is_usd

        # Align product value to query currency
        aligned_val = val
        if query_is_inr and prod_is_usd:
            aligned_val = val * 83.0
        elif query_is_usd and prod_is_inr:
            aligned_val = val / 83.0

        if min_price <= aligned_val <= max_price:
            price_filtered.append(p)
            if i < len(embeddings):
                filtered_embeddings.append(embeddings[i])
            else:
                filtered_embeddings.append(None)
                
    if not price_filtered:
        return []
        
    # 2. Rating & Review Mathematical Sort
    query_lower = query.lower()
    is_rating_sort = any(word in query_lower for word in ["rating", "ratings", "star", "stars", "best rated", "top rated"])
    is_review_sort = any(word in query_lower for word in ["review", "reviews", "feedback", "most reviewed", "popular"])
    
    if is_rating_sort or is_review_sort:
        def _parse_rating_count(val):
            if not val: return 0
            val_str = str(val).lower().replace(",", "")
            if 'k' in val_str:
                try: return int(float(val_str.replace('k', '')) * 1000)
                except: pass
            if 'm' in val_str:
                try: return int(float(val_str.replace('m', '')) * 1000000)
                except: pass
            digits = re.sub(r'\D', '', val_str)
            return int(digits) if digits else 0

        if is_rating_sort:
            print(f"⭐ Mathematical sorting by rating requested", flush=True)
            price_filtered.sort(
                key=lambda x: (
                    float(x.get("rating_avg") or 0.0),
                    _parse_rating_count(x.get("rating_count"))
                ),
                reverse=True
            )
        elif is_review_sort:
            print(f"💬 Mathematical sorting by reviews count requested", flush=True)
            price_filtered.sort(
                key=lambda x: _parse_rating_count(x.get("rating_count")),
                reverse=True
            )
        return price_filtered[:limit]

    # 3. If semantic search is available and query is not empty
    import re
    query_clean = re.sub(r'(?i)(?:under|below|less than|less|budget of|within|max|maximum)\s*(?:[\$₹\u20b9£€]|rs\.?|inr)?\s*([\d,]+\.?\d*)', '', query).strip()
    # Strip common conversational filler words so pure price filters don't trigger random semantic matching
    query_clean = re.sub(r'(?i)\b(?:show|me|the|ones|those|which|are|filter|only|cheapest|cheaper|from|these|among|them|any|have|price|less|more|than)\b', '', query_clean).strip()
    
    if session_embedder and query_clean and len(filtered_embeddings) == len(price_filtered) and all(e is not None for e in filtered_embeddings):
        query_emb = session_embedder.encode([query_clean])[0]
        scored = []
        for i, p in enumerate(price_filtered):
            score = cosine_similarity_np(query_emb, filtered_embeddings[i])
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Apply a basic threshold or just take top results
        return [p for s, p in scored if s > 0.1][:limit]
    
    # Fallback to simple keyword match if no embeddings
    if query_clean:
        keywords = set(re.findall(r'\b\w+\b', query_clean.lower()))
        def _score(p):
            text = f"{p.get('name','')} {p.get('brand','')} {p.get('details','')} {p.get('category','')}".lower()
            return sum(1 for k in keywords if k in text)
        scored = [(_score(p), p) for p in price_filtered]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored if s > 0][:limit]

    return price_filtered[:limit]

@app.post("/api/chat")
async def chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        body = await req.json()
        print(f"📥 Body received", flush=True)

        query = body.get("message")
        messages_list = body.get("messages", [])
        conv_id = body.get("id") or "default_session"
        
        # Pagination override
        if query == "__load_more__":
            session = _session_cache.get(conv_id)
            if not session:
                return JSONResponse({"type": "message", "response": "No active search session found.", "intent": "shopping"})
            
            offset = session["display_offset"]
            pool = session["raw_pool"]
            
            if offset >= len(pool):
                return JSONResponse({"type": "message", "response": "You've seen all available results for this search.", "intent": "shopping"})
                
            next_batch = pool[offset:offset+PAGE_SIZE]
            session["display_offset"] += len(next_batch)
            
            bot_response = f"Here are {len(next_batch)} more options.\n\n<product_grid>{json.dumps(next_batch)}</product_grid>"
            return JSONResponse({"type": "message", "response": bot_response, "intent": "shopping"})

        if not query and messages_list:
            last = messages_list[-1]
            query = last.get("content") or ""
            if not query and "parts" in last:
                query = " ".join([p.get('text', '') for p in last['parts']])

        query = (query or "hi").strip()
        query_lower = query.lower()
        print(f"🔥 Query: {query}", flush=True)

        intent = kimi_service.detect_intent(query)
            
        # ── Contextual RAG Step ────────────────────────────────────────────────
        # If the user asks something like "under $100" after "kids shoes", 
        # we expand it to "kids shoes under $100".
        if intent == "shopping" and messages_list:
            query = await kimi_service.rewrite_query_contextual(query, messages_list)
            query_lower = query.lower()

        print(f"🧠 Intent: {intent}", flush=True)

        live_products = []
        local_results = []
        bot_response = ""

        # Route by intent
        if any(x in query_lower for x in ["image", "photo", "pic", "picture", "images"]):
            img_res = await kimi_service.search_images(query)
            live_products = img_res.get("results", [])
            bot_response = f"Here are some images for **{query}**."

        elif intent == "vehicle":
            v_res = await kimi_service.get_vehicle_data(query)
            if isinstance(v_res, dict):
                live_products = [v_res]
                bot_response = ""
            else:
                bot_response = v_res

        elif intent == "shopping":
            import re as _re

            _max_price = float('inf')
            _min_price = 0.0
            
            # Enhanced price extraction
            _max_match = _re.search(r'(?i)(?:under|below|less than|less|budget of|within|max|maximum|at most)\s*(?:[\$₹\u20b9£€]|rs\.?|inr)?\s*([\d,]+\.?\d*)', query_lower)
            if _max_match:
                try:
                    _max_price = float(_max_match.group(1).replace(',', ''))
                    print(f"💰 Detected max price limit: {_max_price}", flush=True)
                except:
                    pass

            _min_match = _re.search(r'(?i)(?:above|over|more than|more|at least|min|minimum)\s*(?:[\$₹\u20b9£€]|rs\.?|inr)?\s*([\d,]+\.?\d*)', query_lower)
            if _min_match:
                try:
                    _min_price = float(_min_match.group(1).replace(',', ''))
                    print(f"💰 Detected min price limit: {_min_price}", flush=True)
                except:
                    pass

            is_followup = _is_followup(query, conv_id)
            
            if is_followup:
                print(f"♻️ Follow-up detected! Filtering session pool...", flush=True)
                pool_results = _filter_pool(conv_id, _max_price, query, limit=PAGE_SIZE, min_price=_min_price)
                
                if len(pool_results) >= RESULT_THRESHOLD:
                    print(f"✅ Found {len(pool_results)} items in pool. Serving instantly.", flush=True)
                    live_products = pool_results
                    session = _session_cache[conv_id]
                    session["display_offset"] = len(live_products)
                    bot_response = f"Here are the options from our current search matching **{query}**:"
                else:
                    # Tier 2: Augment pool
                    print(f"⚠️ Pool too thin ({len(pool_results)}). Augmenting via live API...", flush=True)
                    try:
                        last_topic = _session_cache[conv_id]["last_topic"]
                        # Clean conversational parts
                        import re as _re
                        q_clean = _re.sub(r'(?i)(show me|the ones|which ones|above|below|under|over|more than|less than|less|more|max|min|budget|within|between|[\d,.]+)', '', query).strip()
                        aug_query = f"{last_topic} {q_clean}".strip()
                        if aug_query:
                            print(f"🔄 Augmenting session {conv_id} with query: {aug_query}", flush=True)
                            new_live = await kimi_service.get_fast_bing_data(aug_query, num_results=60)
                            if new_live:
                                extend_session_pool(conv_id, new_live)
                                pool_results = _filter_pool(conv_id, _max_price, query, limit=PAGE_SIZE, min_price=_min_price)
                    except Exception as e:
                        print(f"Error augmenting pool: {e}")
                        
                    live_products = pool_results
                    if conv_id in _session_cache:
                        session = _session_cache[conv_id]
                        session["display_offset"] = len(live_products)
                    
                    if live_products:
                        bot_response = f"I dug deeper. Here are the best available options for **{query}**:"
                    else:
                        bot_response = f"I'm sorry, I couldn't find anything matching **{query}** even after a deeper search."
            
            else:
                # New topic
                print(f"🆕 New topic detected! Fetching fresh data...", flush=True)
                clear_session(conv_id)
                
                # 1. Bypass old static RAG database to ensure fresh live Amazon/Walmart data
                print(f"⚠️ Bypassing static RAG. Fetching Live API...", flush=True)
                
                # Append "products" if query is a single generic category word
                live_query = query.strip()
                if live_query.lower() in ["electronics", "mobiles", "fashion", "kids shoes"]:
                    live_query = f"{live_query} products"
                    
                try:
                    # Use the robust parallel scrapers (Amazon/Walmart/eBay/Flipkart)
                    live_products = await kimi_service.get_fast_bing_data(live_query, num_results=60)
                except Exception as e:
                    print(f"Live API error: {e}")
                    live_products = []

                if live_products:
                    # Apply price filters (only if a price constraint was specified)
                    if _max_price < float('inf') or _min_price > 0.0:
                        # Detect query currency
                        query_is_inr = any(sym in query_lower for sym in ['₹', 'rs', 'inr', 'rupee'])
                        query_is_usd = any(sym in query_lower for sym in ['$', 'usd', 'dollar'])
                        if not query_is_inr and not query_is_usd:
                            # If max price > 500, assume INR (phones/electronics in INR are typically 5000+)
                            query_is_inr = (_max_price > 500)
                            query_is_usd = not query_is_inr

                        def _align_price(raw_price):
                            """Returns (aligned_value, within_budget) for a product price."""
                            if not raw_price or any(w in str(raw_price) for w in ["Check", "Verifying", "Request", "check", "request"]):
                                return None, True  # No price → always keep
                            val = kimi_service._parse_price(raw_price)
                            if val == float('inf'):
                                return None, True  # Unparseable → always keep
                            price_str_lower = str(raw_price).lower()
                            prod_is_inr = any(sym in price_str_lower for sym in ['₹', 'rs', 'inr', 'rupee'])
                            prod_is_usd = any(sym in price_str_lower for sym in ['$', 'usd', 'dollar'])
                            if not prod_is_inr and not prod_is_usd:
                                prod_is_usd = (val < 500)
                                prod_is_inr = not prod_is_usd
                            aligned = val
                            if query_is_inr and prod_is_usd:
                                aligned = val * 84.0  # USD → INR
                            elif query_is_usd and prod_is_inr:
                                aligned = val / 84.0  # INR → USD
                            within = _min_price <= aligned <= _max_price
                            return aligned, within

                        # ── HARD FILTER (preserved, currently disabled) ──────────────────────
                        # Uncomment this block and comment out the SOFT FILTER block below
                        # to restore strict price filtering that removes all out-of-budget items.
                        #
                        # filtered_live = []
                        # for p in live_products:
                        #     _, within = _align_price(p.get("price", ""))
                        #     if within:
                        #         filtered_live.append(p)
                        # before_filter = len(live_products)
                        # live_products = filtered_live
                        # print(f"💰 Hard filter: {len(live_products)}/{before_filter} kept "
                        #       f"(max={_max_price}, {'INR' if query_is_inr else 'USD'})", flush=True)
                        # ────────────────────────────────────────────────────────────────────

                        # ── SOFT FILTER (active) ─────────────────────────────────────────────
                        # Step 1: Try the strict filter first
                        SOFT_MIN_RESULTS = 8  # Minimum products we want to show
                        strictly_filtered = [p for p in live_products
                                             if _align_price(p.get("price", ""))[1]]

                        before_filter = len(live_products)
                        if len(strictly_filtered) >= SOFT_MIN_RESULTS:
                            # Enough products pass the budget — use strict results
                            live_products = strictly_filtered
                            print(f"💰 Soft filter (strict): {len(live_products)}/{before_filter} kept "
                                  f"(max={_max_price}, {'INR' if query_is_inr else 'USD'})", flush=True)
                        else:
                            # Too few pass — FALLBACK: keep all products but sort by price closeness
                            # so the cheapest / most on-budget items appear first
                            print(f"💰 Soft filter FALLBACK: only {len(strictly_filtered)}/{before_filter} "
                                  f"in budget — showing all {before_filter} sorted by price closeness", flush=True)

                            def _price_closeness(p):
                                aligned, _ = _align_price(p.get("price", ""))
                                if aligned is None:
                                    return float('inf')  # no-price items go to the end
                                if aligned <= _max_price:
                                    return 0             # within budget → top of list
                                return aligned - _max_price  # over budget → sorted by overage amount

                            live_products = sorted(live_products, key=_price_closeness)
                        # ────────────────────────────────────────────────────────────────────
                    else:
                        # No price constraint — keep all products as-is
                        pass

                set_session_pool(conv_id, live_products, topic=live_query, query=live_query)
                if conv_id in _session_cache:
                    _session_cache[conv_id]["display_offset"] = len(live_products)
                if live_products:
                    bot_response = f"Here are the best options for **{query}**:"
                else:
                    bot_response = f"I couldn't find results for **{query}** right now. Please try again in a moment."

        else:
            bot_response = await chat_with_bot(
                query=query, live_context=[], intent_type=intent, local_docs=[]
            )

        # Build final response
        if live_products and intent in ("shopping", "images", "global_search", "supplier_sourcing", "vehicle"):
            # STRICT DB PRIORITY: Since we add DB results to live_products first, 
            # we simply use the original order to ensure they appear first in the UI.
            ordered = live_products
            
            # Build product grid - prefer images but allow placeholder for products without
            INVALID_IMG_STRS = {"not found", "null", "none", "n/a", "undefined"}
            items = []
            for p in ordered:
                img = p.get("s3_image_url") or p.get("image_url") or p.get("Image URL")
                img_str = str(img).lower() if img else ""
                
                # Only hard-skip obviously broken placeholder images
                if img and (img_str in INVALID_IMG_STRS or "placehold.co" in img_str or "no image" in img_str):
                    img = None  # Reset to None so we can use a generic placeholder
                
                # We allow items with no image — the frontend shows a fallback card
                if len(items) >= 50:
                    break

                # Parse reviews
                reviews = []
                if p.get("reviews"):
                    try:
                        if isinstance(p.get("reviews"), str):
                            reviews = json.loads(p.get("reviews"))
                        else:
                            reviews = p.get("reviews")
                    except:
                        pass

                def _guess_category(name, current_cat):
                    if current_cat and current_cat != "fashion": return current_cat
                    n = name.lower()
                    if any(w in n for w in ["tent", "mat", "camp", "outdoor", "sport", "yoga", "gym"]): return "sports-outdoors"
                    if any(w in n for w in ["phone", "laptop", "tech", "gadget", "earbud", "usb"]): return "electronics"
                    if any(w in n for w in ["toy", "doll", "kid", "baby", "toddler"]): return "baby-kids"
                    if any(w in n for w in ["home", "kitchen", "cook", "furniture"]): return "home-kitchen"
                    return current_cat or "fashion"

                items.append({
                    "name": p.get("name") or p.get("title") or "Product",
                    "brand": p.get("brand") or p.get("source") or "Store",
                    "price": kimi_service._extract_price_from_snippet(p.get("price")),
                    "image_url": img,
                    "category": _guess_category(p.get("name") or p.get("title") or "", p.get("category")),
                    "source_url": p.get("source_url") or p.get("url") or p.get("source"),
                    "source": p.get("source") or "Search",
                    "rating_avg": p.get("rating_avg") or p.get("rating") or "",
                    "rating_count": p.get("rating_count") or "",
                    "reviews": reviews,
                    "details": p.get("details") or p.get("description") or "",
                    "moq": p.get("moq") or None,
                    "location": p.get("location") or None,
                    "supplier_years": p.get("supplier_years") or None,
                    "is_verified": bool(p.get("is_verified") or False),
                })
            
            # FINAL DE-DUPLICATION
            # Strategy: deduplicate by URL path (strip query params so tracking params don't
            # create false uniqueness). For products with no proper URL, use name+source as key.
            final_items = []
            seen_srcs = set()
            for item in items:
                src_url = item["source_url"]
                # Build de-dup key
                if src_url and isinstance(src_url, str) and src_url.startswith("http"):
                    try:
                        _p = urlparse(src_url)
                        # Use scheme+netloc+path as the key (strips tracking/search query params)
                        dedup_key = f"{_p.scheme}://{_p.netloc}{_p.path}".rstrip("/")
                        # If the URL is just a generic search page (scraper didn't get specific product URL),
                        # we must append the product name so we don't collapse all products from that store into 1.
                        if _p.path in ["", "/", "/s", "/search", "/results"] or len(_p.path) < 10:
                            dedup_key = f"{dedup_key}::__name__{item.get('name', '')}"
                    except Exception:
                        dedup_key = src_url
                elif item.get("name"):
                    # No valid URL — use source + name combo as a stable identity key
                    dedup_key = f"__name__{item.get('source', '')}::{item.get('name', '')}"
                else:
                    dedup_key = None  # Cannot deduplicate, just include it
                
                if dedup_key and dedup_key in seen_srcs:
                    continue
                final_items.append(item)
                if dedup_key:
                    seen_srcs.add(dedup_key)
            
            print(f"📡 Grid: {len(final_items)} products sent to UI (from {len(items)} before dedup)", flush=True)
            grid = f"<product_grid>{json.dumps(final_items)}</product_grid>"
            
            # Format text response and append the product grid
            if bot_response:
                final = f"{bot_response}\n\n{grid}"
            else:
                final = grid
        else:
            final = format_response(bot_response)
            print(f"📡 Text response: {len(final)} chars", flush=True)

        print(f"✅ Done in {time.time()-start_time:.1f}s", flush=True)
        return JSONResponse({"type": "message", "response": final, "intent": intent})

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
