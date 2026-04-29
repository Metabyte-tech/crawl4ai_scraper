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
from crawler import crawl_site, crawl_site_recursive
from ingest import add_content_to_store, add_multiple_contents_to_store
from vector_store import clear_vector_store
from query import fast_query
from bot import chat_with_bot
from kimi_service import kimi_service
from arq import create_pool
from arq.connections import RedisSettings
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


async def background_ingest(url: str, max_pages: int = 1):
    try:
        if max_pages <= 1:
            content, _ = await crawl_site(url)
            if content and len(content.strip()) > 10:
                await add_content_to_store(content, {"source": url})
                update_last_domain(url)
        else:
            results = await crawl_site_recursive(url, max_pages=max_pages)
            if results:
                await add_multiple_contents_to_store(results)
    except Exception as e:
        print(f"Background ingest error for {url}: {e}", flush=True)


async def background_crawl_and_ingest(query: str, fast_products: list):
    try:
        print(f"🔄 BACKGROUND: Deep crawl for '{query}'...", flush=True)
        deep_results = await kimi_service.run_deep_crawl_process(query, fast_products)
        if deep_results:
            await kimi_service.cache_and_store_products(deep_results, query)
        print(f"✅ BACKGROUND: Done for '{query}'!", flush=True)
    except Exception as e:
        print(f"❌ BACKGROUND: Failed for '{query}': {e}", flush=True)
        import traceback
        traceback.print_exc()


app = FastAPI(title="Retail AI RAG API")

@app.on_event("startup")
async def startup():
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    print(f"📡 API attempting to connect to Redis at: {redis_url}", flush=True)
    try:
        app.state.arq_pool = await create_pool(RedisSettings.from_dsn(redis_url))
        # Test connection
        await app.state.arq_pool.set('api_health_check', 'ok')
        print("🚀 ARQ Redis Pool initialized and verified", flush=True)
        
        # Initialize PostgreSQL
        await db_service.init_db()
        print("💾 PostgreSQL History DB initialized", flush=True)
    except Exception as e:
        print(f"❌ ARQ Redis Pool failed to initialize: {e}", flush=True)
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown():
    await app.state.arq_pool.close()
    print("💤 ARQ Redis Pool closed", flush=True)


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


class CrawlBatchRequest(BaseModel):
    urls: List[str]



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
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    # Enqueue the job instead of running it in the API process
    # to avoid concurrent SQLite/Chroma locking issues between API and Worker.
    await req.app.state.arq_pool.enqueue_job('ingest_url_task', url=request.url, max_pages=1)
    return {"status": "success", "message": f"Ingestion queued for {request.url}"}


@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest, req: Request):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    # Offload to worker
    await req.app.state.arq_pool.enqueue_job('deep_crawl_task', query=request.url, fast_products=[])
    return {"status": "success", "message": "Deep ingestion started"}


# ── Batch endpoints (multiple URLs at once) ──────────────────────────────────

@app.post("/crawl/batch")
async def crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    """Light crawl for multiple URLs. URLs are processed concurrently by the worker."""
    valid = [u for u in request.urls if u.startswith("http")]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid URLs provided (must start with http/https)")

    for url in valid:
        await req.app.state.arq_pool.enqueue_job('ingest_url_task', url=url, max_pages=1)

    return {
        "status": "success",
        "message": f"Ingestion queued for {len(valid)} URL(s)",
        "urls": valid,
    }


@app.post("/crawl/deep/batch")
async def deep_crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    """Deep crawl for multiple URLs. Offloads to background worker for deep recursive crawling."""
    valid = [u for u in request.urls if u.startswith("http")]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid URLs provided")

    for url in valid:
        await req.app.state.arq_pool.enqueue_job('ingest_url_task', url=url, max_pages=100000)

    return {"status": "success", "message": f"Deep ingestion queued for {len(valid)} URL(s)", "urls": valid}


# ── Admin endpoints (tracked batch crawls) ───────────────────────────────────

@app.post("/admin/crawl/batch")
async def admin_crawl_batch_endpoint(request: CrawlBatchRequest, req: Request):
    """Deep crawl for multiple URLs with admin-level status tracking."""
    valid = [u for u in request.urls if u.startswith("http")]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid URLs provided")

    batch_id = await admin_service.start_batch(valid)
    
    for url in valid:
        # Enqueue with higher priority if needed, or just same pool
        await req.app.state.arq_pool.enqueue_job(
            'admin_ingest_url_task', 
            batch_id=batch_id, 
            url=url, 
            max_pages=100000
        )

    return {
        "status": "success", 
        "message": f"Tracked batch crawl started for {len(valid)} URL(s)", 
        "batch_id": batch_id,
        "urls": valid
    }

@app.get("/admin/crawl/batches")
async def list_admin_batches():
    """Returns a list of recent admin crawl batches."""
    batches = await admin_service.list_recent_batches()
    return {"batches": batches}

@app.get("/admin/crawl/batch/{batch_id}")
async def get_admin_batch_status(batch_id: str):
    """Returns the detailed status of a specific crawl batch."""
    status = await admin_service.get_batch_status(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")
    return status

@app.delete("/admin/crawl/batch/{batch_id}")
async def delete_admin_batch(batch_id: str):
    await admin_service.delete_batch(batch_id)
    return {"status": "deleted", "batch_id": batch_id}

# ── History endpoints (Permanent storage) ────────────────────────────────────

@app.get("/admin/crawl-history")
async def list_crawl_history(limit: int = 20):
    """Returns a list of historical crawl batches from PostgreSQL."""
    batches = await db_service.list_batches(limit)
    return {"batches": batches}

@app.get("/admin/crawl-history/{batch_id}")
async def get_crawl_history_detail(batch_id: str):
    """Returns full detail for a historical batch from PostgreSQL."""
    detail = await db_service.get_batch_detail(batch_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Historical batch not found")
    return detail


@app.post("/clear")
async def clear_endpoint():
    clear_vector_store()
    return {"status": "success", "message": "Memory cleared successfully"}

@app.get("/api/categories")
async def get_categories():
    # Use more specific search terms to avoid vector overlap (e.g. 'toys' matching 'kids shoes')
    category_queries = {
        "Toys": "children's toys, games, and play sets",
        "Clothes": "fashion clothing, apparel, shirts, and pants",
        "Shoes": "footwear, sneakers, boots, and sandals",
        "Laptops": "laptops, notebooks, and computing hardware",
        "Mobiles": "smartphones, mobile phones, and cellular devices"
    }
    result = []
    
    for display_name, query in category_queries.items():
        docs_scores = fast_query(query, k=50) # Get enough to filter out items without images
        
        items = []
        seen_urls = set()
        
        for doc, score in docs_scores:
            meta = doc.metadata
            url = meta.get("url") or meta.get("source_url") or meta.get("source") or ""
            
            if url in seen_urls:
                continue
                
            img = meta.get("image_url") or meta.get("s3_image_url") or meta.get("Image URL")
            if not img or not img.startswith("http"):
                # Use a beautiful placeholder if image is missing so the list isn't empty
                img = f"https://placehold.co/600x600?text={display_name}+Item"
            
            # Aggressive black-list for Toys to avoid shoes/clothes overlap
            product_name = (meta.get("name") or meta.get("title") or "").lower()
            product_content = (doc.page_content or "").lower()
            
            if display_name == "Toys":
                # Use a Whitelist for Toys because generic names like "Product Option 1" bypass blacklists
                toy_white_list = [
                    "toy", "game", "play", "puzzle", "doll", "lego", "figure", "hobby", "rc ", 
                    "remote control", "plush", "stuffed", "car", "vehicle", "racing", "track", 
                    "wheels", "ride-on", "bike", "nerf", "barbie", "hot wheels", "blocks",
                    "sorting", "stacking", "activity", "center", "learning", "educational",
                    "preschool", "toddler", "baby", "robot", "kit", "squishy", "slime",
                    "math", "science", "steam", "stem", "anatomy", "chemistry", "physics", "experiment"
                ]
                is_toy = any(word in product_name for word in toy_white_list) or \
                         any(word in product_content for word in toy_white_list)
                
                # Also block obviously wrong things that might have "play" or "game" in content (like shoes/boots)
                shoe_black_list = ["shoe", "boot", "sneaker", "nike", "adidas", "puma", "footwear", "sandal", "heel"]
                if not is_toy or any(word in product_name for word in shoe_black_list):
                    continue
                
                # Block generic titles
                if "product option" in product_name:
                    continue
                
            # Parse reviews if they are stored as JSON string
            reviews = []
            if meta.get("reviews"):
                try:
                    reviews = json.loads(meta.get("reviews"))
                except:
                    pass

            items.append({
                "name": meta.get("name") or meta.get("title") or "Unnamed Product",
                "price": kimi_service._extract_price_from_snippet(meta.get("price") or meta.get("Price") or ""),
                "url": url,
                "image_url": img,
                "score": float(score),
                "brand": meta.get("brand") or "Product",
                "rating_avg": meta.get("rating_avg") or meta.get("rating") or "",
                "rating_count": meta.get("rating_count") or "",
                "reviews": reviews,
                "details": meta.get("details") or meta.get("description") or ""
            })
            seen_urls.add(url)
            
            if len(items) >= 5: # Top 5 distinct products per category
                break
                
        result.append({
            "name": display_name,
            "items": items
        })
        
    return JSONResponse(content={"categories": result})


@app.get("/api/templates")
async def get_templates():
    """Returns the library of expert templates and categories."""
    template_path = "templates.json"
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return json.load(f)
    return {"categories": []}

@app.post("/api/plan")
async def generate_plan(request: Request):
    """Generates a multi-step execution plan for a business query."""
    data = await request.json()
    query = data.get("query")
    template_id = data.get("template_id")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
        
    plan = await kimi_service.generate_execution_plan(query, template_id)
    return {"plan": plan}


@app.post("/api/chat")
async def chat_endpoint(req: Request, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        body = await req.json()
        print(f"📥 Body received", flush=True)

        query = body.get("message")
        messages_list = body.get("messages", [])
        if not query and messages_list:
            last = messages_list[-1]
            query = last.get("content") or ""
            if not query and "parts" in last:
                query = " ".join([p.get('text', '') for p in last['parts']])

        query = (query or "hi").strip()
        query_lower = query.lower()
        print(f"🔥 Query: {query}", flush=True)

        intent = kimi_service.detect_intent(query)
        if body.get("template_id"):
            intent = "agent_task"
            
        print(f"🧠 Intent: {intent}", flush=True)

        live_products = []
        local_results = []
        bot_response = ""

        # Route by intent
        if any(x in query_lower for x in ["image", "photo", "pic", "picture", "show me", "images"]):
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

        elif intent == "agent_task":
            template_id = body.get("template_id", "")
            subject = body.get("subject") or query
            
            # Start report generation
            report_task = kimi_service.generate_agent_report(
                query=query, 
                template_id=template_id,
                subject=subject
            )
            
            products_task = None
            # If template implies tangible physical products, design, or sourcing, fetch images/products concurrently
            if any(x in template_id for x in ["design", "source", "product", "inclusive", "appeal", "trend", "validate", "search", "find", "bestseller", "investigate"]):
                if "supplier" in template_id or "source" in template_id or "search" in template_id or "bestseller" in template_id:
                    products_task = kimi_service.get_fast_bing_data(subject)
                else:
                    products_task = kimi_service.search_images(subject)
            
            if products_task:
                bot_response, prod_res = await asyncio.gather(report_task, products_task)
                if isinstance(prod_res, dict) and "results" in prod_res:
                    live_products = prod_res["results"]
                elif isinstance(prod_res, list):
                    live_products = prod_res
            else:
                bot_response = await report_task

        elif intent == "shopping":
            # ── Step 1: Query DB first (max recall) ──────────────────────────────
            try:
                rag_results = fast_query(query, threshold=2.0, k=100)
            except Exception as e:
                print(f"RAG search error: {e}", flush=True)
                rag_results = []

            # ── Relevance filter ──────────────────────────────────────────────────
            import re as _re

            _stop = {
                'the','and','for','with','are','this','that','from','have','has',
                'its','not','of','in','on','at','by','be','a','an','to','do',
                'is','it','me','my','we','us','or','so','if','as','up','go',
            }

            # Extract meaningful keywords — allow 2-char terms so 'rc' counts
            _raw_kw = [w for w in _re.split(r'\W+', query.lower()) if len(w) >= 2 and w not in _stop]

            # Simple plural/singular normalisation: strip trailing 's' for matching
            def _stem(w: str) -> str:
                return w.rstrip('s') if len(w) > 3 else w

            _query_keywords = list(dict.fromkeys(_raw_kw))  # deduplicated, preserves order

            def _word_match(keyword: str, text: str) -> bool:
                """True if the keyword (or its stem) appears as a whole word in text."""
                kw = _re.escape(keyword)
                st = _re.escape(_stem(keyword))
                pattern = rf'\b({kw}|{st})\b'
                return bool(_re.search(pattern, text))

            def _score(text: str) -> int:
                t = text.lower()
                return sum(1 for kw in _query_keywords if _word_match(kw, t))

            def _is_relevant(haystack: str) -> bool:
                if not _query_keywords:
                    return True
                n = len(_query_keywords)
                # For 1–2 keyword queries, ALL must match (no slack)
                # For 3+ keyword queries, require ceil(67%) to match
                if n <= 2:
                    required = n
                else:
                    required = max(2, -(-n * 2 // 3))  # ceiling division of 2/3
                return _score(haystack) >= required

            def _candidate_relevant(name: str, details: str = "") -> bool:
                if not name:
                    return False
                # Try kimi_service's own validator first
                try:
                    if kimi_service._validate_product_relevance(name.strip(), query)[0]:
                        return True
                except Exception:
                    pass
                return _is_relevant(f"{name} {details}")

            if _query_keywords:
                _filtered = [
                    (_doc, _sc) for _doc, _sc in rag_results
                    if _is_relevant(
                        f"{_doc.metadata.get('name','')} "
                        f"{_doc.metadata.get('title','')}"
                    )
                ]
                # Fall back to page_content match if name/title filter removes everything
                if not _filtered:
                    _filtered = [
                        (_doc, _sc) for _doc, _sc in rag_results
                        if _is_relevant(_doc.page_content[:500])
                    ]
                if _filtered:
                    rag_results = _filtered

            # Process RAG results — collapse chunks from the same URL into one product
            cached_products = []
            seen_urls = set()
            seen_names = set()

            for doc, score in rag_results:
                meta = doc.metadata
                img = meta.get("s3_image_url") or meta.get("image_url")
                if not img or not str(img).startswith("http"):
                    continue
                url = meta.get("source") or meta.get("source_url")
                original_name = str(meta.get("name") or "Product").strip()

                is_generic = '|' in original_name or len(original_name) <= 15 or "toys" in original_name.lower().split()
                extracted_sub_products = False

                if is_generic and doc.page_content:
                    import re
                    links = re.findall(r'\[([^\]]{5,100})\]\((https?://[^\s\)]+)\)', doc.page_content)
                    for link_text, link_url in links:
                        lt_lower = link_text.lower()
                        if any(x in lt_lower for x in ['home', 'contact', 'about', 'login', 'cart', 'checkout', 'privacy', 'policy', 'terms', 'subscribe', 'search']):
                            continue
                        if not _candidate_relevant(link_text, doc.page_content):
                            continue
                        norm_sub_url = kimi_service._normalize_url(link_url)
                        if norm_sub_url in seen_urls:
                            continue
                        seen_urls.add(norm_sub_url)
                        price = kimi_service._extract_price_from_snippet(doc.page_content)
                        seen_names.add(link_text.strip().lower())
                        cached_products.append({
                            "name": link_text.strip(),
                            "price": price,
                            "source_url": link_url,
                            "image_url": img,
                            "brand": meta.get("brand") or original_name[:15],
                            "source": meta.get("store_source") or "Cached",
                            "rating_avg": meta.get("rating_avg"),
                            "rating_count": meta.get("rating_count"),
                            "details": f"From {original_name}",
                            "reviews": meta.get("reviews")
                        })
                        extracted_sub_products = True

                if not extracted_sub_products:
                    if not _candidate_relevant(original_name, doc.page_content):
                        continue
                    norm_url = kimi_service._normalize_url(url)
                    if norm_url in seen_urls:
                        continue
                    seen_urls.add(norm_url)
                    seen_names.add(original_name.lower())
                    cached_products.append({
                        "name": original_name,
                        "price": meta.get("price"),
                        "source_url": url,
                        "image_url": img,
                        "brand": meta.get("brand"),
                        "source": meta.get("store_source") or "Cached",
                        "rating_avg": meta.get("rating_avg"),
                        "rating_count": meta.get("rating_count"),
                        "details": meta.get("details") or meta.get("description"),
                        "reviews": meta.get("reviews")
                    })

                if len(cached_products) >= 20:
                    break

            # ── Step 2: Gap-fill with Kimi for whatever DB is missing (max 20 total) ──
            new_live_products = []

            if len(cached_products) < 20:
                needed = 20 - len(cached_products)
                print(f"⚡ DB:{len(cached_products)} → gap-filling {needed} from Kimi", flush=True)
                live_results = await kimi_service.get_fast_bing_data(query)
                count = 0
                for p in live_results:
                    if count >= needed:
                        break
                    name = (p.get("name") or p.get("title", "")).strip()
                    name_lower = name.lower()
                    url = p.get("source_url") or p.get("url") or p.get("source")
                    norm_url = kimi_service._normalize_url(url) if url else None

                    # Skip if already seen
                    if name_lower in seen_names or norm_url in seen_urls:
                        continue

                    # Skip if Kimi result is irrelevant to the query
                    if not _candidate_relevant(name, str(p.get('details',''))):
                        continue

                    new_live_products.append(p)
                    seen_names.add(name_lower)
                    if norm_url:
                        seen_urls.add(norm_url)
                    count += 1
            else:
                print(f"⚡ DB full ({len(cached_products)} items) — skipping Kimi", flush=True)

            # ── Step 3: Combine ───────────────────────────────────────────────────
            live_products = cached_products + new_live_products
            print(f"🚀 {len(cached_products)} DB + {len(new_live_products)} Kimi = {len(live_products)} total", flush=True)

            # Background enrichment - OFFLOAD TO REDIS WORKER
            if live_products:
                await req.app.state.arq_pool.enqueue_job('cache_products_task', products=live_products, query=query)
                await req.app.state.arq_pool.enqueue_job('deep_crawl_task', query=query, fast_products=live_products)

            # For simple shopping with no template: skip the LLM completely for speed
            # The carousel already shows everything the user needs
            if live_products:
                bot_response = f"Here are the best options for **{query}**:"
            else:
                bot_response = f"I couldn't find results for **{query}** right now. Please try again in a moment."

        else:
            bot_response = await chat_with_bot(
                query=query, live_context=[], intent_type=intent, local_docs=[]
            )

        # Build final response
        if live_products and intent in ("shopping", "images", "global_search", "supplier_sourcing", "agent_task", "vehicle"):
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
                if len(items) >= 20:
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

                items.append({
                    "name": p.get("name") or p.get("title") or "Product",
                    "brand": p.get("brand") or p.get("source") or "Store",
                    "price": kimi_service._extract_price_from_snippet(p.get("price")),
                    "image_url": img,
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
            final_items = []
            seen_srcs = set()
            for item in items:
                u = kimi_service._normalize_url(item["source_url"])
                # We strictly enforce one card per unique URL so the same product is never repeated
                if u in seen_srcs:
                    continue
                final_items.append(item)
                seen_srcs.add(u)
            
            grid = f"<product_grid>{json.dumps(final_items)}</product_grid>"
            
            # Format text response and append the product grid
            if bot_response:
                final = f"{bot_response}\n\n{grid}"
            else:
                final = grid
                
            print(f"📡 Grid: {len(items)} products", flush=True)
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
