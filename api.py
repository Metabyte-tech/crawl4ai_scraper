from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
from urllib.parse import urlparse

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
    if isinstance(res, dict) and res.get("type") == "images":
        results = res.get("results", [])
        return f"\n\n<product_carousel>{json.dumps(results)}</product_carousel>\n\n"
    if isinstance(res, list):
        if not res:
            return "No products found."
        return "\n\n".join([
            f"🛍️ {p.get('name', 'Product')} - {p.get('price', '')}\n🔗 {p.get('url', p.get('source_url', ''))}"
            for p in res if isinstance(p, dict)
        ])
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
            print(f"Carousel reconstruct error: {e}")
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
        print(f"Background ingest error for {url}: {e}")


async def background_crawl_and_ingest(query: str, fast_products: list):
    try:
        print(f"🔄 BACKGROUND: Deep crawl for '{query}'...")
        deep_results = await kimi_service.run_deep_crawl_process(query, fast_products)
        if deep_results:
            await kimi_service.cache_and_store_products(deep_results, query)
        print(f"✅ BACKGROUND: Done for '{query}'!")
    except Exception as e:
        print(f"❌ BACKGROUND: Failed for '{query}': {e}")
        import traceback
        traceback.print_exc()


app = FastAPI(title="Retail AI RAG API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🔍 {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"📉 {request.method} {request.url.path} → {response.status_code}")
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
async def crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    background_tasks.add_task(background_ingest, request.url, max_pages=1)
    return {"status": "success", "message": f"Ingestion started for {request.url}"}


@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    background_tasks.add_task(background_ingest, request.url, max_pages=15)
    return {"status": "success", "message": "Deep ingestion started"}


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
                "price": meta.get("price") or meta.get("Price") or "",
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
        print(f"📥 Body received")

        query = body.get("message")
        messages_list = body.get("messages", [])
        if not query and messages_list:
            last = messages_list[-1]
            query = last.get("content") or ""
            if not query and "parts" in last:
                query = " ".join([p.get('text', '') for p in last['parts']])

        query = (query or "hi").strip()
        query_lower = query.lower()
        print(f"🔥 Query: {query}")

        intent = kimi_service.detect_intent(query)
        if body.get("template_id"):
            intent = "agent_task"
            
        print(f"🧠 Intent: {intent}")

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
            bot_response = format_response(v_res)

        elif intent == "agent_task":
            bot_response = await kimi_service.generate_agent_report(
                query=query, 
                template_id=body.get("template_id"),
                subject=body.get("subject")
            )
            # For Agent tasks, we return the raw text report
            return JSONResponse({"type": "message", "response": bot_response, "intent": intent})

        elif intent == "shopping":
            rag_start = time.time()
            rag_results = fast_query(query, category="retail", threshold=1.2)
            print(f"🛒 RAG: {len(rag_results)} docs in {time.time()-rag_start:.2f}s")
            random.shuffle(rag_results)

            results_with_images = [
                r for r in rag_results
                if r[0].metadata.get("image_url") or r[0].metadata.get("s3_image_url")
            ]

            # Keyword check
            keywords = [w for w in query_lower.split() if len(w) > 2]
            for word in keywords:
                hit = any(
                    word in (str(r[0].page_content) + str(r[0].metadata.get("name", ""))).lower()
                    for r in results_with_images
                )
                if not hit:
                    print(f"⚠️ RAG rejected: '{word}' not in results")
                    results_with_images = []
                    rag_results = []
                    break

            local_results = rag_results

            if len(results_with_images) < 6:
                # Fast search takes 1-2s. Deep crawling happens in background
                print(f"DEBUG: Triggering live search (RAG only found {len(results_with_images)} docs)")
                live_products = await kimi_service.get_fast_bing_data(query)
                print(f"⚡ Bing: {len(live_products)} products")
                if live_products:
                    background_tasks.add_task(background_crawl_and_ingest, query, live_products)
            else:
                live_products = [r[0].metadata for r in results_with_images]

            bot_response = await chat_with_bot(
                query=query, live_context=live_products,
                intent_type=intent, local_docs=local_results
            )

        else:
            bot_response = await chat_with_bot(
                query=query, live_context=[], intent_type=intent, local_docs=[]
            )

        # Build final response
        if live_products and intent in ("shopping", "images"):
            ordered = sorted(
                live_products,
                key=lambda p: 0 if (p.get("image_url") or p.get("s3_image_url")) else 1
            )
            items = []
            for p in ordered[:10]:
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
                    "price": p.get("price") or "Check Site",
                    "image_url": p.get("s3_image_url") or p.get("image_url"),
                    "source_url": p.get("source_url") or p.get("url"),
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
            carousel = f"<product_carousel>{json.dumps(items)}</product_carousel>"
            final = f"Here are the best results I found:\n\n{carousel}"
            print(f"📡 Carousel: {len(items)} products")
        else:
            final = format_response(bot_response)
            print(f"📡 Text response: {len(final)} chars")

        print(f"✅ Done in {time.time()-start_time:.1f}s")
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
