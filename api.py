from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from crawler import crawl_site, crawl_site_recursive
from ingest import add_content_to_store, add_multiple_contents_to_store
from vector_store import clear_vector_store
from query import fast_query
from bot import chat_with_bot
from retail_crawler import retail_crawler
from kimi_service import kimi_service
from urllib.parse import urlparse

# Track last crawled domain
last_crawled_domain = None


def update_last_domain(url):
    global last_crawled_domain
    try:
        domain = urlparse(url).netloc
        if domain:
            last_crawled_domain = domain
            print(f"DEBUG: Updated last_crawled_domain to: {last_crawled_domain}")
    except Exception:
        pass


# ✅ FORMAT RESPONSE FOR FRONTEND (VERY IMPORTANT)
def format_response(res):
    if isinstance(res, str):
        return res

    # 🚗 Vehicle formatting
    if isinstance(res, dict) and not res.get("type") and "price" in res:
        return f"""
🚗 {res.get('name', 'Vehicle')}
💰 Price: {res.get('price', 'N/A')}
⛽ Mileage: {res.get('mileage', 'N/A')}
🔋 Fuel: {res.get('fuel', 'N/A')}
"""

    # 🖼️ Images (Horizontal Carousel)
    if isinstance(res, dict) and res.get("type") == "images":
        import json
        results = res.get("results", [])
        carousel_json = json.dumps(results)
        # Use newlines to ensure markdown renderer treats this as a block
        return f"\n\n<product_carousel>\n{carousel_json}\n</product_carousel>\n\n"

    # 🛒 Products
    if isinstance(res, list):
        if not res: return "No products found."
        return "\n\n".join([
            f"🛍️ {p.get('name', 'Product')} - {p.get('price', '')}\n🔗 {p.get('url', p.get('source_url', ''))}"
            for p in res if isinstance(p, dict)
        ])

    return str(res)


async def background_ingest(url: str, max_pages: int = 1):
    try:
        print(f"Background ingestion started for: {url} (max_pages={max_pages})")
        if max_pages <= 1:
            content, _ = await crawl_site(url)
            if content and len(content.strip()) > 10:
                await add_content_to_store(content, {"source": url})
                update_last_domain(url)
        else:
            results = await crawl_site_recursive(url, max_pages=max_pages)
            if results:
                await add_multiple_contents_to_store(results)
                print(f"Background deep ingestion complete for {url}")
    except Exception as e:
        print(f"Error in background ingestion for {url}: {e}")


app = FastAPI(title="Retail AI RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrawlRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    message: str


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


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        query = request.message.strip()
        query_lower = query.lower()
        print(f"\n🔥 Query: {query}")

        # -------------------------------
        # 🧠 INTENT DETECTION
        # -------------------------------
        intent = kimi_service.detect_intent(query_lower)
        print(f"🧠 Intent: {intent} (Took {time.time() - start_time:.2f}s)")

        # -------------------------------
        # 🚗 VEHICLE FLOW
        # -------------------------------
        if intent == "vehicle":
            print("🚗 Vehicle flow")
            result = await kimi_service.get_vehicle_data(query)
            return {"status": "success", "response": format_response(result)}

        # -------------------------------
        # 🛒 SHOPPING FLOW
        # -------------------------------
        live_products = []
        local_results = []
        if intent == "shopping":
            rag_start = time.time()
            local_results = fast_query(query, threshold=2.2)
            print(f"🛒 RAG Check: Found {len(local_results)} docs (Took {time.time() - rag_start:.2f}s)")

            if len(local_results) < 3:
                kimi_start = time.time()
                live_products = await kimi_service.get_product_data(query)
                print(f"⚡ Kimi Search: Found {len(live_products)} products (Took {time.time() - kimi_start:.2f}s)")

                if live_products:
                    background_tasks.add_task(kimi_service.cache_and_store_products, live_products, retail_crawler, query)
                    # For extremely fast response, we can return live products directly
                    # return {"status": "success", "response": format_response(live_products)}

        # -------------------------------
        # 🌐 FALLBACK / BOT RESPONSE
        # -------------------------------
        bot_start = time.time()
        print("🧠 Waiting for Bot response...")
        response = await chat_with_bot(
            query,
            discovered_stores=[], # Can be populated if needed
            live_context=live_products,
            intent_type=intent,
            local_docs=local_results
        )
        print(f"✅ Bot Done (Took {time.time() - bot_start:.2f}s)")
        print(f"🚀 Total Response Time: {time.time() - start_time:.2f}s")

        return {"status": "success", "response": format_response(response)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
