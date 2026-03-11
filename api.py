from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from crawler import crawl_site, crawl_site_recursive
from ingest import add_content_to_store, add_multiple_contents_to_store
from vector_store import vector_store, clear_vector_store
from query import fast_query
from bot import chat_with_bot
from retail_crawler import retail_crawler
from kimi_service import kimi_service
from urllib.parse import urlparse

# Track the most recently crawled domain to prioritize it in RAG
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

async def background_ingest(url: str, max_pages: int = 1):
    """Generic background task for crawling and ingestion."""
    try:
        print(f"Background ingestion started for: {url} (max_pages={max_pages})")
        if max_pages <= 1:
            content, _ = await crawl_site(url)
            if content and len(content.strip()) > 10:
                await add_content_to_store(content, {"source": url})
                update_last_domain(url)
            else:
                print(f"Background crawl failed for {url}: No content")
        else:
            results = await crawl_site_recursive(url, max_pages=max_pages)
            if results:
                await add_multiple_contents_to_store(results)
                print(f"Background deep ingestion complete for {url}")
            else:
                print(f"Background deep crawl failed for {url}: No results")
    except Exception as e:
        print(f"Error in background ingestion for {url}: {e}")

app = FastAPI(title="Retail AI RAG API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
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
    try:
        # Check if URL looks valid
        if not request.url.startswith("http"):
            raise HTTPException(status_code=400, detail="Invalid URL protocol")
            
        background_tasks.add_task(background_ingest, request.url, max_pages=1)
        return {"status": "success", "message": f"Ingestion for {request.url} started in background."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    try:
        # Check if URL looks valid
        if not request.url.startswith("http"):
            raise HTTPException(status_code=400, detail="Invalid URL protocol")
            
        print(f"Deep crawl requested for: {request.url}")
        background_tasks.add_task(background_ingest, request.url, max_pages=15)
            
        return {"status": "success", "message": f"Deep ingestion for {request.url} (up to 15 pages) started in background."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_endpoint():
    try:
        clear_vector_store()
        return {"status": "success", "message": "Memory cleared successfully."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # 2. Heuristic: Determine if we need live discovery
        user_message = request.message.lower()
        
        # Determine topic and if it's retail or general info
        retail_keywords = [
            # Specific tech products first (most specific → least specific order)
            "macbook", "iphone", "ipad", "apple watch", "airpods", "iphone", "ipad",
            "samsung galaxy", "pixel phone",
            "laptop", "mac", "apple", "samsung",
            "mobile", "electronics", "camera", "gadgets", "phone",
            # Clothing — specific first
            "t-shirt", "tshirt", "tshirts", "polo shirt",
            "shirt", "shirts",
            "jeans", "pants", "dress", "hoodie", "jacket", "sweater",
            "sneakers", "sneaker", "footwear", "shoes",
            "clothing", "apparel", "fashion", "clothes",
            # Other retail
            "furniture", "toys", "watch", "walkie", "talkie",
            "game", "puzzle", "doll", "action figure", "plush", "bottle", "baby",
            "shopping", "shop", "buy", "store", "find", "price"
        ]
        shopping_verbs = ["find", "buy", "shop", "search", "where can i", "get me", "show me", "looking for", "how much"]
        question_starts = ["what", "how", "why", "who", "when", "tell", "explain", "describe", "define", "is there", "are there"]
        
        # High-confidence shopping markers that override "is_question"
        shopping_markers = ["price", "cost", "how much", "buy", "shop", "purchase", "where to buy"]
        
        is_retail_query = any(kw in user_message for kw in retail_keywords)
        has_shopping_intent = any(verb in user_message for verb in shopping_verbs) or any(m in user_message for m in shopping_markers)
        is_question = any(user_message.startswith(q) for q in question_starts) or "?" in user_message
        
        # Explicit intent detection
        # 1. If it has explicit shopping markers, it's shopping even if it's a question
        # 2. If it has retail keywords but is NOT a general info question (why/how/explain)
        is_shopping = False
        if has_shopping_intent:
            is_shopping = True
        elif is_retail_query and not (is_question and any(q in user_message for q in ["why", "how", "explain", "describe", "define"])):
            is_shopping = True
        
        # New: Force shopping if specific tech/product keywords are found with "best" or "latest"
        if any(kw in user_message for kw in ["macbook", "iphone", "laptop", "phone"]) and any(w in user_message for w in ["best", "latest", "price", "config"]):
            is_shopping = True

        if not is_shopping and len(user_message.split()) <= 10 and not is_question:
            is_shopping = True
            
        intent_type = "shopping" if is_shopping else "info"
        
        seeds = []
        live_products = []
        if is_shopping:
            # 1. NEW: Identify preferred source domain (e.g., from query or context)
            preferred_source = last_crawled_domain
            if "brightminds" in user_message:
                preferred_source = "brightminds.co.uk"
            elif "baby" in user_message or "brands" in user_message:
                preferred_source = "babybrandsdirect.co.uk"
            elif "puckator" in user_message:
                preferred_source = "puckator-dropship.co.uk"
            
            # 2. Check LOCAL Database first
            print(f"DEBUG: Checking local DB for shopping query: '{request.message}' (Preferred: {preferred_source})")
            
            # Use a more relaxed threshold (2.2) and deeper search (k=25)
            local_results = fast_query(request.message, threshold=2.2, preferred_source=preferred_source)
            # If still few results, try a broader search without source filter
            if len(local_results) < 3:
                local_results = fast_query(request.message, threshold=2.2)
            
            # Refined Relevance Check: Include shorter words but filter out common stop words/symbols and generic retail terms
            ignore_list = [
                "some", "show", "find", "this", "that", "with", "from", "for", "the", "and", "any", "all",
                "products", "items", "give", "list", "search", "looking", "about", "please", "me", "after",
                "toys", "toy", "related", "other", "some", "showing",
                "price", "cost", "much", "how", "what", "is", "of", "good", "nice", "great", 
                "awesome", "perfect", "better", "top", "with", "configuration"
            ]
            query_keywords = [w.lower().strip("?!.,&") for w in user_message.split() if len(w.strip("?!.,&")) >= 2 and w.lower() not in ignore_list]
            
            # Extract product_type here so it's available in the relevance check below
            product_type = "products"
            for kw in retail_keywords:
                if kw in user_message:
                    product_type = kw
                    break
            
            print(f"DEBUG: Local results count: {len(local_results)}. Query keywords: {query_keywords}. Product type: {product_type}")
            
            has_relevant_local = False
            for i, (doc, score) in enumerate(local_results):
                content_lower = doc.page_content.lower()
                img_url = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url")
                source = doc.metadata.get("source") or doc.metadata.get("source_url") or ""
                
                # STICKY DB POLICY: Strict Matching
                # Require a high percentage of unique keywords to match
                matches = [kw for kw in query_keywords if kw in content_lower]
                relevance_ratio = len(matches) / len(set(query_keywords)) if query_keywords else 1.0
                
                # Extra check: at least the core product keyword must be in the document
                # e.g. if user asks for "t-shirt", shoes shouldn't match just because of "mens"
                core_kw = product_type if product_type != "products" else None
                product_type_match = (core_kw is None) or (core_kw.replace("-", " ") in content_lower) or (core_kw in content_lower)
                
                source_consistent = True
                if preferred_source and preferred_source not in source.lower():
                    source_consistent = False
                
                # DB PRIORITY: Only skip external search if we have a high relevance match
                # Raise threshold to 0.8 (from 0.6) to prevent cross-category false positives
                if relevance_ratio >= 0.8 and source_consistent and product_type_match:
                    has_relevant_local = True
                    break

            if has_relevant_local:
                print(f"DEBUG: Found relevant results locally ({len(local_results)} docs). Skipping external search.")
            else:
                reason = "No relevant local images/keywords" if local_results else "No local results found"
                print(f"DEBUG: {reason}. Triggering Kimi Search.")
                import re
                match = re.search(r"(?:in|near|at|around)\s+([a-zA-Z\s,]+)", request.message, re.IGNORECASE)
                location = match.group(1).strip() if match else "local area"
                
                # Pass only the clean product keyword to Kimi, not a verbose sentence
                if location == "local area":
                    retail_topic = product_type  # e.g. "macbook", "iphone", "t-shirt"
                else:
                    retail_topic = f"{product_type} {location}"  # e.g. "macbook london"
                    
                seeds = await kimi_service.search_sources(retail_topic)
                
                if preferred_source:
                    source_url = f"https://www.{preferred_source}"
                    if not any(preferred_source in s for s in (seeds or [])):
                        seeds = [source_url] + (seeds or [])
                    else:
                        matching = [s for s in seeds if preferred_source in s]
                        others = [s for s in seeds if preferred_source not in s]
                        seeds = matching + others

                if seeds:
                    print("="*50)
                    print(f"🔥 FAST RESPONSE: Fetching products directly from Kimi for '{product_type}'")
                    live_products = await kimi_service.search_products(product_type)
                    if seeds and len(seeds) > 0:
                        update_last_domain(seeds[0])
                    print(f"🔥 FAST RESPONSE COMPLETE: Found {len(live_products)} products.")
                    print("="*50)
                    
                    # Background task to cache Kimi images to S3 and store in DB
                    background_tasks.add_task(kimi_service.cache_and_store_products, live_products, retail_crawler, product_type)
                    
                    # Background sync the pages for future searches — limit to 10 to prevent memory exhaustion
                    background_tasks.add_task(retail_crawler.sync_store, seeds[0], max_pages=10, target_category=product_type)
                    for seed in seeds[1:3]: # Limit background sync
                        background_tasks.add_task(retail_crawler.sync_store, seed, max_pages=10, target_category=product_type)
        else:
            # General Discovery: Handle non-retail fields dynamically
            is_question = "?" in user_message or any(user_message.startswith(q) for q in ["what", "how", "why", "who", "when", "tell", "explain", "describe", "define"])
            
            if is_question:
                topic = request.message.replace("?", "").strip()
                print(f"General discovery triggered for topic: {topic}")
                seeds = await kimi_service.search_sources(topic)
                for seed in seeds[:2]:
                    background_tasks.add_task(background_ingest, seed, max_pages=15)
        
        # 1. Get Response from Bot (RAG) with knowledge of discovered seeds and LIVE products
        # Pass intent_type to bot for specialized prompting
        bot_response = chat_with_bot(
            request.message, 
            seeds[:3], 
            live_products, 
            intent_type=intent_type,
            local_docs=local_results if is_shopping else None
        )
        
        return {"status": "success", "response": bot_response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retail/sync")
async def sync_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger sync for a specific retail site.
    """
    background_tasks.add_task(retail_crawler.sync_store, request.url)
    update_last_domain(request.url)
    return {"status": "success", "message": f"Sync task for {request.url} added to background queue."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
