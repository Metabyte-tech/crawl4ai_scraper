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

async def deep_crawl_endpoint_logic(url: str):
    """Internal logic for deep crawling and ingestion."""
    try:
        print(f"Background deep crawl started for: {url}")
        results = await crawl_site_recursive(url, max_pages=15)
        if results:
            await add_multiple_contents_to_store(results)
            print(f"Background ingestion complete for {url}")
    except Exception as e:
        print(f"Error in background crawl for {url}: {e}")

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
async def crawl_endpoint(request: CrawlRequest):
    try:
        content, _ = await crawl_site(request.url)
        if content and len(content.strip()) > 10:
            add_content_to_store(content, {"source": request.url})
            return {"status": "success", "message": f"Successfully crawled and ingested {request.url}"}
        else:
            return {"status": "error", "message": "No meaningful content extracted. The site might require JavaScript or be blocking the crawler."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest):
    try:
        print(f"Starting deep crawl for: {request.url}")
        results = await crawl_site_recursive(request.url, max_pages=100)
        
        if not results:
            return {"status": "error", "message": "Deep crawl failed to extract any content."}
            
        print(f"Pages crawled: {len(results)}. Starting batch ingestion...")
        await add_multiple_contents_to_store(results)
            
        return {"status": "success", "message": f"Deep crawl complete. Ingested {len(results)} pages from {request.url}"}
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
            "shoes", "clothes", "shopping", "shop", "buy", "store", "find", "price", 
            "laptop", "mobile", "electronics", "furniture", "toys", "watch", "camera", 
            "sneakers", "sneaker", "footwear", "apparel", "gadgets", "phone",
            "shirt", "shirts", "t-shirt", "tshirts", "jeans", "pants", "clothing", "dress", "fashion"
        ]
        shopping_verbs = ["find", "buy", "shop", "search", "where can i", "get me", "show me", "looking for"]
        question_starts = ["what", "how", "why", "who", "when", "tell", "explain", "describe", "define"]
        
        is_retail_query = any(kw in user_message for kw in retail_keywords)
        has_shopping_intent = any(verb in user_message for verb in shopping_verbs)
        is_question = any(user_message.startswith(q) for q in question_starts) or "?" in user_message
        
        # Explicit intent detection
        # 1. If it has shopping verbs or retail keywords, it's shopping
        # 2. If it's a very short query (<= 3 words) and NOT a question, assume shopping intent
        is_shopping = (is_retail_query or has_shopping_intent) and not (is_question and not has_shopping_intent)
        if not is_shopping and len(user_message.split()) <= 3 and not is_question:
            is_shopping = True
            
        intent_type = "shopping" if is_shopping else "info"
        
        seeds = []
        live_products = []
        if is_shopping:
            # 1. NEW: Identify preferred source domain (e.g., from query or context)
            preferred_source = None
            if "brightminds" in user_message:
                preferred_source = "brightminds.co.uk"
            
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
                "toys", "toy", "games", "game", "products", "item", "related", "other", "some", "showing"
            ]
            query_keywords = [w.lower().strip("?!.,&") for w in user_message.split() if len(w.strip("?!.,&")) >= 2 and w.lower() not in ignore_list]
            
            print(f"DEBUG: Local results count: {len(local_results)}. Query keywords: {query_keywords}")
            
            has_relevant_local = False
            for i, (doc, score) in enumerate(local_results):
                content_lower = doc.page_content.lower()
                img_url = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url")
                source = doc.metadata.get("source") or doc.metadata.get("source_url") or ""
                
                # Check keyword match for semantic safety
                keyword_match = any(kw in content_lower for kw in query_keywords) if query_keywords else True
                
                # STICKY DB POLICY: Consider it relevant if we have *any* keyword match in the local data
                # This ensures that after crawling a site, we "stick" to it for relevant keywords.
                # Added protection: Ensure it's not JUST category words, and respect preferred source if mentioned.
                has_keyword = any(kw in content_lower for kw in query_keywords)
                source_consistent = True
                if preferred_source and preferred_source not in source.lower():
                    source_consistent = False
                
                if has_keyword and source_consistent:
                    has_relevant_local = True
                    break

            if has_relevant_local:
                print(f"DEBUG: Found relevant results locally. Skipping external search.")
            else:
                reason = "No relevant local images/keywords" if local_results else "No local results found"
                print(f"DEBUG: {reason}. Triggering Kimi Search.")
                import re
                match = re.search(r"(?:in|near|at|around)\s+([a-zA-Z\s,]+)", request.message, re.IGNORECASE)
                location = match.group(1).strip() if match else "local area"
                
                product_type = "products"
                remaining_msg = re.sub(r"(?:find|buy|shop|search|for|in|near|at|around|get|show|me|some|any|all|the|about|want|to)\s+", "", user_message, flags=re.IGNORECASE).strip()
                if remaining_msg:
                    product_type = remaining_msg
                
                for kw in retail_keywords:
                    if kw in user_message:
                        product_type = kw
                        break
                
                if location == "local area":
                    retail_topic = f"direct online shopping links for {product_type} from top retailers"
                else:
                    retail_topic = f"{product_type} shopping stores in {location}"
                    
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
                    print(f"🔥 HOT SYNC STARTING: Primary source -> {seeds[0]} (Limiting to 3 pages for fast response)")
                    # Reduce initial sync from 15 to 3 pages to prevent frontend timeouts
                    sync_result = await retail_crawler.sync_store(seeds[0], max_pages=3, target_category=product_type)
                    live_products = sync_result if isinstance(sync_result, list) else []
                    print(f"🔥 HOT SYNC COMPLETE: Found {len(live_products)} products.")
                    print("="*50)
                    
                    # Background sync the rest of the pages for this seed plus the others
                    background_tasks.add_task(retail_crawler.sync_store, seeds[0], max_pages=100, target_category=product_type)
                    for seed in seeds[1:3]: # Limit background sync
                        background_tasks.add_task(retail_crawler.sync_store, seed, max_pages=100, target_category=product_type)
        else:
            # General Discovery: Handle non-retail fields dynamically
            is_question = "?" in user_message or any(user_message.startswith(q) for q in ["what", "how", "why", "who", "when", "tell", "explain", "describe", "define"])
            
            if is_question:
                topic = request.message.replace("?", "").strip()
                print(f"General discovery triggered for topic: {topic}")
                seeds = await kimi_service.search_sources(topic)
                for seed in seeds[:2]:
                    background_tasks.add_task(deep_crawl_endpoint_logic, seed)
        
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
    return {"status": "success", "message": f"Sync task for {request.url} added to background queue."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
