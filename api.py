from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from crawler import crawl_site, crawl_site_recursive
from ingest import add_content_to_store, add_multiple_contents_to_store
from vector_store import clear_vector_store
from bot import chat_with_bot
from retail_crawler import retail_crawler
from kimi_service import kimi_service

async def deep_crawl_endpoint_logic(url: str):
    """Internal logic for deep crawling and ingestion."""
    try:
        print(f"Background deep crawl started for: {url}")
        results = await crawl_site_recursive(url, max_pages=15)
        if results:
            add_multiple_contents_to_store(results)
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
        results = await crawl_site_recursive(request.url, max_pages=20)
        
        if not results:
            return {"status": "error", "message": "Deep crawl failed to extract any content."}
            
        print(f"Pages crawled: {len(results)}. Starting batch ingestion...")
        add_multiple_contents_to_store(results)
            
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
            import re
            match = re.search(r"(?:in|near|at|around)\s+([a-zA-Z\s,]+)", request.message, re.IGNORECASE)
            location = match.group(1).strip() if match else "local area"
            
            # Dynamically detect product type (fallback to broad search)
            product_type = "products"
            # Extract most likely noun as category if no keywords match specifically
            remaining_msg = re.sub(r"(?:find|buy|shop|search|for|in|near|at|around|get|show|me|some|any|all|the|about|want|to)\s+", "", user_message, flags=re.IGNORECASE).strip()
            if remaining_msg:
                # Take the last few words as the category usually (e.g. "blue sneakers")
                # but if there are keywords, they override
                product_type = remaining_msg
            
            for kw in retail_keywords:
                if kw in user_message:
                    # If keyword found, it's a stronger signal
                    product_type = kw
                    break
            
            # Refine retail topic
            if location == "local area":
                retail_topic = f"direct online shopping links for {product_type} from top retailers"
            else:
                retail_topic = f"{product_type} shopping stores in {location}"
                
            seeds = await kimi_service.search_sources(retail_topic)
            if seeds:
                print("="*50)
                print(f"🔥 HOT SYNC STARTING: Primary source -> {seeds[0]}")
                # Use concurrency and speed optimizations implemented in crawler.py
                sync_result = await retail_crawler.sync_store(seeds[0], max_pages=3, target_category=product_type)
                live_products = sync_result if isinstance(sync_result, list) else []
                print(f"🔥 HOT SYNC COMPLETE: Found {len(live_products)} products.")
                print("="*50)
                
                for seed in seeds[1:3]: # Limit background sync to top 2 additional seeds to save tokens
                    background_tasks.add_task(retail_crawler.sync_store, seed, target_category=product_type)
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
        bot_response = chat_with_bot(request.message, seeds[:3], live_products, intent_type=intent_type)
        
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
