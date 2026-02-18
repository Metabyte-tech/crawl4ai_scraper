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
        is_retail = any(kw in user_message for kw in ["kids", "child", "shoes", "clothes", "shopping", "shop", "buy", "store", "buy", "find", "price"])
        
        seeds = []
        live_products = []
        if is_retail:
            import re
            match = re.search(r"(?:in|near|at|around)\s+([a-zA-Z\s,]+)", request.message, re.IGNORECASE)
            location = match.group(1).strip() if match else "local area"
            
            product_type = "shopping brands"
            for kw in ["shoes", "clothes", "toys", "electronics", "furniture", "kids"]:
                if kw in user_message:
                    product_type = kw
                    break
            
            # Legacy store search (now specialized version of general search)
            if location == "local area":
                retail_topic = f"direct online shopping links for kids {product_type} from top Indian retailers"
            else:
                retail_topic = f"{product_type} shopping stores in {location}"
                
            seeds = await kimi_service.search_sources(retail_topic)
            if seeds:
                # HOT SYNC: Wait for the first seed to ensure we have REAL S3 URLs in the first response
                # We limit this to 3 pages to keep the response time reasonable
                print("="*50)
                print(f"🔥 HOT SYNC STARTING: Primary source -> {seeds[0]}")
                sync_result = await retail_crawler.sync_store(seeds[0], max_pages=3)
                live_products = sync_result if isinstance(sync_result, list) else []
                print(f"🔥 HOT SYNC COMPLETE: Found {len(live_products)} products.")
                print("="*50)
                
                # Background sync for the rest of the discovery results
                for seed in seeds[1:]:
                    background_tasks.add_task(retail_crawler.sync_store, seed)
            else:
                print("No seeds found for retail discovery.")
        else:
            # General Discovery Heuristic: If it looks like a question or specific technical term
            # and we want to ensure we have the latest info (e.g. "Iroha 2")
            technical_topics = ["iroha", "chroma", "langchain", "fastapi", "python", "react", "nextjs"]
            is_technical = any(topic in user_message for topic in technical_topics)
            is_question = "?" in user_message or any(user_message.startswith(q) for q in ["what", "how", "why", "who", "when"])
            
            if is_technical or is_question:
                # Extract the main topic (simple heuristic)
                topic = request.message.replace("?", "").strip()
                print(f"General discovery triggered for topic: {topic}")
                seeds = await kimi_service.search_sources(topic)
                for seed in seeds:
                    # Generic deep crawl and ingest
                    background_tasks.add_task(deep_crawl_endpoint_logic, seed)
        
        # 1. Get Response from Bot (RAG) with knowledge of discovered seeds and LIVE products
        response = chat_with_bot(request.message, discovered_stores=seeds, live_context=live_products)
        
        return {"status": "success", "response": response}
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
