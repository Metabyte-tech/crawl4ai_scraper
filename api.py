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
        # 2. Heuristic: If user is looking for products, shops, or location-based items
        user_message = request.message.lower()
        seeds = []
        shopping_keywords = ["kids", "child", "shoes", "clothes", "shopping", "shop", "buy", "store", "buy", "find", "price"]
        if any(keyword in user_message for keyword in shopping_keywords):
            import re
            # Improved location extraction
            match = re.search(r"(?:in|near|at|around)\s+([a-zA-Z\s,]+)", request.message, re.IGNORECASE)
            location = match.group(1).strip() if match else "local area"
            
            # Simple product type extraction
            product_type = "products"
            for kw in ["shoes", "clothes", "toys", "electronics", "furniture"]:
                if kw in user_message:
                    product_type = kw
                    break
            
            # Use Kimi to find new seed URLs (blocking but fast enough)
            seeds = await kimi_service.search_stores(location, product_type)
            for seed in seeds:
                background_tasks.add_task(retail_crawler.sync_store, seed)
        
        # 1. Get Response from Bot (RAG) with knowledge of discovered seeds
        response = chat_with_bot(request.message, discovered_stores=seeds)
        
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
