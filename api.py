from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from crawler import crawl_site, crawl_site_recursive
from vector_store import add_content_to_store, add_multiple_contents_to_store, clear_vector_store
from bot import chat_with_bot

app = FastAPI(title="Crawl4AI RAG API")

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
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_with_bot(request.message)
        return {"status": "success", "response": response}
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
