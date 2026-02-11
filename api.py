from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from crawler import crawl_site
from vector_store import add_content_to_store
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
        content = await crawl_site(request.url)
        if content and len(content.strip()) > 10:
            add_content_to_store(content, {"source": request.url})
            return {"status": "success", "message": f"Successfully crawled and ingested {request.url}"}
        else:
            return {"status": "error", "message": "No meaningful content extracted. The site might require JavaScript or be blocking the crawler."}
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
