import asyncio
import sys
from crawler import crawl_site
from vector_store import add_content_to_store
from bot import chat_with_bot

async def main():
    print("=== Crawl4AI -> LangChain -> ChatBot RAG System ===")
    
    url = input("\nEnter a URL to crawl and ingest (or press Enter to skip to chat): ").strip()
    
    if url:
        print(f"\nCrawling {url}...")
        content = await crawl_site(url)
        if content:
            print("Ingesting content into vector store...")
            add_content_to_store(content, {"source": url})
            print("Ingestion complete!")
        else:
            print("Crawling failed. Proceeding with existing data (if any).")
    
    print("\nStarting ChatBot. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query:
            continue
            
        print("Bot is thinking...")
        try:
            response = chat_with_bot(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")
            print("TIP: Ensure Ollama is running and you have the 'llama3' model.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
