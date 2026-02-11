import asyncio
import os
from crawler import crawl_site
from vector_store import add_content_to_store, get_vector_store
from bot import chat_with_bot

async def test_crawler():
    print("\n--- Testing Crawler ---")
    url = "https://example.com"
    content = await crawl_site(url)
    if content and "Example Domain" in content:
        print("✅ Crawler Test Passed")
        return content
    else:
        print("❌ Crawler Test Failed")
        return None

def test_vector_store(content):
    print("\n--- Testing Vector Store ---")
    try:
        add_content_to_store(content, {"source": "test_example"})
        # Check if we can retrieve something
        store = get_vector_store()
        results = store.similarity_search("Example Domain", k=1)
        if len(results) > 0:
            print("✅ Vector Store Test Passed")
            return True
    except Exception as e:
        print(f"❌ Vector Store Test Failed: {e}")
    return False

def test_bot():
    print("\n--- Testing Bot (RAG) ---")
    query = "What is the content of the example domain?"
    try:
        response = chat_with_bot(query)
        print(f"Bot Response: {response}")
        if response and len(response) > 10:
            print("✅ Bot Test Passed")
            return True
    except Exception as e:
        print(f"❌ Bot Test Failed: {e}")
        print("   Note: Ensure Ollama is running and 'llama3' is pulled.")
    return False

async def run_all_tests():
    print("Starting System Integration Tests...")
    
    content = await test_crawler()
    if content:
        if test_vector_store(content):
            test_bot()
    
    print("\nTests Completed.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
