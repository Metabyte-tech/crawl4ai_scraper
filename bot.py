from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from query import get_cached_retriever
from langchain.prompts import PromptTemplate
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=100)
def get_llm():
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        anthropic_api_key=os.getenv("MOONSHOT_API_KEY"),
        temperature=0
    )

def chat_with_bot(query: str, discovered_stores: list = None, live_context: list = None):
    """
    Sends a query to the chatbot and returns the response.
    """
    retriever = get_cached_retriever()
    docs = retriever._get_relevant_documents(query)
    rag_context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format LIVE products into context (guaranteed S3 URLs)
    live_context_str = ""
    if live_context:
        live_items = []
        for p in live_context:
            live_items.append(
                f"LIVE_PRODUCT_S3: {p.get('name')}\n"
                f"Brand: {p.get('brand')}\n"
                f"Price: {p.get('price')} {p.get('currency')}\n"
                f"Image URL: {p.get('image_url')}\n"
                f"Source URL: {p.get('source_url')}"
            )
        live_context_str = "\n\n".join(live_items)
        print(f"DEBUG: Injected {len(live_context)} live products into context.")

    # Combine: prioritize LIVE data over stale DB data
    context = f"{live_context_str}\n\n{rag_context}".strip()
    
    print("-" * 50)
    print(f"DEBUG: RAG Context for '{query}':")
    print(context)
    print("-" * 50)
    
    stores_str = ", ".join(discovered_stores) if discovered_stores else "None yet"
    
    template = """You are a Hybrid AI Shopping Assistant.

    CONTEXT (RAG + LIVE DATA):
    {context}

    IMPORTANT: If there are products marked as "LIVE_PRODUCT_S3" in the context, you MUST show them in the carousel. These are the most recent results found for the user's query.

    INSTRUCTIONS:
    1. **Direct Answer**: Start your answer immediately. If you have "LIVE_PRODUCT_S3" items, summarize them briefly and then show the carousel.
    2. **Mandatory Carousel**: If "LIVE_PRODUCT_S3" items exist, you MUST provide a product carousel. Do NOT say "no products found" if live items are present.
    3. **Carousel Format**: Use this EXACT format for products:
       <product_carousel>
       [
         {{
           "name": "Product Name",
           "brand": "Brand Name",
           "price": "₹ 000",
           "image_url": "S3_URL",
           "source_url": "Source_URL",
           "details": "Short snippet of details"
         }}
       ]
       </product_carousel>
       CRITICAL RULES:
       - Use ONLY the JSON structure inside the tag.
       - **IMAGE PRIORITY**: Always use the 'Image URL' provided in the context. If an S3 URL (from 'ai-gent-storage') is available, you MUST use it as the `image_url`.
       - NEVER use the "Product: name, Brand: brand..." plain text format found in the Context.
       - Do NOT include any markdown bullets or extra text inside the tag.
       - If it's not valid JSON, the system will crash.
    
    Question: {question}
    
    Answer:"""
    
    prompt = template.format(
        context=context,
        question=query
    )
    
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content
    
    # Final safety check: if we see relative-looking domains in the JSON, fix them
    # (Doing this via simple string replacement to avoid complex regex for now)
    base_retail_domains = ["amazon.in", "flipkart.com", "ajio.com", "myntra.com", "firstcry.com", "nykaafashion.com", "m.media-amazon.com", "static.ajio.com"]
    for domain in base_retail_domains:
        # Prepend https:// if domain start without it and is preceded by quote
        content = content.replace(f'"{domain}', f'"https://{domain}')
        # Handle cases where double protocol might occur from the above replacement
        content = content.replace("https://https://", "https://")

    return content

if __name__ == "__main__":
    # Test chat
    query = "What is this project about?"
    try:
        response = chat_with_bot(query)
        print(f"Bot: {response}")
    except Exception as e:
        print(f"Error communicating with bot: {e}")
        print("Make sure Ollama is running and 'llama3' model is pulled.")
