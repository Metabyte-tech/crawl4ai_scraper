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
def chat_with_bot(query: str, discovered_stores: list = None, live_context: list = None, intent_type: str = "shopping", local_docs: list = None):
    """
    Sends a query to the chatbot and returns the response.
    """
    if local_docs is None:
        retriever = get_cached_retriever()
        docs = retriever._get_relevant_documents(query)
    else:
        # local_docs can be tuples of (Document, score) or plain Documents
        docs = [d[0] if isinstance(d, tuple) else d for d in local_docs]
    
    rag_items = []
    seen_products = set()  # Deduplication set
    
    for doc in docs:
        item = f"CONTENT: {doc.page_content}"
        # Standardize to IMAGE_URL and SOURCE_URL for LLM consistency
        img = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url") or doc.metadata.get("Image URL")
        src = doc.metadata.get("source") or doc.metadata.get("source_url") or doc.metadata.get("Source URL")
        name = doc.metadata.get("name") or doc.metadata.get("Product Name") or ""
        
        # Deduplication based on Source URL and Name
        product_key = (src, name)
        if product_key in seen_products:
            continue
        seen_products.add(product_key)
        item = f"CONTENT: {doc.page_content}"
        if img: item += f"\nIMAGE_URL: {img}"
        if src: item += f"\nSOURCE_URL: {src}"
        rag_items.append(item)
        
    rag_context = "\n\n---\n\n".join(rag_items)
    
    # Format LIVE products into context
    live_context_str = ""
    if live_context:
        live_items = []
        for p in live_context:
            live_items.append(
                f"LIVE_PRODUCT_S3: {p.get('name')}\n"
                f"Brand: {p.get('brand')}\n"
                f"Price: {p.get('price')} {p.get('currency')}\n"
                f"IMAGE_URL: {p.get('image_url')}\n"
                f"SOURCE_URL: {p.get('source_url')}"
            )
        live_context_str = "\n\n".join(live_items)
    context = f"{live_context_str}\n\n{rag_context}".strip()
    
    if intent_type == "shopping":
        template = """You are a Hybrid AI Shopping Assistant.
        
        CONTEXT:
        {context}
        INSTRUCTIONS:
        1. **Direct Answer**: Summarize the best options found.
        2. **Carousel**: You MUST provide a product carousel for the items found.
        3. **Carousel Format**: 
           <product_carousel>
           [ {{ "name": "...", "brand": "...", "price": "...", "image_url": "...", "source_url": "..." }} ]
           </product_carousel>
        
        CRITICAL PRODUCT RULES:
        1. **IMAGE ACCURACY**: ONLY use the URL labeled `IMAGE_URL` for the `image_url` field. NEVER use `SOURCE_URL` as an image.
        2. **IMAGE NECESSITY**: ONLY include products in the carousel if they have a valid, non-empty `IMAGE_URL`.
        2. **IMAGE NECESSITY**: ONLY include products in the carousel if they have a valid, non-empty `IMAGE_URL` that starts with http or https.
        3. **Source Consistency**: If the user is asking about a specific store (e.g., Brightminds), ONLY show products from that store in the carousel. Exclude generic marketplace matches (like Flipkart/Amazon) if they appear in the CONTEXT unless they are the only options found.
        4. **Brand Matching**: Ensure the products shown match the brand/topic requested. Do not show irrelevant items.
        5. **NO RAW JSON**: NEVER output raw JSON blocks or internal tags like `<product_carousel>` inside your natural language explanation text. They must be separate.
        Question: {question}
        Answer:"""
    else:
        template = """You are an Intelligent Informational Assistant.
        
        CONTEXT:
        {context}
        INSTRUCTIONS:
        1. **Deep Explanation**: Provide a comprehensive and clear explanation of the topic.
        2. **Real-World Examples**: Include at least one practical example or scenario to illustrate the concept.
        3. **No Carousels**: Do NOT use the <product_carousel> tag or show products as cards. Use text only.
        4. **Structured Layout**: Use headers and bullet points for readability.
        
        Question: {question}
        Answer:"""
    
    prompt = template.format(context=context, question=query)
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content
    
    # URL prefixing logic - ensuring all domains have https:// protocol
    base_retail_domains = [
        "amazon.in", "flipkart.com", "ajio.com", "myntra.com", 
        "firstcry.com", "nykaafashion.com", "m.media-amazon.com", 
        "static.ajio.com", "ai-gent-storage.s3.ap-south-1.amazonaws.com",
        "brightminds.co.uk"
        "brightminds.co.uk", "mxwholesale.co.uk"
    ]
    for domain in base_retail_domains:
        # Match "domain or //domain and replace with https://domain
        content = content.replace(f'"{domain}', f'"https://{domain}')
        content = content.replace(f'"//{domain}', f'"https://{domain}')
        # Clean double https if accidentally created
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
