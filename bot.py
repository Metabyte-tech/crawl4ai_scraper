from langchain_anthropic import ChatAnthropic
from query import get_cached_retriever
from langchain.prompts import PromptTemplate
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        anthropic_api_key=os.getenv("MOONSHOT_API_KEY"),
        temperature=0
    )

async def chat_with_bot(query: str, discovered_stores: list = None, live_context: list = None, intent_type: str = "shopping", local_docs: list = None):
    """
    Sends a query to the chatbot asynchronously and returns the response.
    """
    if local_docs is None:
        retriever = get_cached_retriever()
        # Since cached_query is sync, we run it in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, retriever._get_relevant_documents, query)
    else:
        docs = [d[0] if isinstance(d, tuple) else d for d in local_docs]
    
    rag_items = []
    seen_products = set()
    
    for doc in docs:
        img = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url") or doc.metadata.get("Image URL")
        
        if not img:
            import re
            markdown_imgs = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
            html_imgs = re.findall(r'<img.*?src=["\'](.*?)["\']', doc.page_content, flags=re.IGNORECASE)
            all_content_imgs = markdown_imgs + html_imgs
            if all_content_imgs:
                for u in all_content_imgs:
                    if "logo" not in u.lower() and "icon" not in u.lower():
                        img = u
                        break
                if not img: img = all_content_imgs[0]

        src = doc.metadata.get("source") or doc.metadata.get("source_url") or doc.metadata.get("Source URL")
        name = doc.metadata.get("name") or doc.metadata.get("Product Name") or ""
        
        product_key = (src, name)
        if product_key in seen_products:
            continue
        seen_products.add(product_key)
        
        item = f"CONTENT: {doc.page_content}"
        if img: item += f"\nIMAGE_URL: {img}"
        if src: item += f"\nSOURCE_URL: {src}"
        rag_items.append(item)
        
    rag_context = "\n\n---\n\n".join(rag_items)
    
    live_context_str = ""
    if live_context:
        live_items = []
        for p in live_context:
            live_items.append(
                f"LIVE_PRODUCT: {p.get('name')}\n"
                f"Brand: {p.get('brand')}\n"
                f"Price: {p.get('price')} {p.get('currency', '')}\n"
                f"Rating: {p.get('rating', 'N/A')}\n"
                f"Offers: {p.get('offers', 'None')}\n"
                f"Source: {p.get('source') or p.get('store_source') or 'Search'}\n"
                f"IMAGE_URL: {p.get('image_url')}\n"
                f"SOURCE_URL: {p.get('url') or p.get('source_url')}"
            )
        live_context_str = "\n\n".join(live_items)
    
    context = f"{live_context_str}\n\n{rag_context}".strip()
    
    if intent_type == "shopping":
        has_images = "IMAGE_URL" in context
        carousel_instruction = ""
        if has_images:
            carousel_instruction = """
        2. Provide a product carousel.
        3. Format: <product_carousel> ["Product Name 1", "Product Name 2"] </product_carousel>
        CRITICAL: ONLY use REAL names from CONTEXT. No hallucination.
        CRITICAL: The list MUST be a compact, SINGLE-LINE string."""
        
        template = f"""You are a Hybrid AI Shopping Assistant.
        CONTEXT: {{context}}
        INSTRUCTIONS:
        1. Summarize the best options.{carousel_instruction}
        CRITICAL: ONLY use REAL URLs from CONTEXT. No hallucination.
        Question: {{question}}
        Answer:"""
    else:
        template = """You are an Intelligent Informational Assistant.
        CONTEXT: {context}
        Question: {question}
        Answer:"""
    
    prompt = template.format(context=context, question=query)
    llm = get_llm()
    response = await llm.ainvoke(prompt)
    content = response.content
    
    # Simple URL cleaning
    content = content.replace("https://https://", "https://").replace("https://http://", "https://")
    
    if not content.strip():
        if intent_type == "shopping":
            return "I couldn't find any specific products matching your query at the moment. Please try searching for something else or let me know if you need help with anything else!"
        return "I'm sorry, I couldn't find any information on that. Could you please rephrase your question?"
        
    return content
