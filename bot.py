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

def chat_with_bot(query: str, discovered_stores: list = None, live_context: list = None, intent_type: str = "shopping"):
    """
    Sends a query to the chatbot and returns the response.
    """
    retriever = get_cached_retriever()
    docs = retriever._get_relevant_documents(query)
    rag_context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format LIVE products into context
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
    
    # URL fixing logic
    base_retail_domains = ["amazon.in", "flipkart.com", "ajio.com", "myntra.com", "firstcry.com", "nykaafashion.com", "m.media-amazon.com", "static.ajio.com"]
    for domain in base_retail_domains:
        content = content.replace(f'"{domain}', f'"https://{domain}')
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
