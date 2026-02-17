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

def chat_with_bot(query: str, discovered_stores: list = None):
    """
    Sends a query to the chatbot and returns the response.
    """
    retriever = get_cached_retriever()
    docs = retriever._get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    stores_str = ", ".join(discovered_stores) if discovered_stores else "None yet"
    
    template = """You are a direct and helpful assistant. You have two sources of info: 1) The Context below, and 2) Your general training data.

    CRITICAL INSTRUCTION:
    - **JUMP STRAIGHT TO THE ANSWER**: Do not explain where your info comes from. 
    - **NEVER MENTION THE CONTEXT**: Never use words like "context", "provided info", "document", "source", or "metadata".
    - **NO PREAMBLES**: Do not say "I apologize", "Unfortunately", "Based on my knowledge", or "Here is an overview".
    - **NO EXPLANATIONS**: If the question is not in the Context, just answer it directly from your brain. Do not point out the mismatch.

    FORMATTING RULES:
    1. **RETAIL DATA**: If (and ONLY if) you find specific products in the Context, use this format:
       ### **Product Name** (Brand)
       - **Price**: ₹ [Price]
       - **Details**: [Details]
       - **[Click here to see Image](S3_URL)**
       - **[Direct Purchase Link](Source_URL)**
    2. **BACKGROUND SEARCH**: Only if Discovered Stores is NOT "None yet", start with this exact line: "I've started searching these stores for live updates: {discovered_stores}."

    Discovered Stores: {discovered_stores}
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = template.format(
        context=context,
        question=query,
        discovered_stores=stores_str
    )
    
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    # Test chat
    query = "What is this project about?"
    try:
        response = chat_with_bot(query)
        print(f"Bot: {response}")
    except Exception as e:
        print(f"Error communicating with bot: {e}")
        print("Make sure Ollama is running and 'llama3' model is pulled.")
