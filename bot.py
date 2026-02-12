from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from vector_store import get_vector_store
from langchain.prompts import PromptTemplate

def get_chatbot_chain():
    """
    Creates a RAG chain for the chatbot.
    """
    llm = ChatOllama(model="llama3", temperature=0)
    vector_store = get_vector_store()
    
    template = """You are a technical documentation assistant. Answer the user's question directly using ONLY the provided context.
    
    RESPONSE GUIDELINES:
    1. **Analyze the question type**:
       - Installation/setup questions ("how to install", "how to set up", "getting started") → Provide step-by-step instructions with code snippets
       - Conceptual questions ("what is", "explain", "describe") → Provide clear explanations focusing on concepts, NOT installation steps
    
    2. **Be direct and natural**:
       - Start answering immediately without disclaimers or preambles
       - Don't say "The provided context does not contain..." at the start
       - Don't add notes like "Please note that this explanation is based solely on..."
       - Just answer the question naturally
    
    3. **For installation/setup questions**:
       - List all steps in order
       - Include exact commands from the context
       - Include prerequisites and verification steps
    
    4. **For conceptual questions**:
       - Explain what it is, why it's used, and how it works
       - Do NOT include installation steps
       - Keep it concise and focused
    
    5. **If information is missing**:
       - Only mention missing information at the END if relevant
       - Say something like "Note: The documentation doesn't cover [specific aspect]"
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

def chat_with_bot(query: str):
    """
    Sends a query to the chatbot and returns the response.
    """
    qa_chain = get_chatbot_chain()
    response = qa_chain.invoke(query)
    return response["result"]

if __name__ == "__main__":
    # Test chat
    query = "What is this project about?"
    try:
        response = chat_with_bot(query)
        print(f"Bot: {response}")
    except Exception as e:
        print(f"Error communicating with bot: {e}")
        print("Make sure Ollama is running and 'llama3' model is pulled.")
