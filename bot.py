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
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
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
