# Crawl4AI RAG Assistant

An intelligent assistant that uses **Crawl4AI** to ingest websites into a **ChromaDB** vector store, enabling a RAG (Retrieval-Augmented Generation) chat experience powered by **LangChain** and **Ollama (llama3)**.

## Project Structure

- **Backend (Python/FastAPI)**:
  - `api.py`: FastAPI endoints for crawling and chatting.
  - `crawler.py`: Crawl4AI integration for web scraping.
  - `vector_store.py`: ChromaDB management and text chunking.
  - `bot.py`: RAG chain logic using LangChain.
- **Frontend (React)**:
  - Located in the `/frontend` directory.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with `llama3` model pulled.
- Node.js (for frontend)

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Backend**:
   ```bash
   python api.py
   ```
3. **Run Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
