# Crawl4AI RAG Assistant

A Retreival-Augmented Generation (RAG) assistant that crawls websites, ingests their content into a local vector database, and allows you to chat with the documentation using a local LLM (Llama3).

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
*   **Node.js & npm**: [Download Node.js](https://nodejs.org/)
*   **Ollama**: [Download Ollama](https://ollama.com/) (Required for local LLM and embedding models)

## Setup Instructions

### 1. Ollama Setup

This project uses **Llama3** for the chat capabilities and **nomic-embed-text** for creating vector embeddings.

1.  Start the Ollama server:
    ```bash
    ollama serve
    ```
2.  Pull the required models:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```

### 2. Backend Setup (Python)

1.  Navigate to the project root directory.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3.  Activate the virtual environment:
    *   **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```
    *   **Windows**:
        ```bash
        venv\Scripts\activate
        ```
4.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Install Playwright browsers (required for crawling):
    ```bash
    playwright install
    ```

### 3. Frontend Setup (React/Vite)

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install the dependencies:
    ```bash
    npm install
    ```

## Running the Application

You need to run both the backend server and the frontend development server.

### 1. Start the Backend API

From the project root (with your virtual environment activated):

```bash
python api.py
```
*The backend API will start at `http://localhost:8000`.*

### 2. Start the Frontend Client

From the `frontend` directory:

```bash
npm run dev
```
*The frontend application will start at `http://localhost:5173` (or the port shown in your terminal).*

## Project Flow

Here is how the application works step-by-step:

1.  **Input**: The user enters a website URL in the frontend interface.
2.  **Crawling**: The backend `crawler.py` (powered by `crawl4ai` and Playwright) visits the URL and extracts the text content. It handles dynamic JavaScript content automatically.
3.  **Processing**: The text is passed to `vector_store.py`, where it is split into smaller, manageable chunks.
4.  **Embedding**: Each chunk is converted into a vector embedding using the `nomic-embed-text` model via Ollama.
5.  **Storage**: These vector embeddings are stored locally in a ChromaDB database (`chroma_db` folder).
6.  **Query**: The user asks a question in the chat interface.
7.  **Retrieval**: The system converts the question into a vector and searches the ChromaDB for the most relevant content chunks (RAG).
8.  **Answer**: `bot.py` sends the user's question along with the retrieved context to the `Llama3` model (via Ollama), which generates a precise, context-aware answer.
