# PDF Chatbot
 
A RAG-powered chatbot that lets you upload multiple PDFs and ask questions about them — entirely free, no OpenAI API key required.
 
## Tech Stack
 
| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free) |
| Vector Store | FAISS (in-memory) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API (free) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
 
## How It Works
 
1. Upload one or more PDFs via the sidebar
2. Text is extracted, chunked, and embedded into a FAISS vector index
3. Ask a question — relevant chunks are retrieved and passed to the LLM
4. The model generates a grounded answer with memory of the conversation across multiple turns
 
## Project Structure
```
pdf-chatbot/
├── main.py          # FastAPI backend + RAG pipeline
├── requirements.txt
├── .env             # HuggingFace token (not committed)
├── .gitignore
└── static/
    └── index.html   # Frontend chat UI
```