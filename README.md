# PDF Chatbot

An AI app that lets you upload multiple PDFs and ask questions about them.

## Tech Stack
- **LangChain** — document chunking, embeddings, and conversational retrieval
- **FAISS** — vector similarity search
- **OpenAI GPT** — language model
- **Streamlit** — frontend UI

## How It Works
1. Upload one or more PDFs
2. The app splits and embeds the text into a FAISS vector store
3. Ask questions — the app retrieves relevant chunks and generates answers with memory of the conversation
