import os
import io
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

sessions = {}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

def extract_text_from_pdfs(files_bytes: List[bytes]) -> str:
    text = ""
    for file_bytes in files_bytes:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def build_rag_chain(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Embeddings run locally — no API needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Use HF router via OpenAI-compatible API
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct:fastest",
        openai_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        openai_api_base="https://router.huggingface.co/v1",
        temperature=0.5,
        max_tokens=512,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False
    )
    return chain

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    files_bytes = [await f.read() for f in files]
    text = extract_text_from_pdfs(files_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDFs")

    chain = build_rag_chain(text)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chain": chain, "history": []}

    return {"session_id": session_id, "message": f"Processed {len(files)} file(s) successfully."}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload PDFs first.")

    chain = session["chain"]
    result = chain.invoke({"question": req.question})

    answer = result.get("answer", "").strip()
    session["history"].append({"user": req.question, "bot": answer})
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}