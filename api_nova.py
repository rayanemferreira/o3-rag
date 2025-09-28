import os
import uuid
import re
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_data")
TOP_K_DEFAULT = 5

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
COLLECTION_NAME = "documents"
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    logger.info(f"Using existing collection: {COLLECTION_NAME}")
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    logger.info(f"Created new collection: {COLLECTION_NAME}")

app = FastAPI(title="FastAPI + ChromaDB + Ollama RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    final = []
    for s in sentences:
        for part in s.split('\n'):
            p = part.strip()
            if p:
                final.append(p)
    return final

def ollama_embed(text: str) -> List[float]:
    try:
        logger.debug(f"Embedding text: {text[:50]}...")
        resp = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return resp["embedding"]
    except Exception as e:
        logger.error(f"Ollama embed error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama embed error: {str(e)}")

def ollama_generate(prompt: str) -> str:
    try:
        logger.debug(f"Generating with prompt: {prompt[:100]}...")
        resp = ollama.generate(model=OLLAMA_LLM_MODEL, prompt=prompt)
        return resp["response"]
    except Exception as e:
        logger.error(f"Ollama generate error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama generate error: {str(e)}")

class QueryRequest(BaseModel):
    text: str
    k: int = TOP_K_DEFAULT

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    contents = await file.read()
    text = contents.decode('utf-8', errors='ignore')
    sentences = split_into_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No text found in file")

    ids, metadatas, documents, embeddings = [], [], [], []

    logger.info(f"Processing file {file.filename}, {len(sentences)} sentences")

    for s in sentences:
        sid = str(uuid.uuid4())
        ids.append(sid)
        metadatas.append({"source": file.filename})
        documents.append(s)
        try:
            emb = ollama_embed(s)
            embeddings.append(emb)
        except Exception as e:
            logger.error(f"Failed to embed sentence: {e}")
            continue

    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate any embeddings")

    collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    try:
        chroma_client.persist()
        logger.info("Chroma persisted successfully")
    except Exception as e:
        logger.warning(f"Chroma persist error: {e}")

    return {"inserted": len(embeddings)}

@app.post("/ia-prompt")
async def query(req: QueryRequest):
    q = req.text
    k = req.k
     
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    logger.info(f"Received query: {q}")

    q_embed = ollama_embed(q)
    logger.info(f"ollama_embed query: {q_embed}")

    results = collection.query(query_embeddings=[q_embed], n_results=k, include=['documents','metadatas','distances'])
    logger.info(f"results query: {results}")

    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]
    ids = results.get('ids', [[]])[0]

    context_pieces = []
    for i, d in enumerate(docs):
        src = metas[i].get('source') if i < len(metas) else None
        context_pieces.append(f"Source: {src or 'unknown'}\n{d}")
    context = "\n---\n".join(context_pieces)

    prompt = f"You are an assistant. Use the following context to answer the user's question.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nPlease provide a helpful answer and cite the relevant context sources in-line."

    answer = ollama_generate(prompt)

    return {
        "answer": answer,
        "sources": [{"id": ids[i], "source": metas[i].get('source'), "distance": distances[i], "text": docs[i]} for i in range(len(docs))]
    }

@app.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "ok"}