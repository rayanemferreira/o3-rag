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
from fastapi.responses import PlainTextResponse

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
    logger.info(f"usar colecao existente {COLLECTION_NAME}")
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    logger.info(f"Criar uma colecao: {COLLECTION_NAME}")

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
        logger.error(f"Ollama generate erro: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama generate erro: {str(e)}")

class QueryRequest(BaseModel):
    text: str
    k: int = TOP_K_DEFAULT

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Somente arquivos .txt são suportados")
    contents = await file.read()
    text = contents.decode('utf-8', errors='ignore')
    sentences = split_into_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Nenhum texto encontrado no arquivo")

    ids, metadatas, documents, embeddings = [], [], [], []

    logger.info(f"Processando o arquivo {file.filename}, {len(sentences)} sentences")

    for s in sentences:
        sid = str(uuid.uuid4())
        ids.append(sid)
        metadatas.append({"fonte": file.filename})
        documents.append(s)
        try:
            emb = ollama_embed(s)
            embeddings.append(emb)
        except Exception as e:
            logger.error(f"Falha ao incorporar a frase: {e}")
            continue

    if not embeddings:
        raise HTTPException(status_code=500, detail="Falha ao gerar embeddings")

    collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    try:
        chroma_client.persist()
        logger.info("Chroma persistiu com sucesso")
    except Exception as e:
        logger.warning(f"Chroma esta com erro: {e}")

    return {"inserted": len(embeddings)}

@app.post("/ia-prompt")
async def query(req: QueryRequest):
    q = req.text
    k = req.k
     
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="a consulta não pode esta vazia")

    logger.info(f"Consulta recebida: {q}")

    q_embed = ollama_embed(q)


    results = collection.query(query_embeddings=[q_embed], n_results=k, include=['documents','metadatas','distances'])
    

    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]
    ids = results.get('ids', [[]])[0]

    context_pieces = []
    for i, d in enumerate(docs):
        src = metas[i].get('source') if i < len(metas) else None
        context_pieces.append(f"Source: {src or 'unknown'}\n{d}")
    context = "\n---\n".join(context_pieces)


    system_msg=("Você é um assistente especializado que responde de forma curta, clara e correta. "
    "Use SOMENTE o contexto fornecido abaixo (Q/A históricas) para fundamentar sua respta correta"
    "Se o contexto não for suficiente, explique a limitação e peça mais detalhes. "
    "Cite quais Q/As inspiraram a resposta, referenciando os IDs quando possivel.")

    user_msg = (
    f"Pergunta do usuário:\n{q}\n\n"
    f" ----- CONTEXTO (Q/As relevantes) ----- \n{context}\n\n"
    "Regras:\n"
    "1) Baseie-se ňo contexto. \n"
    "2) Seja preciso.\n"
    "3) Se faltarem detalhes, diga o que falta. \n"
    "4) Ao final, liste 'Fontes (IDs aproximados)' com os IDs (se disponíveis) ou breve ref"
    )

    
    resp = ollama.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        options={"temperature": 0.2}
    ) 



    answer = resp["message"]["content"]
    distance = distances[0] if distances else None


    return PlainTextResponse(f"{answer}\n\nDistância: {distance}")

@app.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "ok"}