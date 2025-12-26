import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .config import Config
from .embeddings import Embeddings
from .vector_store import MemoryVectorStore

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


STORE_PATH = os.path.join(Config.DATA_DIR, "vector_store.npz")
JSONL_PATH = os.path.join(Config.DATA_DIR, "chunks_with_embeddings.jsonl")


def ensure_store():
    if os.path.exists(STORE_PATH):
        return MemoryVectorStore.load(STORE_PATH)
    if os.path.exists(JSONL_PATH):
        store = MemoryVectorStore.from_jsonl(JSONL_PATH)
        os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
        store.save(STORE_PATH)
        return store
    raise FileNotFoundError("no vector store or chunks found; run ingest first")


@app.post("/query")
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query required")
    store = ensure_store()
    emb = Embeddings()
    q_vec = emb.embed(req.query)[0]
    results = store.search(q_vec, top_k=req.top_k)
    return {"query": req.query, "results": results}
