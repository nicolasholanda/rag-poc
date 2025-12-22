import os

class Config:
    VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")
    EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    DATA_DIR = os.getenv("DATA_DIR", "data")
