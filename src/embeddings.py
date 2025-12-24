import os
try:
    import openai
except Exception:
    openai = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class Embeddings:
    def __init__(self, provider=None):
        self.provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "openai")
        if self.provider == "sentence-transformers":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        elif self.provider == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            self.openai = openai
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.openai.api_key = key
            resp = self.openai.Embedding.create(input=texts, model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
            return [item["embedding"] for item in resp["data"]]
        else:
            vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return vecs.tolist()
