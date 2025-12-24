import os
import json
import numpy as np


class MemoryVectorStore:
    def __init__(self):
        self.ids = []
        self.embeddings = None
        self.metadatas = []

    def add(self, ids, embeddings, metadatas=None):
        arr = np.array(embeddings)
        if self.embeddings is None:
            self.embeddings = arr
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
        self.ids.extend(ids)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([None] * len(ids))

    def search(self, query_embedding, top_k=5):
        q = np.array(query_embedding)
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        dots = np.dot(self.embeddings, q)
        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q) + 1e-12)
        sims = dots / norms
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            results.append({"id": self.ids[i], "score": float(sims[i]), "metadata": self.metadatas[i]})
        return results
