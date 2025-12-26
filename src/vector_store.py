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

    def save(self, path):
        arr_ids = np.array(self.ids, dtype=object)
        arr_meta = np.array(self.metadatas, dtype=object)
        if self.embeddings is None:
            emb = np.empty((0,))
        else:
            emb = self.embeddings
        np.savez(path, embeddings=emb, ids=arr_ids, metadatas=arr_meta)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        inst = cls()
        emb = data.get("embeddings")
        if emb is not None:
            inst.embeddings = emb
        ids = data.get("ids")
        metas = data.get("metadatas")
        if ids is not None:
            inst.ids = ids.tolist()
        if metas is not None:
            inst.metadatas = metas.tolist()
        return inst

    @classmethod
    def from_jsonl(cls, jsonl_path):
        inst = cls()
        ids = []
        embeds = []
        metas = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ids.append(obj.get("id"))
                embeds.append(obj.get("embedding"))
                metas.append({"source": obj.get("source"), "chunk_index": obj.get("chunk_index"), "text": obj.get("text")})
        if embeds:
            inst.embeddings = np.array(embeds)
        inst.ids = ids
        inst.metadatas = metas
        return inst
