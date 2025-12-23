import os
import json
import uuid
import sys
from pypdf import PdfReader
from .config import Config


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= overlap:
        overlap = int(chunk_size / 10)
    step = chunk_size - overlap
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def ingest(path=None, out_file=None, chunk_size=1000, overlap=200):
    base = path or Config.DATA_DIR
    base = os.path.abspath(base)
    if not os.path.exists(base):
        raise FileNotFoundError(base)
    out_file = out_file or os.path.join(base, "chunks.jsonl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    files = []
    for root, _, names in os.walk(base):
        for name in names:
            if name.lower().endswith((".pdf", ".md", ".txt")):
                files.append(os.path.join(root, name))
    written = 0
    with open(out_file, "w", encoding="utf-8") as out:
        for fpath in files:
            try:
                if fpath.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(fpath)
                else:
                    text = read_text_file(fpath)
            except Exception:
                continue
            chunks = chunk_text(text, chunk_size, overlap)
            for idx, chunk in enumerate(chunks):
                obj = {
                    "id": str(uuid.uuid4()),
                    "source": os.path.relpath(fpath, base),
                    "chunk_index": idx,
                    "text": chunk,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
    return {"chunks_written": written, "out_file": out_file}


def run_server():
    raise NotImplementedError


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        res = ingest()
        print(res)
    else:
        run_server()
