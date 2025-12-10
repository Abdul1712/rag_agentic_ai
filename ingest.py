import os
import uuid
import tempfile
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import pdfplumber
from utils import chunk_text, normalize_whitespace
from embeddings_client import batch_get_embeddings
from pinecone_client import PineconeClient

load_dotenv()
PDF_URL = os.getenv("PDF_URL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

def download_pdf(url, dst_path):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in r.iter_content(1024*32):
            if chunk:
                f.write(chunk)

def extract_text_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = normalize_whitespace(text)
            pages.append({"page": i+1, "text": text})
    return pages

def prepare_chunks(pages, max_chars=1800, overlap_chars=200):
    raw = []
    for p in pages:
        header = f"[PAGE {p['page']}]\n"
        raw.append(header + p["text"])
    full = "\n\n".join(raw)
    chunks = chunk_text(full, max_chars=max_chars, overlap_chars=overlap_chars)
    out = []
    for idx, c in enumerate(chunks):
        out.append({
            "id": str(uuid.uuid4()),
            "text": c,
            "source": "Ebook-Agentic-AI.pdf",
            "chunk_index": idx
        })
    return out

def upsert_chunks(chunks):
    pc = PineconeClient()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embs = batch_get_embeddings(texts)
        vectors = []
        for c, emb in zip(batch, embs):
            vectors.append({
                "id": c["id"],
                "values": emb,
                "metadata": {
                    "text": c["text"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"]
                }
            })
        pc.upsert(vectors)
        print(f"Upserted batch {i}..{i+len(batch)}")

if __name__ == "__main__":
    if not PDF_URL:
        raise RuntimeError("Set PDF_URL in .env")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        print("Downloading PDF...")
        download_pdf(PDF_URL, tmp.name)
        print("Extracting...")
        pages = extract_text_from_pdf(tmp.name)
        print("Creating chunks...")
        chunks = prepare_chunks(pages)
        print("Upserting...")
        upsert_chunks(chunks)
        print("Done.")
