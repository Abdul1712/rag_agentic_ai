import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

_local_sbert = None

def _ensure_local_model():
    global _local_sbert
    if _local_sbert is None:
        from sentence_transformers import SentenceTransformer
        _local_sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_sbert

def get_embedding(text: str):
    if OPENAI_API_KEY:
        resp = openai.embeddings.create(input=text, model=EMBED_MODEL)
        return resp["data"][0]["embedding"]
    else:
        model = _ensure_local_model()
        vec = model.encode(text)
        return vec.tolist()

def batch_get_embeddings(texts):
    if OPENAI_API_KEY:
        resp = openai.embeddings.create(input=texts, model=EMBED_MODEL)
        return [item["embedding"] for item in resp["data"]]
    else:
        model = _ensure_local_model()
        vecs = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]
