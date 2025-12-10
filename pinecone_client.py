import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

if PINECONE_API_KEY:
    import pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

class PineconeClient:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set in .env.")
        existing = pinecone.list_indexes()
        if INDEX_NAME not in existing:
            pinecone.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine")
        self.index = pinecone.Index(INDEX_NAME)

    def upsert(self, vectors):
        return self.index.upsert(vectors=vectors)

    def query(self, embedding, top_k=5, include_metadata=True):
        res = self.index.query(vector=embedding, top_k=top_k, include_metadata=include_metadata)
        if hasattr(res, "to_dict"):
            return res.to_dict()
        return res
