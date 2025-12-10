import os
from dotenv import load_dotenv
load_dotenv()

from embeddings_client import get_embedding
from pinecone_client import PineconeClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))

if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You must answer strictly using ONLY the provided context. If not found, respond: "
    "\"I don't know — the document does not contain this information.\""
)

def retrieve_context(query, top_k=TOP_K):
    emb = get_embedding(query)
    pc = PineconeClient()
    res = pc.query(emb, top_k=top_k)
    matches = []
    for m in res.get("matches", []):
        matches.append({
            "id": m["id"],
            "score": m.get("score"),
            "text": m.get("metadata", {}).get("text", ""),
            "meta": m.get("metadata", {})
        })
    return matches

def build_prompt(matches, user_query):
    context_parts = []
    for m in matches:
        context_parts.append(f"[{m['id']} SCORE:{m['score']}]\n{m['text']}\n")
    context = "\n".join(context_parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION: {user_query}"}
    ]

def generate_answer(query):
    matches = retrieve_context(query)
    messages = build_prompt(matches, query)

    if not OPENAI_API_KEY:
        return {
            "answer": "No LLM configured.",
            "retrieved": matches,
            "confidence": float(sum([m["score"] or 0 for m in matches])/len(matches) if matches else 0)
        }

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=512
    )
    answer = resp["choices"][0]["message"]["content"].strip()
    scores = [m["score"] or 0 for m in matches]
    conf = float(sum(scores)/len(scores)) if scores else 0

    return {"answer": answer, "retrieved": matches, "confidence": conf}
