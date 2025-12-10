from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import generate_answer
import uvicorn

class QueryIn(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask")
async def ask(q: QueryIn):
    res = generate_answer(q.question)
    ret = []
    for m in res["retrieved"]:
        txt = m["text"][:1200] + "..." if len(m["text"]) > 1200 else m["text"]
        ret.append({"id": m["id"], "score": m["score"], "text": txt})
    return {"question": q.question, "answer": res["answer"], "retrieved": ret, "confidence": res["confidence"]}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
