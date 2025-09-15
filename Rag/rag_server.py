import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = "AIzaSyDjN4JnStnyC6fiAGPnWHlCKboarg7p-5g"
OUTPUT_API = "http://localhost:9000/receive"  # external endpoint to POST answers
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# -----------------------------
# LOAD KNOWLEDGE BASE FROM TXT
# -----------------------------
with open("personal_details.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(documents)} personal details from file.")

# -----------------------------
# EMBEDDINGS + FAISS
# -----------------------------
embed_model = "models/embedding-001"
dimension = 768  # Gemini embedding size
index = faiss.IndexFlatL2(dimension)
doc_embeddings = []

print("🔄 Creating embeddings for documents...")
for doc in documents:
    try:
        emb = genai.embed_content(model=embed_model, content=doc)["embedding"]
        doc_embeddings.append(emb)
    except Exception as e:
        print(f"❌ Error embedding doc '{doc}':", e)

if not doc_embeddings:
    raise RuntimeError("No embeddings generated. Check your Gemini API key and model.")

index.add(np.array(doc_embeddings, dtype="float32"))
print("✅ Embeddings created and added to FAISS index.")

# -----------------------------
# SCHEMA
# -----------------------------
class QueryRequest(BaseModel):
    query: str

class InsertRequest(BaseModel):
    detail: str

class UpdateRequest(BaseModel):
    index: int
    new_detail: str

# -----------------------------
# RAG PIPELINE
# -----------------------------
def rag_answer(query: str) -> str:
    try:
        # Embed query
        q_emb = genai.embed_content(model=embed_model, content=query)["embedding"]
        q_emb = np.array([q_emb], dtype="float32")

        # Search top-1 doc
        D, I = index.search(q_emb, k=1)
        retrieved_doc = documents[I[0][0]]

        print(f"🔎 Retrieved doc: {retrieved_doc}")

        # Build prompt
        prompt = f"""
You are an assistant with access to personal details.
User query: {query}
Relevant info: {retrieved_doc}
Answer the query based ONLY on the relevant info.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)

        if resp and hasattr(resp, "text"):
            return resp.text.strip()
        else:
            return "⚠️ No response from Gemini model."

    except Exception as e:
        print("❌ Error in rag_answer:", e)
        return f"Error: {e}"

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.post("/ask")
async def ask_question(req: QueryRequest):
    query = req.query
    print(f"👉 Received query: {query}")

    answer = rag_answer(query)
    print(f"✅ Answer: {answer}")

    # POST to external API
    try:
        requests.post(OUTPUT_API, json={"query": query, "answer": answer}, timeout=3)
    except Exception as e:
        print("⚠️ Failed to POST to external API:", e)

    return {"query": query, "answer": answer}


@app.post("/insert")
async def insert_detail(req: InsertRequest):
    detail = req.detail
    documents.append(detail)

    # Embed and add to FAISS
    try:
        emb = genai.embed_content(model=embed_model, content=detail)["embedding"]
        index.add(np.array([emb], dtype="float32"))
        print(f"✅ Inserted new detail: {detail}")
    except Exception as e:
        print("❌ Error embedding new detail:", e)
        return {"status": "error", "message": str(e)}

    return {"status": "ok", "message": "Detail inserted", "total": len(documents)}


@app.post("/update")
async def update_detail(req: UpdateRequest):
    idx = req.index
    if idx < 0 or idx >= len(documents):
        return {"status": "error", "message": "Invalid index"}

    documents[idx] = req.new_detail
    print(f"✏️ Updated index {idx} with new detail: {req.new_detail}")

    # Rebuild FAISS index (since updating one vector in FlatL2 is not straightforward)
    try:
        new_embeddings = []
        for doc in documents:
            emb = genai.embed_content(model=embed_model, content=doc)["embedding"]
            new_embeddings.append(emb)

        # Reset FAISS
        global index
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(new_embeddings, dtype="float32"))
        print("✅ FAISS index rebuilt after update.")

    except Exception as e:
        print("❌ Error rebuilding FAISS index:", e)
        return {"status": "error", "message": str(e)}

    return {"status": "ok", "message": "Detail updated", "total": len(documents)}


@app.post("/receive")
async def receive_answer(request: Request):
    data = await request.json()
    print("📩 Received at /receive:", data)
    return {"status": "ok", "data": data}


@app.get("/receive")
async def get_all_details():
    return {"stored_details": documents, "total": len(documents)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
