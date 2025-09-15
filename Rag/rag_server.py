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
DATA_FILE = "personal_details.txt"

genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI()

# -----------------------------
# LOAD KNOWLEDGE BASE FROM TXT
# -----------------------------
def load_documents():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_documents():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

documents = load_documents()
print(f"✅ Loaded {len(documents)} personal details from file.")

# -----------------------------
# EMBEDDINGS + FAISS
# -----------------------------
embed_model = "models/embedding-001"
dimension = 768  # Gemini embedding size

def build_faiss_index(docs):
    new_embeddings = []
    for doc in docs:
        emb = genai.embed_content(model=embed_model, content=doc)["embedding"]
        new_embeddings.append(emb)

    idx = faiss.IndexFlatL2(dimension)
    idx.add(np.array(new_embeddings, dtype="float32"))
    return idx

index = build_faiss_index(documents)
print("✅ FAISS index built.")

# -----------------------------
# SCHEMAS
# -----------------------------
class QueryRequest(BaseModel):
    query: str

class InsertRequest(BaseModel):
    detail: str

class UpdateRequest(BaseModel):
    index: int
    new_detail: str

class DeleteRequest(BaseModel):
    index: int

# -----------------------------
# RAG PIPELINE
# -----------------------------
def rag_answer(query: str) -> str:
    q_emb = genai.embed_content(model=embed_model, content=query)["embedding"]
    q_emb = np.array([q_emb], dtype="float32")

    # Search top-1 doc
    D, I = index.search(q_emb, k=1)
    retrieved_doc = documents[I[0][0]]

    print(f"🔎 Retrieved doc: {retrieved_doc}")

    prompt = f"""
You are an assistant with access to personal details.
User query: {query}
Relevant info: {retrieved_doc}
Answer the query based ONLY on the relevant info.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip() if resp and hasattr(resp, "text") else "⚠️ No response from Gemini."

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.post("/ask")
async def ask_question(req: QueryRequest):
    answer = rag_answer(req.query)
    try:
        requests.post(OUTPUT_API, json={"query": req.query, "answer": answer}, timeout=3)
    except Exception as e:
        print("⚠️ Failed to POST to external API:", e)
    return {"query": req.query, "answer": answer}


@app.post("/insert")
async def insert_detail(req: InsertRequest):
    documents.append(req.detail)
    save_documents()
    global index
    index = build_faiss_index(documents)
    return {"status": "ok", "message": "Detail inserted", "total": len(documents)}


@app.post("/update")
async def update_detail(req: UpdateRequest):
    if req.index < 0 or req.index >= len(documents):
        return {"status": "error", "message": "Invalid index"}
    documents[req.index] = req.new_detail
    save_documents()
    global index
    index = build_faiss_index(documents)
    return {"status": "ok", "message": "Detail updated", "total": len(documents)}


@app.post("/delete")
async def delete_detail(req: DeleteRequest):
    if req.index < 0 or req.index >= len(documents):
        return {"status": "error", "message": "Invalid index"}
    removed = documents.pop(req.index)
    save_documents()
    global index
    index = build_faiss_index(documents)
    return {"status": "ok", "message": f"Deleted: {removed}", "total": len(documents)}


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
