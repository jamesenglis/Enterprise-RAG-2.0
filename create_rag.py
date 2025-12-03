import os

rag_code = '''from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai
import os
from typing import List
import uuid

app = FastAPI()
client = QdrantClient(host="qdrant", port=6333)
openai.api_key = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "ai_docs_2026"

@app.on_event("startup")
async def startup_event():
    try:
        collections = client.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print("Created collection")
        else:
            print("Collection exists")
    except Exception as e:
        print(f"Error: {e}")

async def get_embedding(text: str) -> List[float]:
    try:
        response = await openai.Embedding.acreate(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed")

@app.get("/")
async def root():
    return {"system": "AI Core 2026 RAG", "status": "ready"}

@app.get("/health")
async def health():
    return {"status": "healthy", "services": "ok"}

@app.post("/add_document")
async def add_document(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    embedding = await get_embedding(text)
    doc_id = str(uuid.uuid4())
    point = PointStruct(
        id=doc_id,
        vector=embedding,
        payload={"text": text}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"status": "added", "id": doc_id}

@app.get("/search")
async def search(query: str, limit: int = 3):
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    query_embedding = await get_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )
    return {
        "query": query,
        "results": [
            {"id": hit.id, "score": hit.score, "text": hit.payload["text"][:100]}
            for hit in results
        ]
    }

@app.get("/embed")
async def embed(text: str = "Hello"):
    embedding = await get_embedding(text)
    return {
        "text": text,
        "dimensions": len(embedding),
        "sample": embedding[:3]
    }
'''

# Write to file
os.makedirs("src/api", exist_ok=True)
with open("src/api/server.py", "w") as f:
    f.write(rag_code)

print("✅ Created src/api/server.py")

# Create requirements
requirements = """fastapi==0.104.0
uvicorn[standard]==0.24.0
qdrant-client==1.6.0
openai==1.3.0
numpy==1.24.0"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("✅ Created requirements.txt")
print("\n🔨 Now run:")
print("docker compose build --no-cache")
print("docker compose up -d")
