#!/bin/bash
echo "🚀 Setting up AI Core 2026 RAG System..."

# Create the full server.py
cat > src/api/server.py << 'PYTHON'
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai
import os
from typing import List
import uuid

app = FastAPI()

# Initialize clients
client = QdrantClient(host="qdrant", port=6333)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Collection name
COLLECTION_NAME = "ai_docs_2026"

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            print(f"📁 Collection exists: {COLLECTION_NAME}")
    except Exception as e:
        print(f"❌ Startup error: {e}")

async def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for text"""
    try:
        response = await openai.Embedding.acreate(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "system": "AI Core 2026 - RAG System",
        "status": "operational",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "add_document": "POST /add_document",
            "search": "GET /search?query=YOUR_QUERY",
            "embed": "GET /embed?text=YOUR_TEXT"
        }
    }

@app.get("/health")
async def health():
    try:
        collections = client.get_collections()
        qdrant_status = "connected" if collections else "disconnected"
        openai_status = "ready" if os.getenv("OPENAI_API_KEY") else "no_key"
        
        return {
            "status": "healthy",
            "services": {
                "qdrant": qdrant_status,
                "openai": openai_status
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/add_document")
async def add_document(text: str, metadata: dict = None):
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        embedding = await get_embedding(text)
        doc_id = str(uuid.uuid4())
        
        payload = {
            "text": text,
            "metadata": metadata or {},
            "timestamp": "2026"
        }
        
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload=payload
        )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

@app.get("/search")
async def search(query: str, limit: int = 5):
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        query_embedding = await get_embedding(query)
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": round(hit.score, 4),
                "text": hit.payload.get("text", ""),
                "preview": hit.payload.get("text", "")[:150] + "..." if len(hit.payload.get("text", "")) > 150 else hit.payload.get("text", "")
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/embed")
async def embed(text: str = "Hello AI 2026"):
    try:
        embedding = await get_embedding(text)
        return {
            "text": text,
            "embedding_dimensions": len(embedding),
            "sample": embedding[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
PYTHON

echo "✅ Created full RAG server.py"

# Update requirements
echo "fastapi==0.104.0" > requirements.txt
echo "uvicorn[standard]==0.24.0" >> requirements.txt
echo "qdrant-client==1.6.0" >> requirements.txt
echo "openai==1.3.0" >> requirements.txt
echo "numpy==1.24.0" >> requirements.txt

echo "✅ Updated requirements.txt"

echo ""
echo "🔨 Now rebuild and restart:"
echo "docker compose build --no-cache"
echo "docker compose up -d"
echo ""
echo "📚 Test endpoints:"
echo "curl http://localhost:8000/"
echo "curl http://localhost:8000/health"
echo ""
echo "🎉 AI Core 2026 RAG System ready!"
