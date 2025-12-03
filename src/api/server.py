from fastapi import FastAPI
import openai
import os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
async def root():
    return {"system": "AI Core 2026", "status": "production"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
