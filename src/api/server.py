from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root(): return {"ai": "core 2026"}
@app.get("/health")
def health(): return {"status": "healthy"}
