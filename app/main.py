from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from .settings import CONFIG
from .rag_pipeline import rag

app = FastAPI(title="MesoDeities RAG API", version="1.0")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")

@app.get("/", response_class=HTMLResponse)
def home():
  with open("frontend/index.html", "r", encoding="utf-8") as f:
    return HTMLResponse(content=f.read(), status_code=200)

@app.get("/health")
def health():
  return {"status": "ok", "build_on_start": CONFIG.BUILD_ON_START}

class QueryIn(BaseModel):
  question: str
  lang: Optional[str] = "en"

@app.post("/api/query")
def api_query(payload: QueryIn):
  return rag.answer(payload.question, lang=payload.lang or "en")

@app.post("/api/reload")
def api_reload():
  rag.load()
  return {"status": "reloaded"}

@app.on_event("startup")
def on_start():
  if CONFIG.BUILD_ON_START:
    rag.load()
