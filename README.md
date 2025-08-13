# MesoDeities — LLM + RAG (FastAPI + Docker)
Run locally:
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000

Docker:
  docker build -t meso-deities .
  docker run --rm -p 8000:8000 meso-deities
