# MesoDeities — LLM + RAG for Mesopotamian (Sumerian) Deities

**Ask about any Mesopotamian deity by name or by attributes** (e.g., “Who is the Sumerian god of knowledge?”) and get concise, grounded answers.

Built as a learn-by-building AI app using **Hugging Face** (LLM + embeddings), **LlamaIndex** (RAG orchestration), **FAISS** (vector search), **BeautifulSoup/pandas** (scraping), and **FastAPI** (serving).  
**No OpenAI required.**

<p align="center">
  <img alt="MesoDeities UI" src="https://dummyimage.com/1200x520/0b1020/e6e9f2&text=MesoDeities+UI" />
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-api">API</a> •
  <a href="#-how-it-works">How it works</a> •
  <a href="#-project-structure">Project structure</a> •
  <a href="#-tuning--troubleshooting">Tuning & Troubleshooting</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## ✨ Features

- **RAG pipeline**: scrape & parse Wikipedia → chunk → embed → **vector search (FAISS)** → grounded answer.
- **Two retrieval modes**:
  - **Table-driven** attribute lookup (from parsed HTML tables).
  - **Full-page text** RAG for richer summaries.
- **Multilingual embeddings** (PT ↔ EN out of the box).
- **FastAPI** backend + **elegant HTML/CSS** frontend.
- **Docker-ready** and **Codespaces-friendly**.

---

## 🚀 Quickstart

### Local (Python)
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

### Docker
```bash
# build from local repo
docker build -t meso-deities .
docker run --rm -p 8000:8000 meso-deities

# or build straight from GitHub (main branch)
docker build https://github.com/<your-user>/<your-repo>.git#main -t meso-deities
docker run --rm -p 8000:8000 meso-deities
```

### GitHub Codespaces
1) Open your repo → **Code** → **Create codespace on main**.  
2) In the terminal:
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
3) Forward **port 8000** when prompted and open in the browser.

> ⏳ First run downloads models and builds the index; it may take a few minutes.

---

## 🔧 Configuration

Environment variables (optional):

| Variable | Default | Description |
|---|---|---|
| `LLM_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face LLM (instruction-tuned) |
| `EMB_ID` | `intfloat/multilingual-e5-small` | Embedding model (multilingual, 384-dim) |
| `TOP_K` | `4` | Number of retrieved chunks for RAG |
| `STORAGE_DIR` | `storage_meso` | Index persistence directory |
| `MESO_URL` | Wikipedia list of Mesopotamian deities | Source URL to scrape |
| `BUILD_ON_START` | `true` | Build index at app startup |

Example:
```bash
LLM_ID="microsoft/Phi-3-mini-4k-instruct" EMB_ID="BAAI/bge-m3" TOP_K=5 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 API

- `GET /` — serves the frontend.  
- `GET /health` — health/status.  
- `POST /api/query` — ask the model.  
  - **Body**:
    ```json
    { "question": "Who is the Sumerian god of knowledge?", "lang": "en" }
    ```
  - **Response**:
    ```json
    { "answer": "...", "suggestions": [ { "name": "Enki (Ea)", "domain": "..." } ], "matched_name": "..." }
    ```
- `POST /api/reload` — re-scrape, rebuild, and reload the index.

**cURL**
```bash
curl -s -X POST http://localhost:8000/api/query   -H "Content-Type: application/json"   -d '{"question":"Tell me about Enlil","lang":"en"}' | jq .
```

---

## 🧠 How it works

1. **Scrape** the Wikipedia page with BeautifulSoup/pandas.  
2. **Normalize** tables → build a **catalog** (name/domain/description/aka…).  
3. **Chunk + Embed** both the **catalog profiles** and the **full page text** with HF embeddings.  
4. Store vectors in **FAISS** via **LlamaIndex** `VectorStoreIndex`.  
5. At query time:
   - detect deity name if present → **direct RAG** summary;
   - otherwise do **attribute search** (embeddings-only) to suggest candidates → RAG justifies the best match.  
6. The **LLM** writes a concise answer **grounded strictly** in retrieved context.

---

## 🗂 Project structure

```
app/
  main.py            # FastAPI app & endpoints
  settings.py        # env config (models, URL, TOP_K, etc.)
  rag_pipeline.py    # scraping, catalog, embeddings, FAISS, RAG orchestration
frontend/
  index.html         # elegant, minimal UI
  assets/
    style.css
    app.js
requirements.txt
Dockerfile
```

---

## ⚙️ Tuning & Troubleshooting

**Quality knobs**
- **LLM**: start small (`Qwen2.5-1.5B-Instruct`, `Phi-3-mini`) for CPU/Colab; scale up later.
- **Embeddings**: `multilingual-e5-small` (fast) vs `bge-m3` (stronger, heavier).
- **Chunking**: 256–512 tokens with ~64 overlap.
- **RAG `TOP_K`**: 3–6; too low misses context, too high dilutes it.
- **Reranker**: add a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder top-k before the LLM.

**Common issues**
- **Slow first answer**: models download + index build on first run.  
- **CUDA OOM**: switch to CPU, use a smaller LLM, or set `torch_dtype` to `bfloat16/float16` (GPU).  
- **Weak answers**: increase `TOP_K`, reduce chunk size, add a reranker, tighten the system prompt (“answer ONLY from context”).  
- **Wikipedia table changed**: the parser is heuristic; tweak `parse_tables()` normalization rules.

---

## 🛣 Roadmap

- [ ] Cross-encoder reranker endpoint & UI toggle  
- [ ] Hybrid retrieval (BM25 + dense)  
- [ ] Tiny evaluation harness (hit@K / faithfulness checklist)  
- [ ] Docker multi-arch build (CI to GHCR)

---

## 📝 License & credits

- Code: MIT (add your LICENSE).  
- Uses public models from **Hugging Face** and **LlamaIndex**.  
- Source data: [Wikipedia — List of Mesopotamian deities](https://en.wikipedia.org/wiki/List_of_Mesopotamian_deities).
