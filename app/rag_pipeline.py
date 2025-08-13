import os
import re
import requests
import numpy as np
import pandas as pd
from typing import Optional
from bs4 import BeautifulSoup
from unidecode import unidecode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from .settings import CONFIG

class MesoRAG:
    def __init__(self):
        self._qe = None
        self._catalog = None

    def setup_models(self):
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(CONFIG.LLM_ID)
        llm_model = AutoModelForCausalLM.from_pretrained(
            CONFIG.LLM_ID,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            # trust_remote_code=True, 
        )
        Settings.llm = HuggingFaceLLM(
            model=llm_model,
            tokenizer=tok,
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name=CONFIG.EMB_ID)
        Settings.context_window = 2048
        Settings.generate_kwargs = {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_new_tokens": 320,
        }

    def fetch_html(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (edu; meso-rag)"}
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.text

    def clean_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        return text

    def parse_tables(self, html: str) -> pd.DataFrame:
        try:
            tables = pd.read_html(html)
        except ValueError:
            tables = []
        if not tables:
            return pd.DataFrame(columns=["name","domain","description","culture","type","aka","name_norm"])
        def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
            mapping = {}
            for c in df.columns:
                lc = str(c).strip().lower()
                if "name" in lc:
                    mapping[c] = "name"
                elif any(k in lc for k in ["domain","sphere","portfolio","function"]):
                    mapping[c] = "domain"
                elif any(k in lc for k in ["description","notes","remarks"]):
                    mapping[c] = "description"
                elif any(k in lc for k in ["culture","sumer","akkad","assyri","babylon","cult"]):
                    mapping[c] = "culture"
                elif any(k in lc for k in ["type","category","class"]):
                    mapping[c] = "type"
                elif any(k in lc for k in ["aka","also","alt","alternate","epithet"]):
                    mapping[c] = "aka"
                else:
                    mapping[c] = lc
            return df.rename(columns=mapping)
        cands = []
        for df in [normalize_cols(d) for d in tables]:
            if "name" in df.columns:
                cols = [c for c in df.columns if c in {"name","domain","description","culture","type","aka"}]
                if len(cols) >= 2:
                    cands.append(df[cols].copy())
        if not cands:
            return pd.DataFrame(columns=["name","domain","description","culture","type","aka","name_norm"])
        cat = pd.concat(cands, ignore_index=True).dropna(how="all")
        for c in cat.columns:
            cat[c] = cat[c].astype(str).strip()
        cat["name_norm"] = cat["name"].apply(lambda s: unidecode(str(s)).lower())
        cat.sort_values(by=["name_norm","description"], key=lambda s: s.str.len(), ascending=[True, False], inplace=True)
        cat = cat.drop_duplicates(subset=["name_norm"], keep="first")
        return cat

    def build_catalog_profiles(self, cat: pd.DataFrame) -> pd.DataFrame:
        if cat.empty:
            return cat
        def make_profile(row):
            parts = []
            for c in ["name","aka","domain","description","culture","type"]:
                val = row.get(c, "")
                if isinstance(val, str) and val.strip():
                    parts.append(f"{c}: {val}")
            return " | ".join(parts)
        out = cat.copy()
        out["profile"] = out.apply(make_profile, axis=1)
        return out

    def df_to_documents(self, df: pd.DataFrame) -> list[Document]:
        docs = []
        for i, row in df.iterrows():
            name = str(row.get("name","")).strip()
            profile = str(row.get("profile","")).strip()
            if not profile:
                continue
            text = profile
            md = {"source": CONFIG.URL, "row_idx": int(i), "name": name}
            docs.append(Document(text=text, metadata=md))
        return docs

    def build_index(self, page_text: str, catalog: pd.DataFrame):
        dim = len(Settings.embed_model.get_text_embedding("dim?"))
        faiss_index = faiss.IndexFlatIP(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage = StorageContext.from_defaults(vector_store=vector_store)
        docs = self.df_to_documents(catalog) + [Document(text=page_text, metadata={"source": CONFIG.URL})]
        index = VectorStoreIndex.from_documents(docs, storage_context=storage)
        qe = index.as_query_engine(similarity_top_k=CONFIG.TOP_K)
        index.storage_context.persist(persist_dir=CONFIG.STORAGE_DIR)
        self._qe = qe
        return index, qe

    def ensure_loaded(self):
        if self._qe is None:
            self.load()
        return self._qe

    def search_catalog(self, query: str, k: int = 3) -> list[dict]:
        cat = self._catalog
        if cat is None or cat.empty:
            return []
        texts = cat["profile"].tolist()
        vecs = [Settings.embed_model.get_text_embedding(t) for t in texts]
        M = np.array(vecs, dtype="float32"); M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        qv = np.array(Settings.embed_model.get_text_embedding(query), dtype="float32")[None, :]; qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        sims = (M @ qv.T).squeeze(-1)
        top = np.argsort(-sims)[:k]
        out = []
        for i in top:
            row = cat.iloc[int(i)]
            out.append({"name": row.get("name",""), "domain": row.get("domain",""), "score": float(sims[int(i)])})
        return out

    def extract_name_from_query(self, q: str) -> Optional[str]:
        cat = self._catalog
        if cat is None or cat.empty: return None
        qn = unidecode(q).lower()
        names = cat["name"].astype(str).tolist()
        names_norm = {unidecode(n).lower(): n for n in names}
        hits = [orig for norm, orig in names_norm.items() if norm in qn and len(norm) >= 3]
        return hits[0] if hits else None

    def load(self):
        self.setup_models()
        html = self.fetch_html(CONFIG.URL)
        page_text = self.clean_text_from_html(html)
        catalog = self.parse_tables(html)
        catalog = self.build_catalog_profiles(catalog)
        self._catalog = catalog
        _, qe = self.build_index(page_text, catalog)
        return qe

    def answer(self, question: str, lang: str = "en") -> dict:
        qe = self.ensure_loaded()
        if lang.lower().startswith("pt"):
            sys_rules = ("Responda em português. Baseie-se apenas no contexto fornecido (a página da Wikipédia e a tabela). "
                         "Se não houver informação, diga que não encontrou. Seja conciso.")
        else:
            sys_rules = ("Answer in English. Ground ONLY in the provided context (the Wikipedia page and the catalog). "
                         "If missing, say you didn't find it. Be concise.")
        name_hit = self.extract_name_from_query(question)
        if name_hit:
            prompt = f"{sys_rules}\n\nUser question: Talk about the deity '{name_hit}': origin, domains/attributes, and relationships.\nSource: {CONFIG.URL}"
            resp = qe.query(prompt)
            return {"answer": str(resp), "suggestions": [], "matched_name": name_hit}
        suggestions = self.search_catalog(question, k=3)
        rag_prompt = f"{sys_rules}\n\nUser question: Based on the context, which deity best matches: {question}? Justify in 1–3 sentences.\nSource: {CONFIG.URL}"
        resp = qe.query(rag_prompt)
        return {"answer": str(resp), "suggestions": suggestions, "matched_name": None}

rag = MesoRAG()
