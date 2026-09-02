"""Hybrid retrieval (dense + BM25) over the scraped knowledge base and prompt assembly."""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config

log = logging.getLogger("rag")

SYSTEM_PROMPT = """You are the property assistant for two real-estate sources:
1. DarGlobal (darglobal.co.uk) - an international luxury developer with branded projects in the UAE, Saudi Arabia, Oman, Qatar, Spain, the UK and the Maldives.
2. Wasalt (wasalt.sa) - a Saudi Arabian property marketplace with listings for sale and rent.

Rules:
- Answer ONLY from the CONTEXT passages below. They were scraped from the two public websites on 2026-09-02.
- If the context does not contain the answer, say you don't have that information in your data and suggest what the user could ask instead. Never invent projects, prices, dates or listings.
- Be concise and factual. Use short paragraphs or bullet points. Prices are in SAR for Wasalt unless stated otherwise.
- When you mention a specific project or listing, cite its source with a markdown link using the URL given in the passage, e.g. [Trump Tower Jeddah](https://...).
- Wasalt listings are a sample of the marketplace, not the full inventory. Market snapshot figures are computed from that sample, say so when quoting them.
- Do not give legal or financial advice; you may summarise what the websites say about investment benefits.
- Reply in the language of the user's question (English or Arabic)."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9؀-ۿ]+", text.lower())


class KnowledgeBase:
    def __init__(self):
        docs_path = config.INDEX_DIR / "docs.json"
        self.docs: list[dict] = json.loads(docs_path.read_text(encoding="utf-8"))
        self.by_id = {d["id"]: d for d in self.docs}
        self.bm25 = BM25Okapi([_tokenize(d["text"]) for d in self.docs])
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.emb = np.load(config.INDEX_DIR / "embeddings.npy")
        self.sources = np.array([d["meta"]["source"] for d in self.docs])
        self.kinds = np.array([d["meta"].get("kind", "") for d in self.docs])
        self.ptypes = np.array([d["meta"].get("property_type", "") for d in self.docs])
        self.cities = np.array([d["meta"].get("city", "") for d in self.docs])
        self.purposes = np.array([d["meta"].get("purpose", "") for d in self.docs])
        log.info("knowledge base ready: %d documents", len(self.docs))

    # ------------------------------------------------------------ retrieval
    def _dense(self, query: str, k: int, source: str | None) -> list[str]:
        q = self.embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.emb @ q
        if source:
            scores = np.where(self.sources == source, scores, -1.0)
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [self.docs[i]["id"] for i in top if scores[i] > 0]

    def _sparse(self, query: str, k: int, source: str | None) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            d = self.docs[i]
            if source and d["meta"]["source"] != source:
                continue
            out.append(d["id"])
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _source_filter(query: str) -> str | None:
        q = query.lower()
        dar = any(w in q for w in ("darglobal", "dar global", "dar al arkan"))
        was = "wasalt" in q
        if dar and not was:
            return "darglobal"
        if was and not dar:
            return "wasalt"
        return None

    # simple intent detection for Wasalt listing questions
    TYPE_WORDS = {
        "apartment": "Apartment", "apartments": "Apartment", "flat": "Apartment", "flats": "Apartment",
        "villa": "Villa", "villas": "Villa", "land": "Land", "plot": "Land", "plots": "Land",
        "building": "Building", "buildings": "Building", "floor": "Floor", "floors": "Floor",
        "rest house": "Rest House", "istiraha": "Rest House", "room": "Room", "rooms": "Room",
        "shop": "Shop", "office": "Office", "warehouse": "Warehouse", "chalet": "Chalet",
    }
    CITY_WORDS = {
        "riyadh": "Riyadh", "jeddah": "Jeddah", "makkah": "Makkah", "mecca": "Makkah", "madinah": "Madinah",
        "medina": "Madinah", "dammam": "Dammam", "khobar": "Al Khobar", "dhahran": "Dhahran", "taif": "Taif",
        "buraidah": "Buraidah", "abha": "Abha", "tabuk": "Tabuk", "jubail": "Jubail", "ahsa": "Al Ahsa",
        "hofuf": "Al Ahsa", "yanbu": "Yanbu", "hail": "Hail",
    }

    def _structured(self, query: str, k: int) -> list[str]:
        """Return listing ids whose metadata matches the city / type / purpose named in the query."""
        q = query.lower()
        purpose = "rent" if re.search(r"\b(rent|rental|lease|renting)\b", q) else ("sale" if re.search(r"\b(sale|buy|purchase|buying)\b", q) else None)
        ptype = next((v for w, v in self.TYPE_WORDS.items() if re.search(rf"\b{w}\b", q)), None)
        city = next((v for w, v in self.CITY_WORDS.items() if w in q), None)
        if not (ptype or city):
            return []
        mask = self.kinds == "listing"
        if ptype:
            mask &= self.ptypes == ptype
        if city:
            mask &= self.cities == city
        if purpose:
            mask &= self.purposes == purpose
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        qv = self.embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.emb[idx] @ qv
        order = idx[np.argsort(-scores)][:k]
        return [self.docs[i]["id"] for i in order]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        source = self._source_filter(query)
        dense = self._dense(query, k * 2, source)
        sparse = self._sparse(query, k * 2, source)
        structured = self._structured(query, 4) if source != "darglobal" else []
        # reciprocal rank fusion
        fused: dict[str, float] = {}
        for rank, did in enumerate(dense):
            fused[did] = fused.get(did, 0) + 1 / (60 + rank)
        for rank, did in enumerate(sparse):
            fused[did] = fused.get(did, 0) + 1 / (60 + rank)
        for rank, did in enumerate(structured):
            fused[did] = fused.get(did, 0) + 1 / (30 + rank)  # metadata match is a strong signal
        ranked = sorted(fused, key=lambda d: -fused[d])
        # make sure both sources appear when the question is not source-specific,
        # and never let one long page (many chunks) fill the whole context
        picked, seen_sources, per_url = [], set(), {}
        for did in ranked:
            d = self.by_id[did]
            url = d["meta"]["url"]
            if per_url.get(url, 0) >= 2:
                continue
            per_url[url] = per_url.get(url, 0) + 1
            picked.append(d)
            seen_sources.add(d["meta"]["source"])
            if len(picked) >= k:
                break
        if not source and len(seen_sources) < 2:
            for did in ranked[k:]:
                if self.by_id[did]["meta"]["source"] not in seen_sources:
                    picked.append(self.by_id[did])
                    break
        return picked

    # --------------------------------------------------------------- prompt
    @staticmethod
    def build_messages(question: str, passages: list[dict], history: list[dict]) -> list[dict]:
        ctx_lines = []
        for i, p in enumerate(passages, 1):
            m = p["meta"]
            ctx_lines.append(f"[{i}] source={m['source']} title={m.get('title')} url={m['url']}\n{p['text']}")
        context = "\n\n".join(ctx_lines) or "(no passages found)"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-config.MAX_HISTORY:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                msgs.append({"role": h["role"], "content": str(h["content"])[:2000]})
        msgs.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})
        return msgs


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
