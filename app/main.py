"""FastAPI application: chat API (streaming and plain) plus the bundled web UI."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm
from .rag import get_kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

app = FastAPI(title=config.APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)


def _sources(passages: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in passages:
        m = p["meta"]
        if m["url"] in seen:
            continue
        seen.add(m["url"])
        out.append({"title": m.get("title") or m["url"], "url": m["url"], "source": m["source"], "kind": m.get("kind")})
    return out


@app.on_event("startup")
async def _warm():
    get_kb()


@app.get("/health")
async def health():
    kb = get_kb()
    return {
        "status": "ok",
        "documents": len(kb.docs),
        "models": config.OPENROUTER_MODELS,
        "llm_key_configured": bool(config.OPENROUTER_API_KEY),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Plain JSON answer (useful for curl and automated tests)."""
    kb = get_kb()
    t0 = time.time()
    passages = kb.search(req.message)
    messages = kb.build_messages(req.message, passages, [h.model_dump() for h in req.history])
    try:
        model, answer = await llm.complete(messages)
    except llm.LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"answer": answer, "model": model, "sources": _sources(passages), "latency_ms": int((time.time() - t0) * 1000)}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-sent events: `sources` first, then `token` events, then `done`."""
    kb = get_kb()
    passages = kb.search(req.message)
    messages = kb.build_messages(req.message, passages, [h.model_dump() for h in req.history])

    async def gen():
        yield f"event: sources\ndata: {json.dumps(_sources(passages))}\n\n"
        try:
            async for kind, val in llm.stream_chat(messages):
                if kind == "model":
                    yield f"event: model\ndata: {json.dumps(val)}\n\n"
                else:
                    yield f"event: token\ndata: {json.dumps(val)}\n\n"
        except llm.LLMError as exc:
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/search")
async def search(q: str, k: int = 6):
    """Debug endpoint: see which passages the retriever picks for a query."""
    kb = get_kb()
    return [{"id": p["id"], "meta": p["meta"], "text": p["text"][:400]} for p in kb.search(q, k)]


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
