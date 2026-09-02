"""Runtime configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

APP_URL = os.getenv("APP_URL", "https://github.com/rishi-sharma/darglobal-wasalt-chatbot")
APP_TITLE = "DarGlobal & Wasalt Property Assistant"


def _models(env: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(env, default).split(",") if m.strip()]


def _real_key(v: str) -> bool:
    return bool(v) and "..." not in v and len(v) > 12


# LLM providers, tried in order. Each is an OpenAI-compatible chat endpoint.
# OpenRouter (free models) is the assignment requirement and comes first;
# Sarvam AI is an optional extra fallback when a key is provided.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

PROVIDERS: list[dict] = []
if _real_key(OPENROUTER_API_KEY):
    PROVIDERS.append({
        "name": "openrouter",
        "base": os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1"),
        "key": OPENROUTER_API_KEY,
        "models": _models(
            "OPENROUTER_MODELS",
            "google/gemma-4-31b-it:free,z-ai/glm-5.2:free,nvidia/nemotron-3-super-120b-a12b:free,minimax/minimax-m2.7:free",
        ),
        "headers": {"HTTP-Referer": APP_URL, "X-Title": APP_TITLE},
    })
if _real_key(SARVAM_API_KEY):
    PROVIDERS.append({
        "name": "sarvam",
        "base": os.getenv("SARVAM_BASE", "https://api.sarvam.ai/v1"),
        "key": SARVAM_API_KEY,
        "models": _models("SARVAM_MODELS", "sarvam-105b-conversations"),
        "headers": {},
    })

MODEL_CHAIN = [f"{p['name']}/{m}" for p in PROVIDERS for m in p["models"]]

INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

TOP_K = int(os.getenv("TOP_K", "8"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
