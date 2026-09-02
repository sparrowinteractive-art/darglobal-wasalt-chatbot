"""Runtime configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "google/gemma-4-31b-it:free,z-ai/glm-5.2:free,nvidia/nemotron-3-super-120b-a12b:free,minimax/minimax-m2.7:free",
    ).split(",")
    if m.strip()
]
APP_URL = os.getenv("APP_URL", "https://github.com/rishi-sharma/darglobal-wasalt-chatbot")
APP_TITLE = "DarGlobal & Wasalt Property Assistant"

INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

TOP_K = int(os.getenv("TOP_K", "8"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
