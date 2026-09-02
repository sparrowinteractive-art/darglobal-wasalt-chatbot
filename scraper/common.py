"""Shared helpers for the scrapers."""
from __future__ import annotations

import json
import re
import time
from html import unescape
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n\s*\n+")


def clean_html(text: str | None) -> str:
    """Strip tags and collapse whitespace, keeping paragraph breaks."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<\s*(br|/p|/div|/li|/h\d)\s*/?>", "\n", text, flags=re.I)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text).replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def rich_text(value) -> str:
    """Flatten Strapi rich-text blocks (lists of {type, children:[{text}]}) or HTML into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_html(value)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return clean_html(value["text"])
        return " ".join(t for t in (rich_text(v) for v in value.values()) if t)
    if isinstance(value, list):
        lines = []
        for block in value:
            if isinstance(block, dict) and isinstance(block.get("children"), list):
                lines.append(" ".join(t for t in (rich_text(c) for c in block["children"]) if t))
            else:
                lines.append(rich_text(block))
        return "\n".join(l for l in lines if l)
    return clean_html(str(value))


def save_json(name: str, data) -> Path:
    path = RAW_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    return path


def load_json(name: str, default=None):
    path = RAW_DIR / name
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def extract_next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        return None
    return json.loads(m.group(1))


class Throttle:
    """Simple polite delay between requests."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        gap = now - self._last
        if gap < self.seconds:
            time.sleep(self.seconds - gap)
        self._last = time.monotonic()
