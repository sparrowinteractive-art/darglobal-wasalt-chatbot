"""OpenRouter chat client with a free-model fallback chain and streaming."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from . import config

log = logging.getLogger("llm")

RETRYABLE = {402, 408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.APP_URL,
        "X-Title": config.APP_TITLE,
    }


async def stream_chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> AsyncIterator[tuple[str, str]]:
    """Yield ("model", name) once, then ("token", text) chunks.

    Tries each model in config.OPENROUTER_MODELS in order; moves to the next
    on rate limits, provider errors, or an empty response. Raises LLMError if
    every model fails.
    """
    if not config.OPENROUTER_API_KEY:
        raise LLMError("OPENROUTER_API_KEY is not set")
    errors = []
    async with httpx.AsyncClient(base_url=config.OPENROUTER_BASE, timeout=httpx.Timeout(90, connect=15)) as client:
        for model in config.OPENROUTER_MODELS:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            produced = False
            try:
                async with client.stream("POST", "/chat/completions", headers=_headers(), json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="ignore")[:300]
                        errors.append(f"{model}: HTTP {resp.status_code} {body}")
                        log.warning("model %s failed: %s %s", model, resp.status_code, body)
                        if resp.status_code in RETRYABLE or resp.status_code == 404:
                            continue
                        raise LLMError(errors[-1])
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("error"):
                            errors.append(f"{model}: {obj['error']}")
                            log.warning("model %s mid-stream error: %s", model, obj["error"])
                            break
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            if not produced:
                                produced = True
                                yield ("model", model)
                            yield ("token", text)
                if produced:
                    return
            except (httpx.HTTPError, httpx.StreamError) as exc:
                errors.append(f"{model}: {exc!r}")
                log.warning("model %s transport error: %r", model, exc)
                if produced:
                    return
                continue
    raise LLMError("All models failed: " + " | ".join(errors))


async def complete(messages: list[dict], **kw) -> tuple[str, str]:
    """Non-streaming helper: returns (model, full_text)."""
    model, parts = "", []
    async for kind, val in stream_chat(messages, **kw):
        if kind == "model":
            model = val
        else:
            parts.append(val)
    return model, "".join(parts)
