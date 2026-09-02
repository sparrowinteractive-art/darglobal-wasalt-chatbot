"""Chat client for OpenAI-compatible providers with a model fallback chain and streaming.

Providers (OpenRouter free models first, then any optional extra provider)
come from ``config.PROVIDERS``. Each model is tried in order; on rate limits,
provider errors, or an empty response the next one is used.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from . import config

log = logging.getLogger("llm")

RETRYABLE = {402, 404, 408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


async def _stream_one(client: httpx.AsyncClient, provider: dict, model: str, messages: list[dict], temperature: float, max_tokens: int):
    """Yield content tokens for one provider/model; raise LLMError on failure."""
    headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json", **provider.get("headers", {})}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
    async with client.stream("POST", provider["base"] + "/chat/completions", headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="ignore")[:300]
            raise LLMError(f"HTTP {resp.status_code} {body}", ) from None
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
                raise LLMError(f"mid-stream error {obj['error']}")
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text


async def stream_chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> AsyncIterator[tuple[str, str]]:
    """Yield ("model", "provider/model") once, then ("token", text) chunks.

    Raises LLMError if every provider/model combination fails.
    """
    if not config.PROVIDERS:
        raise LLMError("No LLM provider configured: set OPENROUTER_API_KEY (or SARVAM_API_KEY)")
    errors = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
        for provider in config.PROVIDERS:
            for model in provider["models"]:
                label = f"{provider['name']}/{model}"
                produced = False
                try:
                    async for text in _stream_one(client, provider, model, messages, temperature, max_tokens):
                        if not produced:
                            produced = True
                            yield ("model", label)
                        yield ("token", text)
                    if produced:
                        return
                    errors.append(f"{label}: empty response")
                except LLMError as exc:
                    errors.append(f"{label}: {exc}")
                    log.warning("model %s failed: %s", label, exc)
                    if produced:
                        return  # partial answer already streamed; do not restart with another model
                except (httpx.HTTPError, httpx.StreamError) as exc:
                    errors.append(f"{label}: {exc!r}")
                    log.warning("model %s transport error: %r", label, exc)
                    if produced:
                        return
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
