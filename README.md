# DarGlobal & Wasalt Property Assistant

An AI chatbot that answers questions about **DarGlobal** luxury developments and
**Wasalt** property listings in Saudi Arabia, using only data scraped from the
two public websites. Retrieval-augmented generation (RAG) keeps the answers
grounded in that data; a free OpenRouter model writes the reply.

| | |
|---|---|
| Live chatbot | `https://<vercel-app>.vercel.app` |
| API (Docker container) | `https://<backend-host>/health` |
| Source | this repository |

## Architecture

```
darglobal.co.uk  ──(Strapi API + headed Chrome)──┐
                                                  ├─►  data/raw/*.json  ──►  ingest/build_index.py  ──►  data/index/
wasalt.sa        ──(Playwright, __NEXT_DATA__)───┘                             bge-small embeddings + BM25    │
                                                                                                              ▼
Browser (Vercel static page) ──► FastAPI /api/chat/stream ──► hybrid retrieval (dense + BM25, RRF) ──► OpenRouter free model
                                    (Docker container)                                                  with fallback chain
```

**Tech stack**

- Python 3.11, FastAPI, uvicorn, httpx
- Playwright (Chrome) for scraping sites behind bot protection
- `BAAI/bge-small-en-v1.5` sentence embeddings (local, CPU) plus `rank-bm25` for keyword recall
- numpy in-memory vector store (a few thousand documents, no database needed)
- OpenRouter free models: `google/gemma-4-31b-it:free` primary, with `z-ai/glm-5.2:free`,
  `nvidia/nemotron-3-super-120b-a12b:free` and `minimax/minimax-m2.7:free` as automatic fallbacks
  when a model is rate limited
- Optional extra provider: Sarvam AI (`sarvam-105b-conversations`) is used as a
  fallback after the OpenRouter chain when `SARVAM_API_KEY` is set. Any
  OpenAI-compatible endpoint can be added the same way in `app/config.py`.
- Docker multi-stage image with the index and embedding model baked in
- Static chat UI deployed on Vercel, calling the containerised API

## Data collected

| Source | What | How | Volume |
|---|---|---|---|
| DarGlobal | Project catalogue (title, country, type, bedrooms, completion date) | Public Strapi endpoint `strapi.darglobal.co.uk/api/projects` | 36 projects, 7 countries |
| DarGlobal | Project pages (about, amenities, why invest, location, FAQs, disclaimers) | Headed Chrome via Playwright, parsing the server-rendered `__NEXT_DATA__` JSON (the site is behind Imperva/Incapsula) | 36 pages |
| DarGlobal | Company pages (about, FAQ, why invest, investor, partners, one-of-one, tokenization...) | same | 24 pages |
| Wasalt | Property listings for sale and rent (price, type, size, beds, district, city, broker type, REGA status) | Playwright, public city and district category pages from the sitemap, parsing `__NEXT_DATA__` | 2,492 listings across 15 cities (plus Riyadh regions) |
| Wasalt | City guide texts and per-city market snapshots (counts, price ranges, medians computed from the sample) | derived | 37 guides + 1 snapshot per city |

Scraping notes: only public pages were used, `robots.txt` was respected
(Wasalt disallows `/search`, which is not used), requests were throttled to a
few seconds apart, and no personal contact data of listing owners is stored.
The Wasalt data is a **sample** of the marketplace, and the chatbot says so.

## Run locally

```bash
cp .env.example .env            # add your OpenRouter key
docker compose up --build       # first build embeds the data, ~5 minutes
open http://localhost:8080
```

Without Docker:

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ingest.build_index
uvicorn app.main:app --reload
```

### Re-scraping

```bash
pip install -r requirements-scrape.txt && playwright install chrome
python -m scraper.darglobal                # opens a Chrome window (Incapsula needs a real browser)
python -m scraper.wasalt --pages 6 --headed
python -m scraper.wasalt --pages 6 --headed --merge --cities north-riyadh,south-riyadh,east-riyadh,west-riyadh,central-riyadh
python -m ingest.build_index
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | status, document count, configured models |
| `POST /api/chat` | `{ "message": "...", "history": [...] }` returns `{ answer, model, sources }` |
| `POST /api/chat/stream` | same input, server-sent events (`sources`, `model`, `token`, `done`) |
| `GET /api/search?q=...` | debug: passages the retriever selects |

```bash
curl -s localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"message":"Which DarGlobal projects are in Oman?"}' | jq .answer
```

## How the answer is produced

1. The question is embedded and matched against every document (cosine), and
   in parallel scored with BM25 so exact names like "Trump Tower Jeddah" or
   district names are never missed. The two rankings are merged with
   reciprocal rank fusion, and the top 8 passages are used.
2. If the question names one source (DarGlobal or Wasalt) retrieval is
   restricted to it; otherwise the retriever makes sure both sources are represented.
3. The system prompt instructs the model to answer only from the passages,
   cite the source URL, and admit when the data does not cover the question.
4. The response streams token by token. If the primary free model returns a
   rate limit or provider error, the next model in the chain is tried
   transparently.

## Deployment

- **Backend**: the Docker image is deployed on Render (free web service, Docker
  runtime). Set `OPENROUTER_API_KEY` and `CORS_ORIGINS=https://<vercel-app>.vercel.app`
  as environment variables. Any container host works the same way (Railway,
  Fly.io, Hugging Face Spaces, a VM with `docker compose up`).
- **Frontend**: the `web/` folder is a static site on Vercel. `web/config.js`
  holds the backend URL.

## Limitations

- Free OpenRouter models have per-minute and per-day quotas; the fallback chain
  mitigates but cannot remove that.
- Wasalt listing descriptions are mostly Arabic; the structured fields are in
  English, so English questions work well and Arabic questions are supported.
- The data is a snapshot taken on 2 September 2026. Re-run the scrapers and
  rebuild the image to refresh it.
