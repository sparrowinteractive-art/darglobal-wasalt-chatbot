"""Turn the scraped JSON into a searchable knowledge base.

Reads data/raw/*.json, builds one text document per project, listing, page
chunk or FAQ, embeds them with a local sentence-transformer and stores the
documents (docs.json) and their normalised embedding matrix (embeddings.npy)
under data/index/. A few thousand documents fit comfortably in memory, so a
numpy dot product is all the vector search we need at runtime.

Usage:
    python -m ingest.build_index
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INDEX = ROOT / "data" / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200


def load(name: str):
    path = RAW / name
    if not path.exists():
        print(f"  (missing {name})", file=sys.stderr)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    parts, start = [], 0
    while True:
        end = start + size
        if end >= len(text):
            parts.append(text[start:].strip())
            break
        cut = text.rfind("\n", start + size // 2, end)
        if cut == -1:
            cut = text.rfind(". ", start + size // 2, end)
        if cut != -1:
            end = cut + 1
        parts.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def money(v, cur="SAR"):
    if v is None:
        return "price on request"
    return f"{cur} {v:,.0f}"


# ---------------------------------------------------------------- DarGlobal

def darglobal_docs() -> list[dict]:
    docs = []
    projects = load("darglobal_projects.json")
    # catalogue overview docs: one per country plus a global one, so that
    # "which projects are in X" questions retrieve a complete list
    by_country = defaultdict(list)
    for p in projects:
        by_country[p.get("country") or "Other"].append(p)
    overview = [f"DarGlobal project catalogue: {len(projects)} projects in {len(by_country)} countries ({', '.join(by_country)})."]
    for country, items in by_country.items():
        lines = [f"DarGlobal projects in {country} ({len(items)} projects):"]
        for p in items:
            bits = [p.get("location"), p.get("project_type")]
            if p.get("bedrooms"):
                bits.append(p["bedrooms"])
            if p.get("completion_date"):
                bits.append(f"completion {p['completion_date']}")
            lines.append(f"- {p['title']}: " + "; ".join(b for b in bits if b) + f". URL: {p['url']}")
        overview.append(f"{country}: " + ", ".join(p["title"] for p in items))
        docs.append({
            "id": f"dg-country-{re.sub(r'[^a-z0-9]+', '-', country.lower())}",
            "text": "\n".join(lines),
            "meta": {"source": "darglobal", "kind": "catalogue", "title": f"DarGlobal projects in {country}", "url": "https://www.darglobal.co.uk/projects", "country": country, "city": "", "property_type": ""},
        })
    docs.append({
        "id": "dg-catalogue-all",
        "text": "\n".join(overview),
        "meta": {"source": "darglobal", "kind": "catalogue", "title": "DarGlobal project catalogue", "url": "https://www.darglobal.co.uk/projects", "country": "", "city": "", "property_type": ""},
    })
    for p in projects:
        head = [
            f"DarGlobal project: {p['title']}",
            f"Country: {p.get('country')}",
            f"Location: {p.get('location')}",
            f"Property type: {p.get('project_type')}",
        ]
        if p.get("bedrooms"):
            head.append(f"Bedrooms: {p['bedrooms']}")
        if p.get("completion_date"):
            head.append(f"Completion date: {p['completion_date']}")
        for k, v in (p.get("details") or {}).items():
            head.append(f"{k}: {v}")
        body = []
        if p.get("about"):
            body.append(f"About: {p['about']}")
        if p.get("amenities"):
            body.append("Features and amenities: " + ", ".join(p["amenities"]))
        if p.get("location_description"):
            body.append(f"Location: {p['location_description']}")
        if p.get("why_invest"):
            body.append(f"Why invest: {p['why_invest']}")
        if p.get("why_invest_points"):
            body.append("Investment highlights: " + "; ".join(p["why_invest_points"]))
        if p.get("disclaimer"):
            body.append(f"Disclaimer: {p['disclaimer']}")
        if p.get("extra_text"):
            body.append(p["extra_text"])
        header = "\n".join(head)
        meta = {
            "source": "darglobal", "kind": "project", "title": p["title"],
            "url": p["url"], "country": p.get("country") or "", "city": p.get("location") or "",
            "property_type": p.get("project_type") or "",
        }
        for i, c in enumerate(chunk("\n".join(body))):
            docs.append({"id": f"dg-proj-{p['path'].strip('/').replace('/', '-')}-{i}", "text": header + "\n" + c, "meta": meta})
        if not body:
            docs.append({"id": f"dg-proj-{p['path'].strip('/').replace('/', '-')}-0", "text": header, "meta": meta})
        for j, f in enumerate(p.get("faqs") or []):
            if f.get("question") and f.get("answer"):
                docs.append({
                    "id": f"dg-faq-{p['path'].strip('/').replace('/', '-')}-{j}",
                    "text": f"DarGlobal project: {p['title']} ({p.get('location')})\nQ: {f['question']}\nA: {f['answer']}",
                    "meta": {**meta, "kind": "faq"},
                })
    for pg in load("darglobal_pages.json"):
        meta = {"source": "darglobal", "kind": "page", "title": f"DarGlobal - {pg['title']}", "url": pg["url"], "country": "", "city": "", "property_type": ""}
        for i, c in enumerate(chunk(pg["text"])):
            docs.append({"id": f"dg-page-{pg['path'].strip('/').replace('/', '-')}-{i}", "text": f"DarGlobal website page: {pg['title']}\n{c}", "meta": meta})
    return docs


# ------------------------------------------------------------------ Wasalt

def listing_text(l: dict) -> str:
    purpose = "for sale" if l["purpose"] == "sale" else "for rent"
    parts = [f"Wasalt listing: {l.get('title') or l.get('property_type')} {purpose} in {l.get('address') or l.get('city')}, Saudi Arabia."]
    price = money(l.get("price"), l.get("currency") or "SAR")
    if l["purpose"] == "rent" and l.get("rent_frequency"):
        price += f" per {l['rent_frequency']}"
    elif l["purpose"] == "sale" and l.get("property_type") == "Land" and (l.get("price") or 0) < 50000:
        price += " (as listed; likely price per sqm)"
    parts.append(f"Price: {price}.")
    specs = []
    if l.get("bedrooms") is not None:
        specs.append(f"{l['bedrooms']} bedrooms")
    if l.get("bathrooms") is not None:
        specs.append(f"{l['bathrooms']} bathrooms")
    if l.get("living_rooms") is not None:
        specs.append(f"{l['living_rooms']} living rooms")
    if l.get("floor_size_sqm"):
        specs.append(f"{l['floor_size_sqm']:g} sqm")
    if specs:
        parts.append("Specs: " + ", ".join(specs) + ".")
    parts.append(f"Property type: {l.get('property_type')} ({l.get('property_usage')}). City: {l.get('city')}. District: {l.get('district')}. Area: {l.get('territory')}.")
    extras = []
    if l.get("furnishing"):
        extras.append(f"furnishing: {l['furnishing']}")
    if l.get("facing"):
        extras.append(f"facing {l['facing']}")
    if l.get("project_name"):
        extras.append(f"project: {l['project_name']}")
    if l.get("listed_by"):
        extras.append(f"listed by {l['listed_by']}")
    if l.get("rega_licensed"):
        extras.append("REGA licensed advertisement")
    if l.get("auction"):
        extras.append("auction property")
    if extras:
        parts.append("; ".join(extras) + ".")
    if l.get("updated_at"):
        parts.append(f"Last updated {l['updated_at'][:10]}.")
    if l.get("description_ar"):
        parts.append("Description (Arabic): " + l["description_ar"][:400])
    return " ".join(parts)


# Wasalt's English city names are transliterations; map them to common spellings
CITY_NAMES = {
    "Aldammam": "Dammam", "Alzahran": "Dhahran", "Bariduh": "Buraidah", "Alttayif": "Taif",
    "Tbwk": "Tabuk", "Hayil": "Hail", "Khobar": "Al Khobar", "Makkah Al Mukarramah": "Makkah",
    "Jubail Industrial City": "Jubail", "Al Ahsa": "Al Ahsa", "Madinah": "Madinah",
}


def wasalt_docs() -> list[dict]:
    docs = []
    listings = load("wasalt_listings.json")
    for l in listings:
        l["city"] = CITY_NAMES.get(l.get("city") or "", l.get("city"))
        if l.get("address"):
            for k, v in CITY_NAMES.items():
                l["address"] = l["address"].replace(k, v)
    for l in listings:
        docs.append({
            "id": f"ws-{l['id']}",
            "text": listing_text(l),
            "meta": {
                "source": "wasalt", "kind": "listing", "title": l.get("title") or "",
                "url": l["url"], "country": "Saudi Arabia", "city": l.get("city") or "",
                "property_type": l.get("property_type") or "", "purpose": l["purpose"],
                "price": float(l["price"]) if l.get("price") else 0.0,
            },
        })
    # aggregate market snapshot per city / purpose / type
    groups = defaultdict(list)
    for l in listings:
        groups[(l.get("city") or "Unknown", l["purpose"], l.get("property_type") or "Other")].append(l)
    by_city = defaultdict(list)
    for (city, purpose, ptype), items in groups.items():
        prices = [x["price"] for x in items if x.get("price")]
        sizes = [x["floor_size_sqm"] for x in items if x.get("floor_size_sqm")]
        line = f"- {ptype} {'for sale' if purpose == 'sale' else 'for rent'}: {len(items)} listings"
        if prices:
            line += f", price range SAR {min(prices):,.0f} to {max(prices):,.0f}, median SAR {statistics.median(prices):,.0f}"
        if sizes:
            line += f", median size {statistics.median(sizes):,.0f} sqm"
        by_city[city].append(line)
    for city, lines in by_city.items():
        n = sum(1 for l in listings if (l.get("city") or "Unknown") == city)
        text = (
            f"Wasalt market snapshot for {city}, Saudi Arabia (based on {n} listings sampled from wasalt.sa on 2026-09-02).\n"
            + "\n".join(sorted(lines))
        )
        docs.append({
            "id": f"ws-stats-{re.sub(r'[^a-z0-9]+', '-', city.lower())}",
            "text": text,
            "meta": {"source": "wasalt", "kind": "stats", "title": f"Wasalt market snapshot - {city}", "url": f"https://wasalt.sa/en/properties-for-sale-in-{re.sub(r'[^a-z0-9]+', '-', city.lower())}", "country": "Saudi Arabia", "city": city, "property_type": ""},
        })
    for g in load("wasalt_city_guides.json"):
        text = f"Wasalt guide: {g.get('title')}. Wasalt lists {g.get('total_listings')} properties {('for sale' if g['purpose']=='sale' else 'for rent')} in {g['city'].replace('-', ' ').title()}.\n{g.get('meta_description') or ''}\n{g.get('text') or ''}"
        slug = g["url"].rstrip("/").rsplit("/", 1)[-1]
        for i, c in enumerate(chunk(text)):
            docs.append({
                "id": f"ws-guide-{slug}-{i}",
                "text": c,
                "meta": {"source": "wasalt", "kind": "guide", "title": g.get("title") or "", "url": g["url"], "country": "Saudi Arabia", "city": g["city"].replace("-", " ").title(), "property_type": ""},
            })
    return docs


def main():
    docs = darglobal_docs() + wasalt_docs()
    print(f"documents: {len(docs)}")
    INDEX.mkdir(parents=True, exist_ok=True)
    (INDEX / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        [d["text"] for d in docs], normalize_embeddings=True, batch_size=32, show_progress_bar=True
    ).astype(np.float32)
    np.save(INDEX / "embeddings.npy", emb)
    print(f"embeddings: {emb.shape}")
    print(f"index written to {INDEX}")


if __name__ == "__main__":
    main()
