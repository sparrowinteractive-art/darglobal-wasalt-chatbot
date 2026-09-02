"""Scrape public project and company information from darglobal.co.uk.

DarGlobal runs a Next.js front end on top of a public Strapi CMS.

* The project catalogue (36 projects across 7 countries) is read straight
  from the public Strapi endpoint ``/api/projects``.
* The web site itself is behind Imperva/Incapsula bot protection, which
  blocks plain HTTP clients and headless browsers. Project detail pages and
  company pages are therefore loaded with a *headed* Chrome via Playwright,
  and the server-rendered ``__NEXT_DATA__`` JSON is parsed from each page.

Usage:
    python -m scraper.darglobal
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
from playwright.async_api import async_playwright

from .common import UA, clean_html, extract_next_data, rich_text, save_json

SITE = "https://www.darglobal.co.uk"
STRAPI = "https://strapi.darglobal.co.uk/api"

CONTENT_PAGES = [
    "/about", "/faq", "/why-invest", "/investor", "/hospitality", "/commercial",
    "/luxury-golf-communities", "/one-of-one", "/exclusive-membership",
    "/tokenization", "/press", "/blog", "/insights", "/get-in-touch",
    "/partners/aston-martin", "/partners/elie-saab", "/partners/fendi-casa",
    "/partners/lamborghini", "/partners/marriott-residences-aida-oman",
    "/partners/missoni", "/partners/mouawad", "/partners/pagani",
    "/partners/the-trump-organization", "/partners/w-hotels",
]

# JSON keys that carry no useful prose
NOISE_KEYS = {
    "id", "createdAt", "updatedAt", "publishedAt", "locale", "vuid",
    "versionNumber", "versionComment", "hash", "mime", "ext", "url", "width",
    "height", "size", "provider", "formats", "previewUrl", "provider_metadata",
    "localizations", "ButtonTheme", "target", "href", "icon", "iconClass",
    "image", "Image", "alt", "alternativeText", "caption", "name_ar",
    "globalData", "isMobile", "otherProjectData", "thankyouPageData",
    "SEO", "metaSocial", "keywords", "canonicalURL", "structuredData",
    "HeroBanner", "HeroBannerMobile", "Background", "video", "Video",
    "PhotoGallery", "galleryList", "file", "path", "isVisibleInListView",
    "projectCode", "slug", "__component", "ProjectCardImage", "cardImage",
    "OtherProject", "reqHeaders", "abTestVariant",
}


def flatten_text(obj, path: str = "", out: list | None = None) -> list[tuple[str, str]]:
    """Collect (path, text) pairs for every human-readable string in a JSON tree."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in NOISE_KEYS:
                continue
            flatten_text(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten_text(v, f"{path}[{i}]", out)
    elif isinstance(obj, str):
        s = clean_html(obj)
        if len(s) > 2 and not s.startswith(("http", "/", "#")):
            out.append((path, s))
    return out


def text_block(obj) -> str:
    """Render a JSON subtree as 'label: text' lines for the knowledge base."""
    lines = []
    for path, text in flatten_text(obj):
        label = path.split(".")[-1].split("[")[0]
        if label and label not in ("value", "text", "name"):
            lines.append(f"{label}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def fetch_catalogue() -> list[dict]:
    r = httpx.get(f"{STRAPI}/projects", params={"populate": "deep"}, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    attrs = r.json()["data"]["attributes"]
    projects = []
    for tab in attrs.get("Tabs", []):
        country = tab.get("TabHeading")
        for card in tab.get("TabData", []):
            projects.append({
                "title": clean_html(card.get("ProjectTitle")),
                "country": country,
                "location": clean_html(card.get("Location")),
                "project_type": card.get("ProjectType"),
                "bedrooms": clean_html(card.get("Bedrooms")) or None,
                "completion_date": card.get("completionDate"),
                "card_description": clean_html(card.get("ProjectDescription")) or None,
                "path": card.get("Link"),
                "url": SITE + card.get("Link", ""),
            })
    return projects


def _names(lst, key="name"):
    return [clean_html(x.get(key)) for x in (lst or []) if isinstance(x, dict) and x.get(key)]


def parse_project(pp: dict, base: dict) -> dict:
    pd = pp.get("projectDetailsData") or pp.get("oneOfOneData") or {}
    if isinstance(pd, list):
        pd = pd[0] if pd else {}
    a = pd.get("attributes", pd) if isinstance(pd, dict) else {}
    details = a.get("ProjectDetails") or {}
    about = details.get("AboutProject") or {}
    amen = details.get("Amenities") or {}
    why = details.get("whyInvest") or {}
    loc = details.get("loaction") or details.get("location") or {}
    locimg = loc.get("locationDetailsWithImages") or {}
    faqs = []
    for block in a.get("faqContent") or []:
        for f in block.get("faqs") or []:
            faqs.append({"question": rich_text(f.get("question")), "answer": rich_text(f.get("answer"))})
    skip = ("AboutProject", "Amenities", "whyInvest", "loaction", "ProjectBanner")
    rec = dict(base)
    rec.update({
        "title": clean_html(a.get("title")) or base["title"],
        "banner_location": clean_html((details.get("ProjectBanner") or {}).get("Location")),
        "about_title": clean_html(about.get("title")),
        "about": rich_text(about.get("description")),
        "details": {clean_html(d.get("name")): clean_html(d.get("value")) for d in about.get("AboutDetailsType") or [] if d.get("name")},
        "amenities": _names(amen.get("amenitiesList")),
        "why_invest": rich_text(why.get("description")),
        "why_invest_points": _names(why.get("investlist")),
        "location_description": rich_text(loc.get("description")),
        "location_name": clean_html(locimg.get("location")),
        "lat": locimg.get("lat"),
        "lon": locimg.get("long"),
        "disclaimer": rich_text(a.get("disclaimer")),
        "faqs": faqs,
        "extra_text": text_block({k: v for k, v in details.items() if k not in skip}),
    })
    if not rec["about"] and pp.get("oneOfOneData"):
        rec["extra_text"] = text_block(pp.get("oneOfOneData")) + "\n" + text_block(pp.get("collectionsData"))
    return rec


async def load_page(page, url: str) -> dict | None:
    await page.goto(url, wait_until="domcontentloaded", timeout=90000)
    for _ in range(15):  # Incapsula challenge resolves itself after a few seconds
        await page.wait_for_timeout(2000)
        html = await page.content()
        data = extract_next_data(html)
        if data:
            return data
    return None


async def scrape(delay: float) -> tuple[list, list]:
    catalogue = fetch_catalogue()
    print(f"catalogue: {len(catalogue)} projects", flush=True)
    projects, pages = [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False, channel="chrome", args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = await browser.new_context(locale="en-GB", viewport={"width": 1366, "height": 800})
        page = await ctx.new_page()
        for base in catalogue:
            t0 = time.time()
            try:
                data = await load_page(page, base["url"])
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {base['url']} failed: {exc}", file=sys.stderr, flush=True)
                data = None
            if not data:
                print(f"  ! {base['url']}: no page data, keeping catalogue card only", file=sys.stderr, flush=True)
                projects.append(base)
                continue
            rec = parse_project(data["props"].get("pageProps", {}), base)
            projects.append(rec)
            print(
                f"  project {rec['title'][:45]:45} about={len(rec.get('about', ''))} "
                f"amen={len(rec.get('amenities', []))} faqs={len(rec.get('faqs', []))} ({time.time() - t0:.1f}s)",
                flush=True,
            )
            await asyncio.sleep(delay)
        for path in CONTENT_PAGES:
            url = SITE + path
            t0 = time.time()
            try:
                data = await load_page(page, url)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {url} failed: {exc}", file=sys.stderr, flush=True)
                continue
            if not data:
                print(f"  ! {url}: no page data", file=sys.stderr, flush=True)
                continue
            pp = data["props"].get("pageProps", {})
            text = text_block(pp)
            title = path.strip("/").replace("-", " ").replace("/", " / ").title()
            pages.append({"url": url, "path": path, "title": title, "text": text})
            print(f"  page {path:45} chars={len(text)} ({time.time() - t0:.1f}s)", flush=True)
            await asyncio.sleep(delay)
        await browser.close()
    return projects, pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    projects, pages = asyncio.run(scrape(args.delay))
    p1 = save_json("darglobal_projects.json", projects)
    p2 = save_json("darglobal_pages.json", pages)
    print(f"saved {len(projects)} projects -> {p1}")
    print(f"saved {len(pages)} pages -> {p2}")


if __name__ == "__main__":
    main()
