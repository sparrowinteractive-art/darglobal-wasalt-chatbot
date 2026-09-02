"""Scrape public property listings from wasalt.sa.

Wasalt is a Next.js app protected by Cloudflare. Plain HTTP requests get a
JavaScript challenge, so we drive headless Chromium with Playwright and read
the server-rendered ``__NEXT_DATA__`` JSON that every page embeds. Search
result pages (SRP) carry 32 listings each with full structured fields, which
is far cheaper than opening every listing.

robots.txt allows everything except /search, which we never touch. We use the
public city category pages that Wasalt publishes in its sitemap.

Usage:
    python -m scraper.wasalt --pages 6 --headed
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time

import httpx
from playwright.async_api import async_playwright

from .common import UA, clean_html, extract_next_data, load_json, save_json

BASE = "https://wasalt.sa"

CITIES = [
    "riyadh", "jeddah", "makkah", "madinah", "dammam", "al-khobar", "dhahran",
    "taif", "buraidah", "abha", "tabuk", "jubail", "al-ahsa", "yanbu", "hail",
]
# Riyadh districts are filed under its five regions in the sitemap
REGION_PREFIXES = ["north-riyadh", "south-riyadh", "east-riyadh", "west-riyadh", "central-riyadh"]
PURPOSES = ["sale", "rent"]
CATEGORY_SITEMAP = "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz"


def category_urls(districts_per_city: int, cities: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Return (city, purpose, url) for the city page plus N district pages each.

    Wasalt paginates its search results client-side, so the server-rendered
    page only ever contains the first 32 listings. District-level category
    pages from the public sitemap give us distinct sets of listings instead.
    """
    xml = httpx.get(CATEGORY_SITEMAP, timeout=60, headers={"User-Agent": UA}).text
    locs = re.findall(r"<loc>(.*?)</loc>", xml)
    out = []
    for city in cities or CITIES + REGION_PREFIXES:
        for purpose in PURPOSES:
            prefix = f"{BASE}/en/properties-for-{purpose}-in-{city}"
            if city not in REGION_PREFIXES:
                out.append((city, purpose, prefix))
            districts = [u for u in locs if u.startswith(prefix + "-")]
            # spread across the district list rather than taking the first N
            step = max(1, len(districts) // districts_per_city) if districts else 1
            for u in districts[::step][:districts_per_city]:
                out.append((city, purpose, u))
    return out

BLOCKED_RESOURCES = {"image", "media", "font", "stylesheet"}


def _attr_map(item: dict) -> dict:
    out = {}
    for a in item.get("attributes") or []:
        key = a.get("key")
        if key:
            out[key] = a.get("value")
    return out


def normalise(item: dict) -> dict | None:
    info = item.get("propertyInfo") or {}
    slug = info.get("slug")
    purpose = info.get("propertyFor") or "sale"
    if not slug:
        return None
    attrs = _attr_map(item)
    owner = item.get("propertyOwner") or {}
    loc = item.get("location") or {}
    price = info.get("salePrice") if purpose == "sale" else info.get("expectedRent")
    try:
        price = float(price) if price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    return {
        "id": item.get("id"),
        "url": f"{BASE}/en/property/{purpose}/{slug}",
        "title": info.get("title"),
        "purpose": purpose,
        "property_type": info.get("propertySubType"),
        "property_usage": info.get("propertyMainType"),
        "city": info.get("city"),
        "district": info.get("district") or info.get("zone"),
        "territory": info.get("territory"),
        "state": info.get("state"),
        "address": info.get("address"),
        "country": info.get("country") or "Saudi Arabia",
        "price": price,
        "currency": info.get("currencyType") or "SAR",
        "rent_frequency": info.get("rentFreq") or None,
        "floor_size_sqm": _to_float(item.get("floorSize")),
        "bedrooms": attrs.get("noOfBedrooms"),
        "bathrooms": attrs.get("noOfBathrooms"),
        "living_rooms": attrs.get("noOfLivingRooms") or attrs.get("noOfLivingrooms"),
        "furnishing": info.get("furnishingType"),
        "facing": info.get("facingType"),
        "project_name": info.get("projectName"),
        "description_ar": clean_html(item.get("regaMojDesc")),
        "listed_by": owner.get("enUserRole"),
        "listed_by_capacity": owner.get("userCapacity"),
        "rega_licensed": bool(item.get("isRegaProp")),
        "verified": bool(item.get("isVerified")),
        "auction": bool(item.get("isAuction")),
        "published_at": item.get("publishedAt"),
        "updated_at": item.get("updatedAt"),
        "lat": _to_float(loc.get("lat")),
        "lon": _to_float(loc.get("lon")),
    }


def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


async def scrape(pages_per_city: int, delay: float, headless: bool = True, cities: list[str] | None = None) -> tuple[dict, list]:
    listings: dict[int, dict] = {}
    city_guides: list[dict] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless, channel="chrome", args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = await browser.new_context(locale="en-US", viewport={"width": 1366, "height": 800})
        if headless:
            await ctx.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in BLOCKED_RESOURCES
                else route.continue_(),
            )
        page = await ctx.new_page()
        targets = category_urls(pages_per_city, cities)
        print(f"{len(targets)} category pages to visit", flush=True)
        blocked_streak = 0
        for city, purpose, url in targets:
            t0 = time.time()
            data = None
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                for _ in range(8):  # Cloudflare interstitial resolves itself after a few seconds
                    data = extract_next_data(await page.content())
                    if data:
                        break
                    await page.wait_for_timeout(2500)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {url} failed: {exc}", file=sys.stderr, flush=True)
            if data is None:
                blocked_streak += 1
                print(f"  ! {url}: no __NEXT_DATA__ (streak {blocked_streak})", file=sys.stderr, flush=True)
                if blocked_streak >= 3:
                    print("  cooling down for 60s and opening a fresh context", flush=True)
                    await asyncio.sleep(60)
                    await page.close()
                    page = await ctx.new_page()
                    blocked_streak = 0
                continue
            blocked_streak = 0
            pp = data["props"]["pageProps"]
            sr = pp.get("searchResult") or {}
            props = sr.get("properties") or []
            new = 0
            for item in props:
                # compounds/projects come as a list of unit dicts
                units = item if isinstance(item, list) else [item]
                for unit in units:
                    if not isinstance(unit, dict):
                        continue
                    rec = normalise(unit)
                    if rec and rec["id"] not in listings:
                        listings[rec["id"]] = rec
                        new += 1
            meta = pp.get("metaData") or {}
            guide_text = clean_html(pp.get("para"))
            if guide_text:
                city_guides.append({
                    "url": url,
                    "city": city,
                    "purpose": purpose,
                    "title": meta.get("h1tag") or meta.get("title_tag"),
                    "meta_description": meta.get("meta_description"),
                    "total_listings": sr.get("count"),
                    "text": guide_text,
                })
            print(f"  {purpose:4} {url.rsplit('-in-', 1)[1]:40} {len(props):3} items, {new:3} new, total {len(listings):5} ({time.time() - t0:.1f}s)", flush=True)
            await asyncio.sleep(delay)
        await browser.close()
    return listings, city_guides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=6, help="district pages per city and purpose (32 listings each)")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--cities", help="comma separated subset of city/region slugs to scrape")
    ap.add_argument("--merge", action="store_true", help="merge into existing wasalt_listings.json instead of replacing it")
    args = ap.parse_args()
    cities = [c.strip() for c in args.cities.split(",")] if args.cities else None
    listings, guides = asyncio.run(scrape(args.pages, args.delay, headless=not args.headed, cities=cities))
    if args.merge:
        for old in load_json("wasalt_listings.json", []):
            listings.setdefault(old["id"], old)
        seen = {g["url"] for g in guides}
        guides += [g for g in load_json("wasalt_city_guides.json", []) if g["url"] not in seen]
    out = sorted(listings.values(), key=lambda r: r["id"])
    p1 = save_json("wasalt_listings.json", out)
    p2 = save_json("wasalt_city_guides.json", guides)
    print(f"saved {len(out)} listings -> {p1}\nsaved {len(guides)} city guides -> {p2}")


if __name__ == "__main__":
    main()
