"""Ask the running API a set of representative questions and print the answers.

Usage: python tests_smoke.py [http://localhost:8000]
"""
import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
QUESTIONS = [
    "What DarGlobal projects are in Saudi Arabia?",
    "Tell me about Trump Tower Jeddah. When will it be completed?",
    "Which DarGlobal projects are in Oman?",
    "What amenities does the Astera in Ras Al Khaimah offer?",
    "Why does DarGlobal say I should invest in Dubai?",
    "Who are DarGlobal's brand partners?",
    "What is the median price of apartments for sale in Riyadh on Wasalt?",
    "Show me some villas for rent in Jeddah.",
    "Are there any land plots for sale in Dammam? What sizes and prices?",
    "How many listings does Wasalt have in Makkah?",
    "What is the weather like in Riyadh today?",
    "ما هي مشاريع دار غلوبال في جدة؟",
]

for q in QUESTIONS:
    t0 = time.time()
    try:
        r = httpx.post(f"{BASE}/api/chat", json={"message": q}, timeout=120)
        r.raise_for_status()
        d = r.json()
        print(f"\n### {q}\n[{d['model']} | {int((time.time()-t0)*1000)} ms]\n{d['answer']}\nsources: {[s['url'] for s in d['sources'][:4]]}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n### {q}\nERROR {exc}")
