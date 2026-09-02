"""API contract tests against a running instance. Usage: python tests_api.py [http://localhost:8080]"""
import json, sys, time
import httpx
B = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"; fails = []
def check(name, cond, info=""):
    print(("PASS" if cond else "FAIL"), name, info)
    if not cond: fails.append(name)
r=httpx.get(B+"/health"); h=r.json()
check("health 200 + fields", r.status_code==200 and h["status"]=="ok" and h["documents"]>3000 and len(h["models"])==4, f"docs={h['documents']}")
r=httpx.get(B+"/"); check("index html served", r.status_code==200 and "Property Assistant" in r.text and "text/html" in r.headers["content-type"])
r=httpx.get(B+"/nope"); check("404 for unknown route", r.status_code==404)
t0=time.time(); r=httpx.get(B+"/api/search",params={"q":"Trump Tower Jeddah","k":3}); dt=time.time()-t0
check("search returns k results fast", r.status_code==200 and len(r.json())==3 and dt<2, f"{dt*1000:.0f}ms")
check("search top hit relevant", "Trump Tower Jeddah" in r.json()[0]["meta"]["title"])
res=httpx.get(B+"/api/search",params={"q":"apartments for rent in Jeddah","k":6}).json()
check("structured filter: rent apartments Jeddah", sum(1 for p in res if p["meta"]["kind"]=="listing" and p["meta"].get("purpose")=="rent" and p["meta"]["property_type"]=="Apartment" and p["meta"]["city"]=="Jeddah")>=3)
res=httpx.get(B+"/api/search",params={"q":"Which DarGlobal projects are in Qatar?","k":6}).json()
check("catalogue injection Qatar", any(p["id"]=="dg-country-qatar" for p in res))
check("source filter: darglobal only", all(p["meta"]["source"]=="darglobal" for p in res))
res=httpx.get(B+"/api/search",params={"q":"cheapest villa on wasalt in Riyadh","k":6}).json()
check("source filter: wasalt only", all(p["meta"]["source"]=="wasalt" for p in res))
res=httpx.get(B+"/api/search",params={"q":"Tell me about Trump Mansions in Riyadh","k":6}).json()
check("landing-page project now retrievable", any("Trump Mansions" in p["meta"]["title"] for p in res), str([p["meta"]["title"][:25] for p in res[:4]]))
res=httpx.get(B+"/api/search",params={"q":"What is the Pagani Penthouse?","k":6}).json()
check("one-of-one project retrievable with real text", any("Pagani Penthouse" in p["meta"]["title"] and "Pagani" in p["text"] and "1of1 is a bespoke" not in p["text"][:200] for p in res))
res=httpx.get(B+"/api/search",params={"q":"tell me about investing in Dubai","k":8}).json()
check("mixed query: both sources present", {p["meta"]["source"] for p in res}=={"darglobal","wasalt"})
urls=[p["meta"]["url"] for p in httpx.get(B+"/api/search",params={"q":"land in Dammam","k":8}).json()]
check("per-url cap <=2", max(urls.count(u) for u in urls)<=2)
check("arabic query no crash", len(httpx.get(B+"/api/search",params={"q":"ما هي مشاريع دار غلوبال","k":3}).json())==3)
check("empty query handled", httpx.get(B+"/api/search",params={"q":"","k":3}).status_code in (200,422))
check("chat validation: empty 422", httpx.post(B+"/api/chat",json={"message":""}).status_code==422)
check("chat validation: too long 422", httpx.post(B+"/api/chat",json={"message":"x"*2001}).status_code==422)
check("chat validation: missing 422", httpx.post(B+"/api/chat",json={}).status_code==422)
r=httpx.post(B+"/api/chat",json={"message":"Which DarGlobal projects are in Oman?","history":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]},timeout=120)
if r.status_code==200:
    d=r.json(); check("chat answers (key configured)", len(d["answer"])>40 and d["model"] and d["sources"], f"{d['model']} {d['latency_ms']}ms :: {d['answer'][:100]!r}")
else:
    check("chat without valid key -> 503 with detail", r.status_code==503 and "detail" in r.json(), f"{r.status_code} {r.text[:80]}")
with httpx.stream("POST",B+"/api/chat/stream",json={"message":"Which DarGlobal projects are in Oman?"},timeout=120) as s:
    body=s.read().decode(); ct=s.headers["content-type"]
events=[l[7:] for l in body.splitlines() if l.startswith("event: ")]
check("stream content-type", "text/event-stream" in ct)
check("stream starts with sources, ends with done", events[0]=="sources" and events[-1]=="done" and ("token" in events or "error" in events), str(sorted(set(events))))
src=json.loads([l for l in body.splitlines() if l.startswith("data: ")][0][6:])
check("stream sources include Oman catalogue", any(s["title"]=="DarGlobal projects in Oman" for s in src))
r=httpx.options(B+"/api/chat",headers={"Origin":"https://example.vercel.app","Access-Control-Request-Method":"POST","Access-Control-Request-Headers":"content-type"})
check("CORS preflight ok", r.status_code==200 and r.headers.get("access-control-allow-origin") in ("*","https://example.vercel.app"))
t0=time.time(); rs=[httpx.get(B+"/api/search",params={"q":f"villa {i}","k":5}) for i in range(10)]; dt=(time.time()-t0)/10
check("10 searches avg < 1s", all(x.status_code==200 for x in rs) and dt<1, f"avg {dt*1000:.0f}ms")
print("\nAPI:", "ALL PASS" if not fails else f"{len(fails)} FAILED {fails}")
sys.exit(1 if fails else 0)
