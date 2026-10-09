"""Cached, throttled Manifold Markets API client (public, no key)."""
from __future__ import annotations
import json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

API = "https://api.manifold.markets/v0"
_lock = threading.Lock()
_last = [0.0]


def _get(path, params, min_gap=0.15, tries=6):
    for k in range(tries):
        with _lock:
            wait = _last[0] + min_gap - time.time()
            if wait > 0: time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 200: return r.json()
        if r.status_code == 429 or r.status_code >= 500: time.sleep(2 ** k); continue
        return None
    return None


def list_resolved_binary(out: Path, pages: int = 80, min_bettors: int = 20, max_bettors: int = 400):
    """Page /markets backwards in time; keep resolved YES/NO binary markets with enough (not too many) bettors."""
    if out.exists(): return json.loads(out.read_text())
    keep, before = [], None
    for p in range(pages):
        batch = _get("markets", {"limit": 1000, **({"before": before} if before else {})})
        if not batch: break
        before = batch[-1]["id"]
        for m in batch:
            if (m.get("outcomeType") == "BINARY" and m.get("isResolved") and m.get("resolution") in ("YES", "NO")
                    and min_bettors <= m.get("uniqueBettorCount", 0) <= max_bettors):
                keep.append({k: m.get(k) for k in ("id", "createdTime", "closeTime", "resolutionTime", "resolution",
                                                    "uniqueBettorCount", "volume", "question")})
        print(f"page {p}: scanned to {batch[-1]['createdTime']}, kept {len(keep)}", flush=True)
    out.write_text(json.dumps(keep)); return keep


def fetch_bets(cid: str, d: Path):
    f = d / f"{cid}.json"
    if f.exists(): return
    bets, before = [], None
    while True:
        b = _get("bets", {"contractId": cid, "limit": 1000, **({"before": before} if before else {})})
        if b is None: return
        bets += [{k: x.get(k) for k in ("userId", "outcome", "amount", "probBefore", "probAfter", "createdTime",
                                         "isCancelled", "limitProb", "orderAmount", "isRedemption")} for x in b]
        if len(b) < 1000: break
        before = b[-1]["id"]
    f.write_text(json.dumps(bets))


def fetch_all(markets, d: Path, workers: int = 6):
    d.mkdir(parents=True, exist_ok=True)
    todo = [m["id"] for m in markets if not (d / f"{m['id']}.json").exists()]
    done = [0]
    def job(cid):
        fetch_bets(cid, d); done[0] += 1
        if done[0] % 100 == 0: print(f"bets fetched {done[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(job, todo))
