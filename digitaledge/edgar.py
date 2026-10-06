"""SEC EDGAR access (public, requires a descriptive User-Agent, <= 10 req/s): ticker -> CIK,
filing lists, and filing text for word-count features. Everything is cached on disk."""
import hashlib
import html
import json
import os
import pathlib
import re
import threading
import time
import requests

UA = os.environ.get("EDGAR_UA", "kalshi-edge-research research@example.com")
CACHE = pathlib.Path(os.environ.get("DIGITALEDGE_CACHE", "data/cache")) / "edgar"
_lock, _last = threading.Lock(), [0.0]


def _get(url, as_json=False, retries=5):
    key = hashlib.sha1(url.encode()).hexdigest()
    path = CACHE / key[:2] / (key + (".json" if as_json else ".txt"))
    if path.exists():
        return json.loads(path.read_text()) if as_json else path.read_text()
    for attempt in range(retries):
        with _lock:
            wait = _last[0] + 0.15 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            if 400 <= r.status_code < 500:
                return None
            data = r.json() if as_json else r.text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data) if as_json else data)
            return data
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def ticker_to_cik():
    d = _get("https://www.sec.gov/files/company_tickers.json", as_json=True) or {}
    return {v["ticker"].upper(): int(v["cic"] if "cic" in v else v["cik_str"]) for v in d.values()}


def filings(cik):
    """Recent filings as a list of dicts (form, filingDate, accessionNumber, primaryDocument)."""
    d = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", as_json=True)
    if not d:
        return []
    r = d["filings"]["recent"]
    return [dict(zip(r.keys(), vals)) for vals in zip(*r.values())]


def last_periodic_before(cik, date, forms=("10-Q", "10-K")):
    """Most recent 10-Q/10-K filed at least 1 day before `date` (causal)."""
    date = str(date)[:10]
    cand = [f for f in filings(cik) if f["form"] in forms and f["filingDate"] < date]
    return max(cand, key=lambda f: f["filingDate"]) if cand else None


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def filing_text(cik, f):
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accessionNumber'].replace('-', '')}/{f['primaryDocument']}"
    raw = _get(url)
    if not raw:
        return ""
    t = _TAG.sub(" ", re.sub(r"(?is)<(script|style).*?</\1>", " ", raw))
    return _WS.sub(" ", html.unescape(t)).lower()


def phrase_count(text, label):
    """Occurrences of any alternative in a market label like 'ai / artificial intelligence'."""
    alts = [a.strip() for a in re.split(r"\s*/\s*", label.lower()) if a.strip()]
    n = 0
    for a in alts:
        n += len(re.findall(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", text))
    return n
