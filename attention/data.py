"""Wikimedia pageviews (public REST API, no key). Cached per article."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import requests

H = {"User-Agent": "research-project/0.1 (educational; contact via GitHub)"}
BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews"


def _get(url, tries=8):
    for k in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** min(k, 5)); continue
        if r.status_code == 200: return r.json()
        if r.status_code == 429:
            time.sleep(min(int(r.headers.get("retry-after", 30)), 120) + 1); continue
        if r.status_code in (500, 502, 503): time.sleep(2 ** min(k, 5)); continue
        return None
    return None


def top_articles(year=2025, month=6, n=700):
    d = _get(f"{BASE}/top/en.wikipedia/all-access/{year}/{month:02d}/all-days")
    arts = [x["article"] for x in d["items"][0]["articles"]
            if ":" not in x["article"] and x["article"] not in ("Main_Page", "-")]
    return arts[:n]


def fetch_series(article, start="20230101", end="20260930", d=Path("data/wiki")):
    f = d / (article.replace("/", "_") + ".json")
    if f.exists(): return json.loads(f.read_text())
    j = _get(f"{BASE}/per-article/en.wikipedia/all-access/user/{requests.utils.quote(article, safe='')}/daily/{start}/{end}")
    if j is None: return None
    days = {x["timestamp"][:8]: x["views"] for x in j["items"]}
    out = {"article": article, "views": days}
    f.write_text(json.dumps(out)); time.sleep(0.3)
    return out


def matrix(articles, start="2023-01-01", end="2026-09-30", d=Path("data/wiki"), min_days=0.97):
    """(n_series, T) float array, NaN-free: missing days are zero views. Series with too many gaps dropped."""
    import pandas as pd
    idx = pd.date_range(start, end)
    keys = [i.strftime("%Y%m%d") for i in idx]
    rows, names = [], []
    for a in articles:
        s = fetch_series(a, d=d)
        if not s: continue
        v = np.array([s["views"].get(k, np.nan) for k in keys], float)
        if np.isnan(v).mean() > 1 - min_days: continue
        rows.append(np.nan_to_num(v, nan=0.0)); names.append(a)
    return np.stack(rows), names, idx
