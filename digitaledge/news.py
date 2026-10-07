"""News-attention features from GDELT DOC 2.0 (daily share of global news volume mentioning a term).
Polite and cached: GDELT allows ~1 request / 5 s and only the trailing ~3 months."""
import hashlib
import json
import pathlib
import re
import time
import numpy as np
import pandas as pd
import requests

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE = pathlib.Path("data/cache/gdelt")
_last = [0.0]


def base_term(label):
    """'trump (3+ times)' -> 'trump'; 'oil / gas / gasoline' stays a multi-alternative label."""
    return re.sub(r"\s*\(.*?\)\s*", " ", label).strip()


def gdelt_query(label):
    alts = [a.strip() for a in re.split(r"\s*/\s*", base_term(label)) if a.strip()]
    q = " OR ".join(f'"{a}"' if " " in a else a for a in alts)
    return f"({q})" if len(alts) > 1 else q


def daily_volume(label, start="20260707000000", end="20261006235959", spacing=12.0, retries=6):
    q = gdelt_query(label)
    key = hashlib.sha1(f"{q}|{start}|{end}".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return pd.Series(json.loads(path.read_text()))
    for attempt in range(retries):
        wait = _last[0] + spacing - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            r = requests.get(URL, params={"query": q, "mode": "timelinevol", "startdatetime": start, "enddatetime": end,
                                          "format": "json", "timelinesmooth": 0}, timeout=60)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                tl = r.json().get("timeline", [])
                data = {p["date"][:8]: float(p["value"]) for p in (tl[0]["data"] if tl else [])}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return pd.Series(data)
            if r.status_code in (400, 404):
                return pd.Series(dtype=float)          # bad query (e.g. too short / stopword)
        except requests.RequestException:
            pass
        time.sleep(min(15 * (attempt + 1), 90))
    return pd.Series(dtype=float)


def salience(series, when, lookback=3, baseline=30):
    """(log ratio of recent mean vs prior baseline mean, log recent level) strictly before `when` (a date)."""
    if series is None or len(series) == 0:
        return np.nan, np.nan
    s = series.copy()
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    d = pd.Timestamp(when).tz_localize(None).normalize()
    rec = s[(s.index >= d - pd.Timedelta(days=lookback)) & (s.index < d)]
    base = s[(s.index >= d - pd.Timedelta(days=lookback + baseline)) & (s.index < d - pd.Timedelta(days=lookback))]
    if len(rec) < 2 or len(base) < 7 or base.mean() <= 0:
        return np.nan, np.nan
    return float(np.log((rec.mean() + 1e-4) / (base.mean() + 1e-4))), float(np.log(rec.mean() + 1e-4))
