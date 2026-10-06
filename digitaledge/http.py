"""Tiny cached, retrying JSON client. Responses are cached on disk (keyed by URL+params)
so research runs are reproducible and re-runs are instant."""
import hashlib
import json
import os
import pathlib
import threading
import time
import requests

CACHE_DIR = pathlib.Path(os.environ.get("DIGITALEDGE_CACHE", "data/cache"))
_session = requests.Session()
_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = float(os.environ.get("DIGITALEDGE_MIN_INTERVAL", "0.15"))  # global request spacing (s)


def _throttle():
    with _lock:
        wait = _last[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def get_json(url, params=None, cache=True, retries=12, timeout=30):
    key = hashlib.sha1(json.dumps([url, sorted((params or {}).items())]).encode()).hexdigest()
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    for attempt in range(retries):
        try:
            _throttle()
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 0)))
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            data = r.json()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            return data
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 60))


def utc(x):
    """Timestamp in UTC whether `x` is naive, a string, or already tz-aware."""
    import pandas as pd
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
