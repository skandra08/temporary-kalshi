"""Breadth screen: which recurring Kalshi series are priced least efficiently?

Stage 1 (this module's `list_series_markets`): settled markets per series.
Stage 2 (`candles_at_leads`): quotes at fixed lead times before close.
Stage 3 (`evaluate`): fee-aware P&L of simple rules, day-clustered inference, FDR-corrected."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .http import get_json
from .sources.kalshi import BASE

FREQS = ("daily", "weekly", "hourly")
KEEP = ["series", "ticker", "event_ticker", "open_time", "close_time", "result", "volume", "strike_type",
        "floor_strike", "cap_strike", "last_price_dollars", "yes_sub_title"]


def candidate_series(catalog_csv="data/series_catalog.csv", exclude_categories=("Sports",)):
    s = pd.read_csv(catalog_csv)
    s = s[s["frequency"].isin(FREQS) & ~s["category"].isin(exclude_categories)]
    return s[["ticker", "title", "category", "frequency", "fee_type", "fee_multiplier"]].reset_index(drop=True)


def list_series_markets(series, max_pages=2):
    """Most recent settled markets for a series (up to max_pages x 1000), live then historical."""
    rows = []
    for path in ("markets", "historical/markets"):
        cur = None
        for _ in range(max_pages):
            p = {"series_ticker": series, "limit": 1000}
            if path == "markets":
                p["status"] = "settled"
            if cur:
                p["cursor"] = cur
            try:
                d = get_json(f"{BASE}/{path}", p)
            except Exception:
                break
            rows += [dict(m, series=series) for m in d.get("markets", [])]
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
        if len(rows) >= 300:      # enough history; skip the historical endpoint
            break
    if not rows:
        return pd.DataFrame(columns=KEEP)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"])].copy()
    df["volume"] = pd.to_numeric(df.get("volume_fp", df.get("volume")), errors="coerce")
    for c in KEEP:
        if c not in df:
            df[c] = np.nan
    return df[KEEP]


def list_all(series_list, workers=3):
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(list_series_markets, series_list))
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def rank_series(markets, catalog, min_volume=20):
    m = markets.copy()
    m["liquid"] = m["volume"] >= min_volume
    g = m.groupby("series").agg(n=("ticker", "size"), n_liquid=("liquid", "sum"), total_volume=("volume", "sum"),
                                median_volume=("volume", "median"), yes_rate=("result", lambda r: (r == "yes").mean()),
                                first=("close_time", "min"), last=("close_time", "max"))
    return g.join(catalog.set_index("ticker")[["title", "category", "frequency", "fee_multiplier"]]).sort_values(
        "n_liquid", ascending=False)


# ---------------------------------------------------------------- stage 2: quotes at lifetime fractions
FRACS = (0.5, 0.8, 0.95)


def _candle_job(args):
    ticker, series, start, end, period = args
    paths = [f"{BASE}/series/{series}/markets/{ticker}/candlesticks",
             f"{BASE}/historical/markets/{ticker}/candlesticks"]
    for url in paths:
        try:
            d = get_json(url, {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
                               "period_interval": period})
        except Exception:
            continue
        rows = []
        for c in d.get("candlesticks", []):
            try:
                rows.append((ticker, c["end_period_ts"], _px(c.get("yes_bid")), _px(c.get("yes_ask")),
                             _num(c.get("volume_fp", c.get("volume")))))
            except (KeyError, TypeError, ValueError):
                continue
        if rows:
            return rows
    return []


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _px(block):
    if not isinstance(block, dict):
        return np.nan
    if block.get("close_dollars") not in (None, ""):
        return _num(block["close_dollars"])
    c = _num(block.get("close"))
    return c / 100 if c == c and c > 1 else c


def sample_markets(markets, per_series=120, min_volume=20):
    m = markets.copy()
    for c in ("open_time", "close_time"):
        m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
    m = m[(m["volume"] >= min_volume) & m["open_time"].notna()]
    m["life_h"] = (m["close_time"] - m["open_time"]).dt.total_seconds() / 3600
    m = m[(m["life_h"] > 0.2) & (m["life_h"] < 24 * 14)]
    return m.sort_values("close_time", ascending=False).groupby("series").head(per_series)


def fetch_candles(sample, out_csv, workers=3, chunk=400):
    """Hourly (or 1-min for short-lived markets) candles over each market's life, flushed in chunks."""
    import os
    done = set(pd.read_csv(out_csv, usecols=["ticker"])["ticker"]) if os.path.exists(out_csv) else set()
    todo = sample[~sample["ticker"].isin(done)]
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        jobs = [(r.ticker, r.series, r.open_time, r.close_time, 1 if r.life_h <= 6 else 60)
                for r in part.itertuples()]
        with ThreadPoolExecutor(workers) as ex:
            rows = [x for res in ex.map(_candle_job, jobs) for x in res]
        # tickers with no candles are recorded with NaNs so a rerun does not refetch them
        got = {r[0] for r in rows}
        rows += [(t, 0, np.nan, np.nan, np.nan) for t in part["ticker"] if t not in got]
        pd.DataFrame(rows, columns=["ticker", "ts", "yes_bid", "yes_ask", "volume"]).to_csv(
            out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"{min(i + chunk, len(todo))}/{len(todo)}", flush=True)


def quotes_at_fractions(sample, candles, fracs=FRACS, max_stale_frac=0.15):
    """Last quote at or before open + f*life for each market and fraction (no look-ahead)."""
    c = candles.dropna(subset=["yes_bid", "yes_ask"]).copy()
    c["t"] = pd.to_datetime(c["ts"], unit="s", utc=True)
    c = c.sort_values("t")
    out = []
    s = sample.set_index("ticker")
    for f in fracs:
        tgt = (s["open_time"] + (s["close_time"] - s["open_time"]) * f).rename("target")
        q = c.merge(tgt, left_on="ticker", right_index=True)
        q = q[q["t"] <= q["target"]].groupby("ticker").tail(1)
        q["frac"] = f
        life = (s["close_time"] - s["open_time"]).reindex(q["ticker"]).to_numpy()
        q = q[(q["target"] - q["t"]).to_numpy() <= life * max_stale_frac]
        out.append(q)
    q = pd.concat(out).merge(sample, on="ticker")
    return q[(q["yes_ask"] > 0) & (q["yes_bid"] < 1)].reset_index(drop=True)


# ---------------------------------------------------------------- stage 3: rules, FDR, replication
RULES = {
    "buy_no_all": ("no", lambda mid: np.ones_like(mid, dtype=bool)),
    "buy_yes_all": ("yes", lambda mid: np.ones_like(mid, dtype=bool)),
    "longshot_no_lt10": ("no", lambda mid: mid < 0.10),
    "favorite_yes_gt90": ("yes", lambda mid: mid > 0.90),
}


def pnl_rule(q, side, fee_mult=1.0):
    """P&L of one contract bought at the ask on `side`, net of the (amortised) taker fee."""
    from .digital import fee_per_contract_amortised
    yes = (q["result"] == "yes").to_numpy().astype(float)
    if side == "yes":
        ask = q["yes_ask"].to_numpy()
        return yes - ask - fee_mult * fee_per_contract_amortised(ask)
    ask = 1 - q["yes_bid"].to_numpy()
    return (1 - yes) - ask - fee_mult * fee_per_contract_amortised(ask)


def boot_mean(x, clusters, n_boot=1500, seed=0):
    """Cluster bootstrap: (mean, ci_lo, ci_hi, one-sided p for H0: mean <= 0)."""
    x = pd.Series(np.asarray(x, float))
    g = pd.Series(np.asarray(clusters))
    agg = x.groupby(g).agg(["sum", "count"])
    s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), float((b <= 0).mean())


def exact_binom_p(pnl, cost_plus_fee, clusters):
    """One-sided exact binomial p-value for H0: win probability <= break-even (price + fee).

    Works when the sample has zero variance (all wins), where a bootstrap would wrongly report
    certainty. Contracts in one event are dependent, so the effective sample size is the number
    of independent events: each event contributes its win fraction."""
    from scipy.stats import binom
    win = (np.asarray(pnl) + np.asarray(cost_plus_fee)) > 0.5          # payoff 1 happened
    ev = pd.Series(win.astype(float)).groupby(np.asarray(clusters)).mean()
    n = len(ev)
    wins = int(round(ev.sum()))
    be = float(np.clip(np.mean(cost_plus_fee), 1e-6, 1 - 1e-6))
    return float(binom.sf(wins - 1, n, be))


def bh_fdr(p, alpha=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries at FDR level alpha."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, len(p) + 1) / len(p))
    ok = p[order] <= thresh
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    mask = np.zeros(len(p), bool)
    mask[order[:k]] = True
    return mask


def evaluate(q, fee_mult_by_series=None, min_n=40, n_boot=1500, min_events=25, min_events_half=10):
    """Test every (series, entry fraction, rule); replicate on the later half of each series' history."""
    rows = []
    fm = fee_mult_by_series or {}
    for (series, frac), g in q.groupby(["series", "frac"]):
        # strikes of one event share one outcome path, so the independent unit is the event
        key = g["event_ticker"] if "event_ticker" in g else pd.to_datetime(g["close_time"], utc=True).dt.floor("D")
        g = g.assign(day=key.to_numpy())
        mid = ((g["yes_bid"] + g["yes_ask"]) / 2).to_numpy()
        cut = g["close_time"].median()
        early = (g["close_time"] <= cut).to_numpy()
        for name, (side, sel) in RULES.items():
            s = sel(mid)
            if s.sum() < min_n:
                continue
            gg, e = g[s], early[s]
            if gg["day"].nunique() < min_events:      # too few independent events: bootstrap is meaningless
                continue
            pnl = pnl_rule(gg, side, fm.get(series, 1.0))
            m, lo, hi, p_boot = boot_mean(pnl, gg["day"], n_boot)
            ask = (gg["yes_ask"] if side == "yes" else 1 - gg["yes_bid"]).to_numpy()
            from .digital import fee_per_contract_amortised
            cpf = ask + fm.get(series, 1.0) * fee_per_contract_amortised(ask)
            p = max(p_boot, exact_binom_p(pnl, cpf, gg["day"]))     # the more conservative of the two
            row = dict(series=series, frac=frac, rule=name, n=len(gg), days=gg["day"].nunique(), pnl=m, lo=lo, hi=hi, p=p,
                       p_boot=p_boot, cost=float(np.mean(ask)))
            if gg["day"][e].nunique() >= min_events_half and gg["day"][~e].nunique() >= min_events_half:
                tm, tl, th, tp = boot_mean(pnl[~e], gg["day"][~e], n_boot)
                tp = max(tp, exact_binom_p(pnl[~e], cpf[~e], gg["day"][~e]))
                trm, trl, trh, trp = boot_mean(pnl[e], gg["day"][e], n_boot)
                row.update(train_pnl=trm, train_p=trp, test_pnl=tm, test_lo=tl, test_hi=th, test_p=tp)
            rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["fdr_sig"] = bh_fdr(t["p"].to_numpy())
    t["replicates"] = t["fdr_sig"] & (t.get("test_p", 1.0) < 0.05)
    return t


def sample_events(markets, n_events=60, per_event=3, min_volume=20, min_events=40):
    """Event-based sample: for each series, n_events events spread evenly over its history, and the
    `per_event` highest-volume liquid markets in each. Gives the many independent events per series
    that cluster-robust inference needs (sampling the most recent markets yields only a few days)."""
    m = markets.copy()
    for c in ("open_time", "close_time"):
        m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
    m = m[(m["volume"] >= min_volume) & m["open_time"].notna()]
    m["life_h"] = (m["close_time"] - m["open_time"]).dt.total_seconds() / 3600
    m = m[(m["life_h"] > 0.2) & (m["life_h"] < 24 * 14)]
    out = []
    for series, g in m.groupby("series"):
        ev = g.groupby("event_ticker")["close_time"].max().sort_values()
        if len(ev) < min_events:
            continue
        pick = ev.index[np.unique(np.linspace(0, len(ev) - 1, min(n_events, len(ev))).round().astype(int))]
        gg = g[g["event_ticker"].isin(pick)].sort_values("volume", ascending=False)
        out.append(gg.groupby("event_ticker").head(per_event))
    return pd.concat(out, ignore_index=True)


def _parse_batch(markets_json):
    rows = []
    for m in markets_json.get("markets", []):
        t = m.get("market_ticker") or m.get("ticker")
        for c in m.get("candlesticks", []):
            try:
                rows.append((t, c["end_period_ts"], _px(c.get("yes_bid")), _px(c.get("yes_ask")),
                             _num(c.get("volume_fp", c.get("volume")))))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def fetch_candles_batch(sample, out_csv, batch=40, workers=2, fallback=True):
    """Same output as fetch_candles but via GET /markets/candlesticks (many markets per call).
    Tickers the batch endpoint returns nothing for fall back to the per-market (live/historical) path."""
    import os
    done = set(pd.read_csv(out_csv, usecols=["ticker"])["ticker"]) if os.path.exists(out_csv) else set()
    todo = sample[~sample["ticker"].isin(done)].copy()
    todo["period"] = np.where(todo["life_h"] <= 6, 1, 60)
    jobs = []
    for period, g in todo.groupby("period"):
        g = g.sort_values("close_time")
        for i in range(0, len(g), batch):
            part = g.iloc[i:i + batch]
            jobs.append((period, list(part["ticker"]), part["open_time"].min(), part["close_time"].max()))

    def run(job):
        period, tickers, start, end = job
        try:
            d = get_json(f"{BASE}/markets/candlesticks", {"market_tickers": ",".join(tickers),
                         "start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()), "period_interval": int(period)})
            return tickers, _parse_batch(d)
        except Exception:
            return tickers, []

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for tickers, rows in ex.map(run, jobs):
            got = {r[0] for r in rows}
            missing = [t for t in tickers if t not in got]
            if fallback and missing:                 # per-market path (historical first for old markets)
                sub = todo[todo["ticker"].isin(missing)]
                for r in sub.itertuples():
                    rows += _candle_job((r.ticker, r.series, r.open_time, r.close_time, int(r.period)))
                got = {r[0] for r in rows}
            rows += [(t, 0, np.nan, np.nan, np.nan) for t in tickers if t not in got]
            pd.DataFrame(rows, columns=["ticker", "ts", "yes_bid", "yes_ask", "volume"]).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            n += len(tickers)
            print(f"{n}/{len(todo)}", flush=True)
