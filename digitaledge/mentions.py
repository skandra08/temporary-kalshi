"""Mention markets ("will the announcer/speaker say WORD during EVENT?"): panel data + causal features.

A market resolves YES the moment the word is said, so `close_time` leaks the outcome for YES
markets. Features here use only events whose *settlement* happened before the target event opened."""
import re
import numpy as np
import pandas as pd
from .http import get_json
from .sources.kalshi import BASE

COLS = ["series", "ticker", "event_ticker", "yes_sub_title", "result", "volume", "open_time", "close_time",
        "title", "rules_primary", "subtitle"]


def list_full(series, max_pages=2):
    """Settled markets with all text fields (same requests as screen.list_series_markets, so cached)."""
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
        if len(rows) >= 300:
            break
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"])].copy()
    df["volume"] = pd.to_numeric(df.get("volume_fp", df.get("volume")), errors="coerce")
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df[COLS]


_EVENT_RE = re.compile(r"as part of\s+(.+?)\s*(?:,|\n|$)", re.S)


def event_name(rules):
    m = _EVENT_RE.search(str(rules))
    return " ".join(m.group(1).split()) if m else ""


def teams_from_name(name):
    """'Spurs vs Knicks Professional Basketball Game' -> ('spurs', 'knicks'); empty if not 'A vs B'."""
    n = re.sub(r"\b(professional|college|basketball|football|baseball|soccer|game|match|fight|mma)\b.*$", "", name,
               flags=re.I).strip()
    parts = re.split(r"\s+(?:vs\.?|v\.?|at|@)\s+", n, flags=re.I)
    return tuple(p.strip().lower() for p in parts[:2]) if len(parts) == 2 else ()


def panel(markets):
    """One row per (event, word) with a normalised word key, event name and teams."""
    m = markets.copy()
    m["word"] = m["yes_sub_title"].astype(str).str.lower().str.strip()
    m["yes"] = (m["result"] == "yes").astype(int)
    ev = m.groupby("event_ticker").agg(ev_open=("open_time", "min"), settle_time=("close_time", "max"),
                                       rules=("rules_primary", "first"))
    ev["event_name"] = ev["rules"].map(event_name)
    ev["teams"] = ev["event_name"].map(teams_from_name)
    m = m.merge(ev[["event_name", "teams", "settle_time", "ev_open"]], left_on="event_ticker", right_index=True)
    return m[m["word"] != ""]


# ------------------------------------------------------------------ causal features
from collections import defaultdict, deque


def build_features(p, a=5.0, b=3.0, rank=3, recent=8):
    """Per (event, word) features using only events settled before the target event opened.

    word_rate : shrunk word base rate in the series; recent : mean of the word's last `recent` outcomes
    team_eff  : mean over the two teams of the shrunk (team,word) residual vs word_rate
    lowrank   : same residual, reconstructed from a rank-`rank` SVD of the team x word residual matrix
                (borrows strength across teams/words, the PCA piece)"""
    out = []
    for series, g in p.groupby("series"):
        ev = g.drop_duplicates("event_ticker").sort_values("ev_open")
        by_settle = ev.sort_values("settle_time")["event_ticker"].tolist()
        rows_by_ev = {e: d for e, d in g.groupby("event_ticker")}
        meta = ev.set_index("event_ticker")
        w_n, w_hit = defaultdict(int), defaultdict(float)
        tw_n, tw_hit = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(float))
        hist = defaultdict(lambda: deque(maxlen=recent))
        tot_n = tot_hit = 0
        ptr, n_prior = 0, 0
        for e in ev["event_ticker"]:
            t_open = meta.loc[e, "ev_open"]
            while ptr < len(by_settle) and meta.loc[by_settle[ptr], "settle_time"] < t_open:
                d = rows_by_ev[by_settle[ptr]]
                teams = meta.loc[by_settle[ptr], "teams"]
                for w, y in zip(d["word"], d["yes"]):
                    w_n[w] += 1; w_hit[w] += y; hist[w].append(y); tot_n += 1; tot_hit += y
                    for t in teams:
                        tw_n[t][w] += 1; tw_hit[t][w] += y
                ptr += 1; n_prior += 1
            d = rows_by_ev[e]
            teams = meta.loc[e, "teams"]
            gl = (tot_hit + 0.4 * 20) / (tot_n + 20)
            words = list(d["word"])

            def wrate(w):
                return (w_hit[w] + a * gl) / (w_n[w] + a)
            # low-rank smoothing of the team x word residual matrix (rebuilt per event; small)
            tlist = [t for t in tw_n if tw_n[t]]
            wlist = sorted(w_n)
            lr = {}
            if len(tlist) >= 3 and len(wlist) >= 3:
                widx = {w: i for i, w in enumerate(wlist)}
                R = np.zeros((len(tlist), len(wlist)))
                for ti, t in enumerate(tlist):
                    for w, n in tw_n[t].items():
                        R[ti, widx[w]] = (tw_hit[t][w] - n * wrate(w)) / (n + b)
                U, S, Vt = np.linalg.svd(R, full_matrices=False)
                k = min(rank, len(S))
                Rh = (U[:, :k] * S[:k]) @ Vt[:k]
                tidx = {t: i for i, t in enumerate(tlist)}
                for t in teams:
                    if t in tidx:
                        for w in words:
                            if w in widx:
                                lr.setdefault(w, []).append(Rh[tidx[t], widx[w]])
            for r in d.itertuples():
                w = r.word
                wr = wrate(w)
                h = hist[w]
                rec = float(np.mean(h)) if len(h) >= 3 else wr
                te = []
                for t in teams:
                    n = tw_n[t].get(w, 0) if t in tw_n else 0
                    if n:
                        te.append((tw_hit[t][w] + b * wr) / (n + b) - wr)
                out.append(dict(ticker=r.ticker, series=series, event_ticker=e, word=w, yes=r.yes, ev_open=t_open,
                                settle_time=meta.loc[e, "settle_time"], n_prior_events=n_prior, n_w=w_n[w], global_rate=gl,
                                word_rate=wr, recent=rec, team_eff=float(np.mean(te)) if te else 0.0,
                                lowrank=float(np.mean(lr[w])) if w in lr else 0.0, volume=r.volume))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ walk-forward evaluation
FEATS = {
    "lr_word": ["lw", "lrec", "ln"],
    "lr_word+team": ["lw", "lrec", "ln", "team_eff"],
    "lr_full(+lowrank)": ["lw", "lrec", "ln", "team_eff", "lowrank"],
}


def _prep(f):
    f = f.copy()
    clip = lambda x: np.clip(x, 0.01, 0.99)
    f["lw"] = np.log(clip(f["word_rate"]) / (1 - clip(f["word_rate"])))
    f["lrec"] = np.log(clip(f["recent"]) / (1 - clip(f["recent"])))
    f["ln"] = np.log1p(f["n_w"])
    return f


def walk_forward(f, burn_in=15, block=20, seed=0):
    """Predict each event from models trained only on events settled before the block opened."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    f = _prep(f).sort_values("ev_open").reset_index(drop=True)
    ev = f.drop_duplicates("event_ticker")[["event_ticker", "ev_open", "n_prior_events"]]
    ev = ev[ev["n_prior_events"] >= burn_in].sort_values("ev_open")
    preds = []
    for i in range(0, len(ev), block):
        blk = ev.iloc[i:i + block]
        t0 = blk["ev_open"].min()
        tr = f[(f["settle_time"] < t0) & (f["n_prior_events"] >= 3)]
        te = f[f["event_ticker"].isin(blk["event_ticker"])].copy()
        if len(tr) < 500 or tr["yes"].nunique() < 2:
            continue
        te["p_global"] = te["global_rate"]
        te["p_word"] = te["word_rate"]
        for name, cols in FEATS.items():
            m = LogisticRegression(C=1.0, max_iter=500).fit(tr[cols], tr["yes"])
            te["p_" + name] = m.predict_proba(te[cols])[:, 1]
        gcols = ["lw", "lrec", "ln", "team_eff", "lowrank", "global_rate"]
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed,
                                            l2_regularization=1.0).fit(tr[gcols], tr["yes"])
        te["p_gbm"] = gb.predict_proba(te[gcols])[:, 1]
        preds.append(te)
    return pd.concat(preds, ignore_index=True)


# ------------------------------------------------------------------ earnings-call dataset (SEC text features)
def earnings_panel(mentions_markets_csv="data/mentions_markets.csv"):
    base = pd.read_csv(mentions_markets_csv, usecols=["series"])["series"].unique()
    ser = [s for s in base if s.startswith("KXEARNINGSMENTION")]
    p = panel(pd.concat([list_full(s) for s in ser], ignore_index=True))
    p["co"] = p["series"].str.replace("KXEARNINGSMENTION", "", regex=False)
    return p


def add_filing_features(p, progress=None):
    """For each event: the company's last 10-Q/10-K filed before the market opened, and each word's count in it."""
    from . import edgar
    t2c = edgar.ticker_to_cik()
    rows, text_cache = [], {}
    events = p.drop_duplicates("event_ticker")[["event_ticker", "co", "ev_open"]]
    info = {}
    for i, e in enumerate(events.itertuples()):
        cik = t2c.get(e.co.upper())
        f = edgar.last_periodic_before(cik, e.ev_open) if cik else None
        if f is None:
            info[e.event_ticker] = None
            continue
        key = (cik, f["accessionNumber"])
        if key not in text_cache:
            text_cache[key] = edgar.filing_text(cik, f)
        info[e.event_ticker] = (f["filingDate"], f["form"], text_cache[key])
        if progress and i % 20 == 0:
            progress(i, len(events))
    out = []
    for r in p.itertuples():
        inf = info.get(r.event_ticker)
        if inf is None or not inf[2]:
            continue
        txt = inf[2]
        c = edgar.phrase_count(txt, r.word)
        out.append(dict(ticker=r.ticker, event_ticker=r.event_ticker, co=r.co, word=r.word, yes=r.yes, volume=r.volume,
                        ev_open=r.ev_open, settle_time=r.settle_time, filing_date=inf[0], form=inf[1],
                        doc_len=len(txt), count=c))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ earnings experiment (text features vs base rates)
def earnings_features(d, a=5.0):
    """Causal features per row: shrunk word prior from earlier settled calls, plus SEC-filing word count."""
    d = d.sort_values("ev_open").reset_index(drop=True)
    by_settle = d.drop_duplicates("event_ticker").sort_values("settle_time")
    ev_open = d.groupby("event_ticker")["ev_open"].first()
    order = list(by_settle["event_ticker"])
    w_n, w_hit, tot_n, tot_hit, ptr = {}, {}, 0, 0.0, 0
    co_n, co_hit = {}, {}
    rows = []
    for e, g in d.groupby("event_ticker", sort=False):
        t0 = ev_open[e]
        while ptr < len(order) and by_settle.loc[by_settle["event_ticker"] == order[ptr], "settle_time"].iloc[0] < t0:
            for r in d[d["event_ticker"] == order[ptr]].itertuples():
                w_n[r.word] = w_n.get(r.word, 0) + 1; w_hit[r.word] = w_hit.get(r.word, 0.0) + r.yes
                co_n[r.co] = co_n.get(r.co, 0) + 1; co_hit[r.co] = co_hit.get(r.co, 0.0) + r.yes
                tot_n += 1; tot_hit += r.yes
            ptr += 1
        gl = (tot_hit + 0.5 * 20) / (tot_n + 20)
        for r in g.itertuples():
            wr = (w_hit.get(r.word, 0.0) + a * gl) / (w_n.get(r.word, 0) + a)
            co = r.co
            cr = (co_hit.get(co, 0.0) + 5 * gl) / (co_n.get(co, 0) + 5)       # company 'talkativeness'
            rows.append(dict(ticker=r.ticker, event_ticker=e, co=co, word=r.word, yes=r.yes, volume=r.volume,
                             ev_open=t0, settle_time=r.settle_time, n_w=w_n.get(r.word, 0), global_rate=gl,
                             word_rate=wr, co_rate=cr, count=r.count, doc_len=r.doc_len,
                             density=1e4 * r.count / max(r.doc_len / 6, 1)))
    f = pd.DataFrame(rows)
    clip = lambda x: np.clip(x, 0.02, 0.98)
    f["lw"] = np.log(clip(f["word_rate"]) / (1 - clip(f["word_rate"])))
    f["lco"] = np.log(clip(f["co_rate"]) / (1 - clip(f["co_rate"])))
    f["lcount"] = np.log1p(f["count"])
    f["has_text"] = (f["count"] > 0).astype(float)
    f["ldens"] = np.log1p(f["density"])
    return f


def earnings_walk_forward(f, block=15, burn_in_events=30):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    ev = f.drop_duplicates("event_ticker").sort_values("ev_open")
    ev = ev.iloc[burn_in_events:]
    sets = {"lr_prior": ["lw", "lco"], "lr_prior+text": ["lw", "lco", "lcount", "has_text", "ldens"]}
    preds = []
    for i in range(0, len(ev), block):
        blk = ev.iloc[i:i + block]
        t0 = blk["ev_open"].min()
        tr = f[f["settle_time"] < t0]
        te = f[f["event_ticker"].isin(blk["event_ticker"])].copy()
        if len(tr) < 300 or tr["yes"].nunique() < 2:
            continue
        te["p_global"] = te["global_rate"]; te["p_word"] = te["word_rate"]
        for name, cols in sets.items():
            te["p_" + name] = LogisticRegression(C=1.0, max_iter=500).fit(tr[cols], tr["yes"]).predict_proba(te[cols])[:, 1]
        g = ["lw", "lco", "lcount", "has_text", "ldens"]
        te["p_gbm"] = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, l2_regularization=1.0,
                                                     random_state=0).fit(tr[g], tr["yes"]).predict_proba(te[g])[:, 1]
        preds.append(te)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
