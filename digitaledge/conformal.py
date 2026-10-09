"""Conformal prediction on market-implied distributions.

A bracket ladder (e.g. "high temp 77-78", "79-80", ... plus tails) is a discrete predictive distribution
for a numeric settlement value. We (1) build its CDF from mid prices, (2) take the probability integral
transform (PIT) of the realised outcome, and (3) wrap central prediction sets in split / adaptive
conformal prediction (Gibbs & Candes 2021, "Adaptive Conformal Inference Under Distribution Shift").
The question is distribution-free: do the market's own 90% intervals contain the outcome 90% of the time?
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Ladder:
    event: str
    series: str
    close: float
    edges: np.ndarray      # interior bin edges, increasing (len k-1 for k bins)
    cdf: np.ndarray        # F at each edge
    lo: int                # index of the realised bin: PIT in [F_{lo-1}, F_lo]
    width: float           # mean interior bin width in native units (nan if unknown)

    def pit_interval(self):
        F = np.concatenate([[0.0], self.cdf, [1.0]])
        return F[self.lo], F[self.lo + 1]

    def quantile(self, u: float) -> float:
        """Inverse CDF with linear interpolation inside bins (tails clipped at one bin width)."""
        F = np.concatenate([[0.0], self.cdf, [1.0]])
        x = np.concatenate([[self.edges[0] - self.width], self.edges, [self.edges[-1] + self.width]])
        return float(np.interp(u, F, x))


def _bins_to_ladder(g: pd.DataFrame) -> Ladder | None:
    """Contiguous less / between / greater bins -> normalised CDF at bin edges."""
    rows = []
    for r in g.itertuples():
        if r.strike_type == "less":
            lo, hi = -np.inf, r.cap_strike
        elif r.strike_type == "between":
            lo, hi = r.floor_strike, r.cap_strike + 1
        elif r.strike_type in ("greater", "greater_or_equal"):
            lo, hi = r.floor_strike + (1 if r.strike_type == "greater" else 0), np.inf
        else:
            return None
        rows.append((lo, hi, r.mid, r.result == "yes"))
    rows.sort()
    for a, b in zip(rows[:-1], rows[1:]):
        if abs(a[1] - b[0]) > 1e-9:
            return None
    p = np.array([r[2] for r in rows]); yes = np.array([r[3] for r in rows])
    if yes.sum() != 1 or p.sum() <= 0:
        return None
    p = p / p.sum()
    edges = np.array([r[1] for r in rows[:-1]])
    w = np.diff(edges).mean() if len(edges) > 1 else 1.0
    return Ladder(g.event_ticker.iloc[0], g.series.iloc[0], g.close.iloc[0], edges, np.cumsum(p)[:-1],
                  int(np.flatnonzero(yes)[0]), float(w))


def _greater_to_ladder(g: pd.DataFrame) -> Ladder | None:
    """Nested P(S > k) ladder -> CDF at strikes (isotone-repaired); outcome bin from results."""
    g = g.sort_values("floor_strike")
    k = g.floor_strike.to_numpy(float)
    p = np.minimum.accumulate(g.mid.to_numpy(float))
    F = 1 - p
    yes = (g.result == "yes").to_numpy()
    if not yes.any() and not (~yes).any():
        return None
    lo = int(yes.sum())  # strikes beaten -> outcome bin index (monotone results)
    if not np.all(np.diff(yes.astype(int)) <= 0):
        return None
    w = float(np.diff(k).mean()) if len(k) > 1 else 1.0
    return Ladder(g.event_ticker.iloc[0], g.series.iloc[0], g.close.iloc[0], k, F, lo, w)


def build_ladders(markets: pd.DataFrame, candles: pd.DataFrame, horizon_h: float, slack_h: float = 3.0):
    """One Ladder per fully-priced event, priced at the last quote >= horizon_h before close."""
    m = markets[markets.strike_type.isin(["less", "between", "greater", "greater_or_equal"])].copy()
    m = m[m.result.isin(["yes", "no"])]
    m["close"] = (pd.to_datetime(m.close_time, utc=True, format="ISO8601") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    c = candles.dropna(subset=["yes_bid", "yes_ask"]).copy()
    c["mid"] = (c.yes_bid + c.yes_ask) / 2
    j = m.merge(c[["ticker", "ts", "mid"]], on="ticker")
    j["h"] = (j.close - j.ts) / 3600
    j = j[(j.h >= horizon_h) & (j.h <= horizon_h + slack_h)]
    j = j.sort_values("ts").groupby("ticker").tail(1)
    n_ev = m.groupby("event_ticker").ticker.size()
    out = []
    for ev, g in j.groupby("event_ticker"):
        if len(g) != n_ev[ev] or len(g) < 4:
            continue
        types = set(g.strike_type)
        lad = _greater_to_ladder(g) if types <= {"greater", "greater_or_equal"} else _bins_to_ladder(g)
        if lad is not None:
            out.append(lad)
    out.sort(key=lambda l: l.close)
    return out


def randomized_pit(ladders, seed=0, reps=1):
    rng = np.random.default_rng(seed)
    lo = np.array([l.pit_interval()[0] for l in ladders]); hi = np.array([l.pit_interval()[1] for l in ladders])
    return lo[None] + rng.random((reps, len(lo))) * (hi - lo)[None]


def central_score(u):
    return np.abs(u - 0.5)


def split_conformal(scores_cal, alpha):
    n = len(scores_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores_cal)[k - 1]


def aci(scores, alpha, gamma=0.02, warm=30):
    """Online adaptive conformal. Returns (miss indicators, q_t) for t>=warm."""
    a_t, miss, qs = alpha, [], []
    for t in range(warm, len(scores)):
        past = scores[:t]
        lvl = min(max(1 - a_t, 0.0), 1.0)
        q = np.inf if lvl >= 1 else np.quantile(past, lvl, method="higher")
        err = float(scores[t] > q)
        miss.append(err); qs.append(q)
        a_t += gamma * (alpha - err)
    return np.array(miss), np.array(qs)
