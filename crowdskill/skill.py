"""Dawid-Skene style trader reliability on prediction-market bets, evaluated prequentially.

Each bettor's *vote* in a market is the sign of their net signed stake in the first K bets (+1 YES, -1 NO).
For each bettor we keep smoothed confusion counts versus the eventual outcome, using ONLY markets that had
already resolved before the evaluated market's anchor time (a replay with no look-ahead). The log-likelihood
ratio of a vote under those counts is the evidence that bettor contributes; the market's anchor price is the
baseline the evidence must beat.
"""
from __future__ import annotations
import heapq
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def load_votes(markets, bet_dir: Path, K: int, min_users: int = 5):
    """List of dicts: id, anchor_t, resolve_t, y, p_anchor, votes {user: +1/-1}, users (distinct among first K)."""
    out = []
    for m in markets:
        f = bet_dir / f"{m['id']}.json"
        if not f.exists(): continue
        b = [x for x in json.loads(f.read_text())
             if not x.get("isCancelled") and not x.get("isRedemption") and x.get("outcome") in ("YES", "NO")
             and (x.get("amount") or 0) != 0 and x.get("probAfter") is not None]
        b.sort(key=lambda x: x["createdTime"])
        if len(b) < K: continue
        w = b[:K]
        net = defaultdict(float)
        for x in w: net[x["userId"]] += x["amount"] if x["outcome"] == "YES" else -x["amount"]
        votes = {u: (1 if s > 0 else -1) for u, s in net.items() if s != 0}
        if len(votes) < min_users: continue
        out.append(dict(id=m["id"], anchor_t=w[-1]["createdTime"], resolve_t=m["resolutionTime"],
                        y=int(m["resolution"] == "YES"), p_anchor=float(np.clip(w[-1]["probAfter"], 0.01, 0.99)),
                        votes=votes))
    out.sort(key=lambda r: r["anchor_t"])
    return out


class Confusion:
    """counts[u] = [[n(v=-1,y=0), n(v=-1,y=1)], [n(v=+1,y=0), n(v=+1,y=1)]]."""

    def __init__(self, prior: float = 2.0):
        self.c = defaultdict(lambda: np.zeros((2, 2))); self.prior = prior; self.cls = np.zeros(2)

    def add(self, votes, y, sign=1):
        for u, v in votes.items(): self.c[u][(v + 1) // 2, y] += sign
        self.cls[y] += sign

    def llr(self, u, v):
        """log P(v | y=1) / P(v | y=0) with Dirichlet smoothing; unseen bettors give exactly 0."""
        if u not in self.c: return 0.0
        c = self.c[u]; k = (v + 1) // 2; a = self.prior
        p1 = (c[k, 1] + a) / (c[:, 1].sum() + 2 * a)
        p0 = (c[k, 0] + a) / (c[:, 0].sum() + 2 * a)
        return float(np.log(p1 / p0))

    def known(self, u):
        return u in self.c and self.c[u].sum() > 0


def replay_features(records, prior: float = 2.0, shuffle_users: bool = False, seed: int = 0):
    """Prequential features: evidence at each anchor uses only markets resolved before that anchor."""
    rng = np.random.default_rng(seed)
    if shuffle_users:   # null: break identity but keep vote structure (permute user ids across the whole sample)
        ids = sorted({u for r in records for u in r["votes"]}); perm = dict(zip(ids, rng.permutation(ids)))
        records = [{**r, "votes": {perm[u]: v for u, v in r["votes"].items()}} for r in records]
    conf = Confusion(prior)
    pending = []   # heap of (resolve_t, idx)
    for i, r in enumerate(records): heapq.heappush(pending, (r["resolve_t"], i))
    feats = []
    for r in records:
        while pending and pending[0][0] < r["anchor_t"]:
            _, j = heapq.heappop(pending); conf.add(records[j]["votes"], records[j]["y"])
        llrs = [conf.llr(u, v) for u, v in r["votes"].items()]
        known = sum(conf.known(u) for u in r["votes"])
        feats.append(dict(id=r["id"], anchor_t=r["anchor_t"], resolve_t=r["resolve_t"], y=r["y"],
                          logit_p=float(np.log(r["p_anchor"] / (1 - r["p_anchor"]))), p=r["p_anchor"],
                          evidence=float(np.sum(llrs)), n_voters=len(r["votes"]), n_known=int(known),
                          net_vote=float(sum(r["votes"].values()))))
    return pd.DataFrame(feats)


def fit_logit(X, y, l2=1.0, iters=50):
    """Newton logistic regression with intercept; returns weights (intercept first)."""
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + R @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))
