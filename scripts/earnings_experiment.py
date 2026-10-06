"""Earnings-call mention markets: do SEC-filing word counts predict which words get said, beyond base rates?

    python scripts/earnings_experiment.py            # full run (resumable; SEC + Kalshi responses are cached)
    python scripts/earnings_experiment.py --limit 60 # quick test on the first 60 events

Set EDGAR_UA to 'your-project your-email' (SEC asks for a contact in the User-Agent)."""
import argparse
import os
import pandas as pd
from digitaledge import mentions as M, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="only the first N events (by open time)")
a = ap.parse_args()

cache = "data/earnings_text.pkl"
if os.path.exists(cache) and not a.limit:
    d = pd.read_pickle(cache)
else:
    p = M.earnings_panel()
    if a.limit:
        keep = p.drop_duplicates("event_ticker").sort_values("ev_open").head(a.limit)["event_ticker"]
        p = p[p["event_ticker"].isin(keep)]
    d = M.add_filing_features(p, progress=lambda i, n: print(f"  filings {i}/{n}", flush=True))
    if not a.limit:
        d.to_pickle(cache)
print(f"{len(d)} word-markets, {d.event_ticker.nunique()} earnings calls, {d.co.nunique()} companies")

f = M.earnings_features(d)
burn = max(5, int(0.25 * f.event_ticker.nunique())) if a.limit else 30
w = M.earnings_walk_forward(f, burn_in_events=burn)
if w.empty:
    raise SystemExit("not enough events for a walk-forward evaluation")
os.makedirs("results", exist_ok=True)
w.to_pickle("results/earnings_walkforward.pkl")
print(f"\nWalk-forward on {len(w)} word-markets / {w.event_ticker.nunique()} calls (models see only calls settled before each block opened)")
rows = []
for m_ in ["p_global", "p_word", "p_lr_prior", "p_lr_prior+text", "p_gbm"]:
    ll, br = scoring.logloss(w[m_], w.yes), scoring.brier(w[m_], w.yes)
    d_ = scoring.paired_diff(ll, scoring.logloss(w["p_word"], w.yes), w.event_ticker, n_boot=2000)
    rows.append(dict(model=m_[2:], logloss=ll.mean(), brier=br.mean(), d_logloss_vs_word_rate=d_[0], ci_lo=d_[1], ci_hi=d_[2]))
r = pd.DataFrame(rows).round(4)
print(r.to_string(index=False))
r.to_csv("results/earnings_scores.csv", index=False)
print("\n(negative d_logloss = better than the plain per-word base rate; CI is clustered by earnings call)")
