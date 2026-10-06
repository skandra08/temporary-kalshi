"""Sports announcer-mention markets: does team / low-rank (SVD) structure beat per-word base rates?
    python scripts/mentions_sports_experiment.py"""
import os
import pandas as pd
from digitaledge import mentions as M, scoring

SERIES = ["KXNBAMENTION", "KXNFLMENTION", "KXMLBMENTION", "KXNCAABMENTION", "KXWCMENTION", "KXFIGHTMENTION"]
p = M.panel(pd.concat([M.list_full(s) for s in SERIES], ignore_index=True))
f = M.build_features(p)
w = M.walk_forward(f)
os.makedirs("results", exist_ok=True)
print(f"{len(w)} word-markets, {w.event_ticker.nunique()} games evaluated walk-forward")
rows = []
for m_ in ["p_global", "p_word", "p_lr_word", "p_lr_word+team", "p_lr_full(+lowrank)", "p_gbm"]:
    ll, br = scoring.logloss(w[m_], w.yes), scoring.brier(w[m_], w.yes)
    d = scoring.paired_diff(ll, scoring.logloss(w["p_word"], w.yes), w.event_ticker, n_boot=2000)
    rows.append(dict(model=m_[2:], logloss=ll.mean(), brier=br.mean(), d_logloss_vs_word_rate=d[0], ci_lo=d[1], ci_hi=d[2]))
r = pd.DataFrame(rows).round(4)
print(r.to_string(index=False))
r.to_csv("results/mentions_sports_scores.csv", index=False)
