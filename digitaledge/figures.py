"""Figures for the README from the study's observation CSV and report.json."""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import scoring
from .digital import implied_vol_per_min
from .volmodels import MODELS

MIN_PER_YEAR = 525600


def reliability_fig(obs, tau, path):
    d = obs[obs["tau"] == tau]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for name, col in [("Kalshi mid", "mid"), ("t_har", "p_t_har"), ("gauss_dvol", "p_gauss_dvol")]:
        r = scoring.reliability(d[col], d["outcome"], bins=10)
        ax.plot(r["p"], r["y"], "o-", ms=4, label=name)
    ax.set_xlabel("forecast probability"); ax.set_ylabel("realised frequency")
    ax.set_title(f"Reliability, {tau} min to settlement"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def dbrier_fig(report, path):
    s = pd.DataFrame(report["scores"])
    s = s[s["model"] != "market_mid"]
    taus = sorted(s["tau"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        d = s[s["model"] == m].set_index("tau").loc[taus]
        x = np.arange(len(taus)) + i * w
        ax.errorbar(x, d["d_brier_vs_mkt"], yerr=[d["d_brier_vs_mkt"] - d["ci_lo"], d["ci_hi"] - d["d_brier_vs_mkt"]],
                    fmt="o", ms=4, capsize=2, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(taus)) + 0.4 - w / 2, [f"{t}m" for t in taus])
    ax.set_ylabel("Brier(model) - Brier(Kalshi mid)\n<0 = model better")
    ax.set_title("Model vs market, 95% event-clustered CI"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def implied_vol_fig(obs, path, tau=30):
    d = obs[(obs["tau"] == tau) & (obs["mid"].between(0.2, 0.8))].copy()
    d["iv"] = [implied_vol_per_min(s, k, p, tau) for s, k, p in zip(d["spot"], d["strike"], d["mid"])]
    d = d.dropna(subset=["iv"])
    d["iv_ann"] = d["iv"] * np.sqrt(MIN_PER_YEAR)
    h = d.groupby(d["obs_time"].dt.floor("D")).agg(kalshi=("iv_ann", "median"), dvol=("dvol", "mean"),
                                                   har=("var_har", lambda v: np.sqrt(v.mean() * MIN_PER_YEAR)))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    h.plot(ax=ax, marker="o", ms=3)
    ax.set_ylabel("annualised vol"); ax.set_title("Kalshi-implied (30m, near-ATM) vs Deribit DVOL vs HAR forecast")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    rep = json.load(open("results/report.json"))
    obs = pd.read_csv("results/report_obs.csv", parse_dates=["obs_time"])
    reliability_fig(obs, 30, "results/reliability_30m.png")
    dbrier_fig(rep, "results/dbrier.png")
    implied_vol_fig(obs, "results/implied_vol.png")
    print("figures written")
