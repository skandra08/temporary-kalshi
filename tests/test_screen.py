import numpy as np
import pandas as pd
from digitaledge import screen as S


def _fake(n_series=25, n=600, planted=("S0", "S1"), edge=0.12, seed=0):
    """Quotes where YES is fairly priced everywhere except planted series, where YES is
    overpriced by `edge` (so buying NO has a true edge)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        name = f"S{s}"
        close = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
        for i, c in enumerate(close):
            true_p = rng.choice([0.05, 0.3, 0.6])
            price = true_p + (edge if name in planted else 0.0)
            y = rng.random() < true_p
            rows.append(dict(series=name, frac=0.5, ticker=f"{name}-{i}", close_time=c, result="yes" if y else "no",
                             yes_bid=max(price - .01, .01), yes_ask=min(price + .01, .99)))
    return pd.DataFrame(rows)


def test_bh_fdr_basic():
    p = np.array([0.001, 0.009, 0.04, 0.2, 0.7])
    m = S.bh_fdr(p, 0.05)
    assert m.tolist() == [True, True, False, False, False]


def test_screen_finds_planted_edges_and_controls_false_positives():
    q = _fake()
    t = S.evaluate(q, n_boot=500)
    hits = set(t[t.fdr_sig & (t["rule"] == "buy_no_all")]["series"])
    assert {"S0", "S1"} <= hits
    false_hits = hits - {"S0", "S1"}
    assert len(false_hits) <= 1                    # FDR-controlled: essentially no spurious series
    assert t[(t.series == "S0") & (t["rule"] == "buy_no_all")].replicates.iloc[0]


def test_quotes_use_no_future_information():
    close = pd.Timestamp("2026-01-02", tz="UTC"); open_ = pd.Timestamp("2026-01-01", tz="UTC")
    sample = pd.DataFrame(dict(ticker=["T"], series=["S"], open_time=[open_], close_time=[close], result=["yes"], volume=[100.0]))
    ts = [int((open_ + pd.Timedelta(hours=h)).timestamp()) for h in (1, 10, 11, 13, 20)]
    candles = pd.DataFrame(dict(ticker="T", ts=ts, yes_bid=[.1, .2, .3, .9, .95], yes_ask=[.2, .3, .4, .95, .99], volume=1.0))
    q = S.quotes_at_fractions(sample, candles, fracs=(0.5,), max_stale_frac=0.5)
    # target = open + 12h: the last candle at or before it is the 11h one (bid .3), never the 13h one
    assert q["yes_bid"].iloc[0] == 0.3
