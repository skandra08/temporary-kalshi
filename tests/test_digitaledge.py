import numpy as np
import pandas as pd
import pytest
from scipy import stats
from digitaledge.backtest import trades
from digitaledge.digital import (digital_prob, effective_horizon, fee_per_contract_amortised,
                                 implied_vol_per_min, kalshi_fee)
from digitaledge.scoring import (brier, calibration_slope, cluster_bootstrap, logloss,
                                 paired_diff, reliability)
from digitaledge.volmodels import MinuteData, predict


def test_digital_matches_black_scholes_nd2():
    S, K, sig, tau = 100.0, 101.0, 0.001, 60.0
    d2 = (np.log(S / K) - 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    p = digital_prob(S, K, sig**2, tau, settlement_adjust=False)
    assert p == pytest.approx(stats.norm.cdf(d2))


def test_digital_monotone_in_strike_and_bounds():
    K = np.linspace(90, 110, 50)
    p = digital_prob(100, K, 1e-6, 30)
    assert np.all(np.diff(p) <= 0) and p.min() >= 0 and p.max() <= 1


def test_settlement_average_shortens_horizon():
    assert effective_horizon(5) == pytest.approx(5 - 2 / 3)
    assert digital_prob(100, 100.0, 1e-6, 5) != digital_prob(100, 100.0, 1e-6, 5, settlement_adjust=False)


def test_implied_vol_roundtrip():
    p = digital_prob(100, 100.5, 4e-7, 20)
    assert implied_vol_per_min(100, 100.5, float(p), 20) == pytest.approx(np.sqrt(4e-7), rel=1e-5)


def test_fat_tails_raise_far_otm_probability():
    g = digital_prob(100, 103, 1e-6, 30)
    t = digital_prob(100, 103, 1e-6, 30, dist="t", nu=3.5)
    assert t > g


def test_kalshi_fee_known_values():
    assert kalshi_fee(0.5) == 0.02                      # ceil(1.75c)
    assert kalshi_fee(0.5, contracts=100) == 1.75
    assert kalshi_fee(0.01) == 0.01                     # ceil(0.0693c)
    assert fee_per_contract_amortised(0.5) == pytest.approx(0.0175)
    assert kalshi_fee(0.5, 100, maker=True) == pytest.approx(0.44)


def test_cluster_bootstrap_ci_wider_than_naive_for_correlated_data():
    rng = np.random.default_rng(0)
    ev = np.repeat(np.arange(60), 20)
    x = np.repeat(rng.normal(size=60), 20) + 0.05 * rng.normal(size=1200)   # near-duplicate within event
    _, lo, hi = cluster_bootstrap(x, ev, seed=1)
    naive = 1.96 * x.std() / np.sqrt(len(x))
    assert (hi - lo) / 2 > 3 * naive


def test_scoring_rules_prefer_the_truth():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert brier(p, y).mean() < brier(np.full_like(p, 0.5), y).mean()
    assert logloss(p, y).mean() < logloss(np.clip(p + 0.2, 0, 1), y).mean()
    a, b = calibration_slope(p, y)
    assert abs(a) < 0.1 and b == pytest.approx(1, abs=0.1)
    r = reliability(p, y)
    assert np.allclose(r["p"], r["y"], atol=0.05)


def _synthetic(n_days=45, sigma_ann=0.5, seed=3):
    """Minute GBM with known constant vol, observations at random strikes, true outcomes."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    sig = sigma_ann / np.sqrt(525600)
    lp = np.log(80000) + np.cumsum(sig * rng.normal(size=n)) - 0.5 * sig**2 * np.arange(n)
    px = pd.Series(np.exp(lp), index=pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"))
    rows = []
    tau = 30
    for pos in range(35 * 1440, n - tau - 1, 60):
        t = px.index[pos]
        spot = px.iloc[pos]
        for m in (-1.5, -0.5, 0, 0.5, 1.5):
            K = spot * np.exp(m * sig * np.sqrt(tau))
            rows.append(dict(obs_time=t, tau=tau, spot=spot, strike=K, dvol=sigma_ann,
                             outcome=int(px.iloc[pos + tau] >= K), event_ticker=str(t)))
    return pd.DataFrame(rows), px


def test_models_are_calibrated_on_synthetic_gbm_and_no_edge_vs_fair_quotes():
    obs, px = _synthetic()
    out = predict(obs, px)
    for m in ("gauss_dvol", "gauss_har", "t_har", "emp_har"):
        a, b = calibration_slope(out["p_" + m], out["outcome"])
        assert abs(a) < 0.25 and 0.7 < b < 1.3, (m, a, b)
    # market quoting exactly the true probability: trading on model must not make money net of fees
    out["yes_bid"] = out["yes_ask"] = out["p_gauss_dvol"]
    tr = trades(out, "p_gauss_har", edge=0.0)
    assert len(tr) < 0.05 * len(out) or tr["pnl"].mean() < 0.005   # fees kill ~all signals


def test_har_forecast_uses_no_future_information():
    """Perturbing prices after an observation day must not change that day's forecasts."""
    obs, px = _synthetic(n_days=40, seed=5)
    day = obs["obs_time"].dt.floor("D").iloc[0] + pd.Timedelta(days=1)
    first = obs[obs["obs_time"] < day]
    base = predict(first, px)["p_t_har"]
    px2 = px.copy()
    px2[px2.index >= day] *= 1.5          # shock the future
    assert np.allclose(base, predict(first, px2)["p_t_har"])


def test_trade_logic_sides_and_fees():
    df = pd.DataFrame(dict(p=[0.9, 0.1], yes_bid=[0.5, 0.5], yes_ask=[0.55, 0.55], outcome=[1, 0]))
    t = trades(df, "p")
    assert list(t["side"]) == ["yes", "no"]
    assert t["pnl"].iloc[0] == pytest.approx(1 - 0.55 - fee_per_contract_amortised(0.55))
    assert t["pnl"].iloc[1] == pytest.approx(1 - 0.5 - fee_per_contract_amortised(0.5))


def test_svi_digital_equals_numerical_minus_dCdK_and_density_ok():
    from digitaledge.svi import SVI, call_price, density_nonneg, digital_prob_svi
    s, F, T = SVI(0.01, 0.1, -0.3, 0.0, 0.15), 100.0, 1 / 52
    K, h = np.array([90, 95, 100, 105, 110.0]), 1e-3
    num = -(call_price(F, K * (1 + h), T, s) - call_price(F, K * (1 - h), T, s)) / (2 * h * K)
    assert np.allclose(num, digital_prob_svi(F, K, T, s), atol=1e-4)
    assert density_nonneg(s)[0]


def test_svi_fit_recovers_known_smile():
    from digitaledge.svi import SVI, fit_svi
    true = SVI(0.02, 0.12, -0.4, 0.02, 0.1)
    k = np.linspace(-0.3, 0.3, 25)
    fit = fit_svi(k, true.w(k))
    assert np.allclose(fit.w(k), true.w(k), atol=1e-6)


def test_skew_adjusted_digital_consistent_with_direct_svi_and_flat_smile():
    from digitaledge.svi import SVI, digital_prob_svi, skew_adjusted_digital
    s = SVI(0.0004, 0.01, -0.5, 0.0, 0.05)
    tau_min = 30.0
    tau_eff = float(effective_horizon(tau_min))
    F = 80000.0
    K = F * np.exp(np.linspace(-0.01, 0.01, 9))
    sig_atm = np.sqrt(s.w(0.0) / tau_eff)
    got = skew_adjusted_digital(F, K, sig_atm, tau_min, s, tau_min)
    assert np.allclose(got, digital_prob_svi(F, K, 1.0, s), atol=1e-4)
    flat = SVI(0.0004, 1e-9, 0.0, 0.0, 0.5)
    assert np.allclose(skew_adjusted_digital(F, K, sig_atm, tau_min, flat, tau_min),
                       digital_prob(F, K, sig_atm**2, tau_min), atol=1e-4)
