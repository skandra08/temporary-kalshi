import numpy as np
import pandas as pd
import pytest
from quantlab import backtest, momentum, mean_reversion, simulate_prices, walk_forward
from quantlab.metrics import deflated_sharpe, max_drawdown, probabilistic_sharpe, sharpe


def test_no_same_bar_lookahead():
    """A signal built from today's return (known at close t) must not earn that same
    return. The engine lags weights one bar; a naive same-bar pairing would show an
    absurd Sharpe, which is exactly the bug this guards against."""
    p = simulate_prices(1500, 6, seed=1)
    r = p.pct_change().fillna(0)
    sig = np.sign(r)
    sig = sig.div(sig.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    assert abs(sharpe(backtest(p, sig, cost_bps=0).returns)) < 1.0
    assert sharpe((sig * r).sum(axis=1)) > 10  # what a leaky engine would report


def test_costs_reduce_returns():
    p = simulate_prices(800, 4, seed=2)
    s = mean_reversion(p, 10)
    free, costly = backtest(p, s, 0), backtest(p, s, 20)
    assert costly.returns.sum() < free.returns.sum()
    assert costly.turnover.equals(free.turnover)


def test_momentum_finds_trending_edge():
    p = simulate_prices(3000, 10, mu=0, phi=0.15, seed=3)
    assert sharpe(backtest(p, momentum(p, 5, 0), 0).returns) > 0.5


def test_mean_reversion_finds_reverting_edge():
    p = simulate_prices(3000, 10, mu=0, phi=-0.2, seed=4)
    assert sharpe(backtest(p, mean_reversion(p, 5), 0).returns) > 0.5


def test_weights_are_dollar_neutral_and_bounded():
    p = simulate_prices(300, 6, seed=5)
    w = momentum(p).iloc[100:]
    assert np.allclose(w.sum(axis=1), 0, atol=1e-9)
    assert (w.abs().sum(axis=1) <= 1 + 1e-9).all()


def test_max_drawdown_known_value():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.0])) == pytest.approx(-0.5)


def test_deflated_sharpe_penalises_many_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    assert deflated_sharpe(r, 100, 1e-4) < probabilistic_sharpe(r)


def test_walk_forward_is_oos_only_and_noise_has_no_edge():
    p = simulate_prices(2000, 8, mu=0, phi=0, seed=6)
    oos, log = walk_forward(p, momentum, {"lookback": [20, 60], "skip": [1, 5]})
    assert len(log) >= 5 and oos.index.is_unique
    assert oos.index.min() >= p.index[504]
    assert sharpe(oos) < 1.0
