"""Vectorised backtest with explicit look-ahead protection and transaction costs."""
from dataclasses import dataclass
import pandas as pd
from .metrics import summary


@dataclass
class BacktestResult:
    returns: pd.Series      # net daily portfolio returns
    gross_returns: pd.Series
    weights: pd.DataFrame   # weights actually held each day
    turnover: pd.Series

    def stats(self):
        out = summary(self.returns)
        out["turnover_ann"] = float(self.turnover.mean() * 252)
        out["gross_sharpe"] = summary(self.gross_returns)["sharpe"]
        return out


def backtest(prices, signal, cost_bps=5.0, max_gross=1.0):
    """Trade `signal` computed at close t using the return from t to t+1.

    The signal is shifted one bar so a weight chosen with information at t earns the
    return of t+1 (no look-ahead). Costs are charged per unit of turnover.
    """
    rets = prices.pct_change().fillna(0.0)
    w = signal.reindex(prices.index).fillna(0.0) * max_gross
    held = w.shift(1).fillna(0.0)                    # decided yesterday, earns today
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    net = gross - turnover * cost_bps / 1e4
    return BacktestResult(net, gross, held, turnover)
