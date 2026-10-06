"""Walk-forward validation: choose parameters in-sample, trade them strictly out-of-sample."""
import itertools
import pandas as pd
from .backtest import backtest
from .metrics import sharpe


def walk_forward(prices, signal_fn, param_grid, train=504, test=126, cost_bps=5.0):
    """Roll a train/test window. For each fold pick the best in-sample Sharpe params,
    then trade the next `test` days with them. Returns (oos_returns, fold_log)."""
    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    # Signals depend only on the past, so computing them on the full history is leak-free.
    results = {i: backtest(prices, signal_fn(prices, **c), cost_bps).returns
               for i, c in enumerate(combos)}
    oos, log = [], []
    for start in range(train, len(prices) - 1, test):
        tr = slice(start - train, start)
        te = slice(start, min(start + test, len(prices)))
        best = max(results, key=lambda i: sharpe(results[i].iloc[tr]))
        oos.append(results[best].iloc[te])
        log.append({"test_start": prices.index[start], "params": combos[best],
                    "is_sharpe": sharpe(results[best].iloc[tr]),
                    "oos_sharpe": sharpe(results[best].iloc[te])})
    return pd.concat(oos), pd.DataFrame(log)
