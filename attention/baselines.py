"""Statistical probabilistic baselines on a (B, C) context -> quantile forecasts (B, Q, H)."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

QS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _weekday_stack(ctx, H, k=8):
    """For each horizon day h, the last k context values on the same weekday. ctx last column = day before forecast start."""
    B, C = ctx.shape
    out = np.empty((B, k, H))
    for h in range(H):
        pos = C + h
        idxs = [pos - 7 * j for j in range(1, 60) if pos - 7 * j < C][:k]
        out[:, :, h] = ctx[:, idxs]
    return out


def empirical_weekday(ctx, H, k=8):
    st = _weekday_stack(ctx, H, k)
    return np.quantile(st, QS, axis=1).transpose(1, 0, 2)


def lognormal_weekday(ctx, H, k=8):
    st = np.log1p(_weekday_stack(ctx, H, k))
    m, s = st.mean(1), st.std(1, ddof=1) + 0.05
    q = np.expm1(m[:, None, :] + s[:, None, :] * norm.ppf(QS)[None, :, None])
    return np.clip(q, 0, None)


def empirical_recent(ctx, H, k=28):
    q = np.quantile(ctx[:, -k:], QS, axis=1).T
    return np.repeat(q[:, :, None], H, 2)
