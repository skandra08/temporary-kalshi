"""DeepAR (Salinas, Flunkert, Gasthaus & Januschowski 2020) with a negative-binomial likelihood, in torch.

Autoregressive LSTM; input at step t is [z_{t-1}/nu, covariates_t, article embedding]; outputs NB mean mu_t = nu*softplus(.)
and shape alpha_t = softplus(.)/sqrt(nu), where nu = 1 + mean(context). Forecasts are ancestral sample paths.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def covariates(idx, t0, T):
    """Calendar covariates for absolute day positions t0..t0+T-1: weekly + yearly harmonics + trend."""
    t = np.arange(t0, t0 + T)
    d = idx[t0:t0 + T]
    dow = np.array([x.dayofweek for x in d]); doy = np.array([x.dayofyear for x in d])
    cols = [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(4 * np.pi * dow / 7), np.cos(4 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), t / 1000.0]
    return np.stack(cols, 1).astype(np.float32)


def nb_nll(z, mu, alpha):
    z, mu, alpha = z.double(), mu.double().clamp_min(1e-6), alpha.double().clamp_min(1e-6)
    r = 1.0 / alpha
    ll = (torch.lgamma(z + r) - torch.lgamma(z + 1) - torch.lgamma(r)
          + r * torch.log(r / (r + mu)) + z * torch.log(mu / (r + mu)))
    return -ll


class DeepAR(nn.Module):
    def __init__(self, n_series: int, ncov: int = 7, hidden: int = 40, layers: int = 2, emb: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_series, emb)
        self.lstm = nn.LSTM(1 + ncov + emb, hidden, layers, batch_first=True, dropout=0.1)
        self.mu = nn.Linear(hidden, 1); self.al = nn.Linear(hidden, 1)

    def step_inputs(self, z_prev, cov, sid, nu):
        e = self.emb(sid)[:, None, :].expand(-1, cov.shape[1], -1)
        return torch.cat([(z_prev / nu[:, None]).unsqueeze(-1), cov, e], -1)

    def heads(self, h, nu):
        mu = nu[:, None] * F.softplus(self.mu(h).squeeze(-1))
        al = F.softplus(self.al(h).squeeze(-1)) / nu.sqrt()[:, None]
        return mu, al

    def forward(self, z, cov, sid, ctx):
        """z: (B, L) values; first `ctx` steps are conditioning. Returns NLL over steps 1..L-1."""
        nu = 1 + z[:, :ctx].mean(1)
        x = self.step_inputs(z[:, :-1], cov[:, 1:], sid, nu)
        h, _ = self.lstm(x)
        mu, al = self.heads(h, nu)
        return nb_nll(z[:, 1:], mu, al).mean()

    @torch.no_grad()
    def sample(self, z_ctx, cov_all, sid, horizon, n_samples=200, seed=0):
        """z_ctx: (B, C) context; cov_all: (B, C+horizon, ncov). Returns (B, n_samples, horizon)."""
        g = torch.Generator().manual_seed(seed)
        B, C = z_ctx.shape
        nu = 1 + z_ctx.mean(1)
        # encode the context
        x = self.step_inputs(z_ctx[:, :-1], cov_all[:, 1:C], sid, nu)
        _, state = self.lstm(x)
        rep = lambda t: t.repeat_interleave(n_samples, 1) if t.dim() == 3 else t
        state = tuple(rep(s) for s in state)
        nu_s = nu.repeat_interleave(n_samples); sid_s = sid.repeat_interleave(n_samples)
        cov_s = cov_all.repeat_interleave(n_samples, 0)
        prev = z_ctx[:, -1].repeat_interleave(n_samples)
        out = []
        for h in range(horizon):
            x = self.step_inputs(prev[:, None], cov_s[:, C + h:C + h + 1], sid_s, nu_s)
            o, state = self.lstm(x, state)
            mu, al = self.heads(o, nu_s)
            mu, al = mu[:, 0].clamp_min(1e-6), al[:, 0].clamp_min(1e-6)
            rate = torch._standard_gamma(torch.full_like(mu, 1.0) / al) * (al * mu) if False else \
                torch.distributions.Gamma(1.0 / al, 1.0 / (al * mu)).sample()
            z = torch.poisson(rate.clamp_max(1e9), generator=g)
            out.append(z); prev = z
        return torch.stack(out, 1).view(B, n_samples, horizon)


def train_deepar(Y, idx, t_end, ctx=90, pred=28, steps=1500, batch=256, lr=1e-3, seed=0, log=print, **kw):
    """Train on windows fully inside [0, t_end). Y: (n, T)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = Y.shape[0]; L = ctx + pred
    model = DeepAR(n, **kw); opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    covs = torch.from_numpy(covariates(idx, 0, Y.shape[1]))
    Yt = torch.from_numpy(Y.astype(np.float32))
    # sample series proportional to sqrt(scale) so quiet series are not ignored (as recommended in the paper)
    w = np.sqrt(1 + Y[:, :t_end].mean(1)); w = w / w.sum()
    for s in range(steps):
        sid = torch.from_numpy(rng.choice(n, batch, p=w))
        st = rng.integers(0, t_end - L, batch)
        z = torch.stack([Yt[i, a:a + L] for i, a in zip(sid.tolist(), st)])
        cv = torch.stack([covs[a:a + L] for a in st])
        loss = model(z, cv, sid, ctx)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1: log(f"  step {s:5d} nll {loss.item():.4f}")
    return model
