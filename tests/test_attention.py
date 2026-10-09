import numpy as np, pandas as pd, torch
from scipy.stats import nbinom
from attention.baselines import QS, empirical_recent, empirical_weekday, lognormal_weekday
from attention.deepar import DeepAR, covariates, nb_nll, train_deepar


def test_nb_nll_matches_scipy():
    z = torch.tensor([0., 3., 17., 400.]); mu = torch.tensor([2., 3., 20., 350.]); al = torch.tensor([.5, .2, .1, .05])
    r = 1 / al.numpy(); p = r / (r + mu.numpy())
    ref = -nbinom.logpmf(z.numpy(), r, p)
    assert np.allclose(nb_nll(z, mu, al).numpy(), ref, atol=1e-4)


def test_baselines_shapes_and_weekly_pattern():
    t = np.arange(120); ctx = (100 + 50 * (t % 7 == 5))[None, :].astype(float)
    for f in (empirical_weekday, lognormal_weekday):
        q = f(ctx, 28)
        assert q.shape == (1, len(QS), 28) and (np.diff(q, axis=1) >= -1e-9).all()
    q = empirical_weekday(ctx, 28)[0, 9]            # median
    assert np.allclose(q, [150 if (120 + h) % 7 == 5 else 100 for h in range(28)])
    assert empirical_recent(ctx, 28).shape == (1, len(QS), 28)


def test_deepar_learns_weekly_level_and_samples_are_counts():
    idx = pd.date_range("2023-01-01", periods=300); rng = np.random.default_rng(0)
    base = 40 + 30 * (np.arange(300) % 7 == 5)
    Y = rng.poisson(np.tile(base, (4, 1))).astype(float)
    m = train_deepar(Y, idx, 240, ctx=60, pred=14, steps=300, batch=64, log=lambda s: None); m.eval()
    cov = torch.from_numpy(covariates(idx, 0, 300))[None].repeat(4, 1, 1)
    s = m.sample(torch.from_numpy(Y[:, 180:240].astype(np.float32)), cov[:, 180:254], torch.arange(4), 14, n_samples=100)
    assert (s >= 0).all() and torch.equal(s, s.round())
    sat = [h for h in range(14) if (240 + h) % 7 == 5]; other = [h for h in range(14) if (240 + h) % 7 != 5]
    assert s[:, :, sat].mean() > 1.2 * s[:, :, other].mean()
