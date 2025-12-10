import numpy as np
import pandas as pd


def silverman_bandwidth(x):
    x = np.asarray(x, float)
    n = max(len(x), 2)
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) if n > 1 else 1.0
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * sigma * n ** (-1 / 5)


def kernel_mean_std(x, y, x_grid):
    """Gaussian Nadaraya–Watson."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = silverman_bandwidth(x)
    U = (x[:, None] - x_grid[None, :]) / h
    w = np.exp(-0.5 * U**2)

    wsum = w.sum(axis=0)
    mu = (w * y[:, None]).sum(axis=0) / wsum
    m2 = (w * (y[:, None] ** 2)).sum(axis=0) / wsum
    std = np.sqrt(np.maximum(m2 - mu**2, 0.0))
    return mu, std


def _weighted_quantile(v, w, qs):
    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]
    cw = np.cumsum(w_sorted)
    if cw[-1] <= 0:
        return np.full(len(qs), np.nan)
    cw /= cw[-1]
    return np.interp(qs, cw, v_sorted)


def rank_uniform(x):
    x = np.asarray(x, float)
    n = x.size
    u = (pd.Series(x).rank(method="average").to_numpy() - 0.5) / n
    eps = 1e-6
    return np.clip(u, eps, 1 - eps)


def inv_ecdf(u, x):
    u = np.asarray(u, float)
    return np.quantile(x, u, method="linear")


# def smooth_prob_curve_rankx(x, p, n_grid=200):
#     x = np.asarray(x, float)
#     eps = 1e-5
#     p = np.clip(np.asarray(p, float), eps, 1 - eps)
#     u = rank_uniform(x)
#     u_grid = np.linspace(0.001, 0.999, n_grid)

#     z = np.log(p / (1 - p))  # logit
#     muz, sdz = kernel_mean_std(u, z, x_grid=u_grid)
#     sigmoid = lambda t: 1 / (1 + np.exp(-t))
#     mean_p = sigmoid(muz)
#     lo = sigmoid(muz - sdz)
#     hi = sigmoid(muz + sdz)
#     xg_real = inv_ecdf(u_grid, x)
#     return xg_real, mean_p, lo, hi


def kernel_quantile_band_rankx(x, y, qs=(0.25, 0.75), n_grid=200):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u = rank_uniform(x)
    u_grid = np.linspace(0.001, 0.999, n_grid)
    h = silverman_bandwidth(u)
    U = (u[:, None] - u_grid[None, :]) / h
    W = np.exp(-0.5 * U**2)
    qlo = np.empty_like(u_grid, dtype=float)
    qhi = np.empty_like(u_grid, dtype=float)
    for j in range(u_grid.size):
        wj = W[:, j]
        if wj.sum() <= 0:
            qlo[j] = qhi[j] = np.nan
        else:
            qlo[j], qhi[j] = _weighted_quantile(y, wj, qs)
    xg_real = inv_ecdf(u_grid, x)
    return xg_real, qlo, qhi