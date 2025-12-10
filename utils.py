import numpy as np
import pandas as pd


def silverman_bandwidth(x):
    x = np.asarray(x, float)
    n = max(len(x), 2)
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) if n > 1 else 1.0
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * sigma * n ** (-1 / 5)


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