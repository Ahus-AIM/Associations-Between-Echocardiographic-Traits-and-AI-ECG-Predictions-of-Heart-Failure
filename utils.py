import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def _paths_to_path_objects(value):
    if isinstance(value, dict):
        return {key: _paths_to_path_objects(item) for key, item in value.items()}
    if isinstance(value, str):
        return Path(value).expanduser()
    return value


def load_paths(config_path="paths.yaml"):
    """Load machine-local data pointers from a gitignored YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. Create it from paths.example.yaml and point it "
            "to the local datasets before running the notebooks."
        )

    with config_path.open("r") as f:
        paths = yaml.safe_load(f) or {}

    return _paths_to_path_objects(paths)


def silverman_bandwidth(x, const=0.9):
    x = np.asarray(x, float)
    n = max(len(x), 2)
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) if n > 1 else 1.0
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    return const * sigma * n ** (-1 / 5)


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


def kernel_mean_quantile_band_rankx(x, y, qs, n_grid, const=0.9, eps=1e-12):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    u = rank_uniform(x)
    u_grid = np.linspace(0.001, 0.999, n_grid)

    h = silverman_bandwidth(u, const=const)
    h = max(h, eps)

    U = (u[:, None] - u_grid[None, :]) / h
    W = np.exp(-0.5 * U**2)  # Gaussian kernel (unnormalized ok; we normalize per grid)

    qmean = np.empty(u_grid.size, dtype=float)
    qmedian = np.empty(u_grid.size, dtype=float)
    qlo = np.full(u_grid.size, np.nan, dtype=float)
    qhi = np.full(u_grid.size, np.nan, dtype=float)

    qs = np.asarray(qs, float)
    need_median = not np.any(np.isclose(qs, 0.5))
    qs_all = np.unique(np.concatenate([qs, [0.5]]) if need_median else qs)

    med_idx = int(np.where(np.isclose(qs_all, 0.5))[0][0])

    for j in range(u_grid.size):
        wj = W[:, j]
        sw = wj.sum()
        if sw <= 0:
            qmean[j] = np.nan
            qmedian[j] = np.nan
            continue

        qmean[j] = np.dot(wj, y) / sw

        q_all = _weighted_quantile(y, wj, qs_all)
        qmedian[j] = q_all[med_idx]

        if qs.size >= 1:
            qlo[j] = _weighted_quantile(y, wj, [qs[0]])[0]
        if qs.size >= 2:
            qhi[j] = _weighted_quantile(y, wj, [qs[1]])[0]

    xg_real = inv_ecdf(u_grid, x)
    return xg_real, qmean, qmedian, qlo, qhi
