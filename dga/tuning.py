"""Validation-based hyper-parameter tuning for DGA.

The last `val_frac` of the (chronologically ordered) training windows is held out. The score is computed on the
fitting part only, every grid point augments the fitting part and is evaluated on the held-out windows, and the
winner is then re-run on the full training set. The held-out windows are never part of the fit that scores them.
"""
import itertools

import numpy as np

from utils.data import split_xy
from utils.metrics import rmse

from .lp import LPConfig
from .pipeline import fit_predict, run_dga
from .scores import target_score

# Paper grid (Section 5): J in 1..5, k_tilde in {1,2,3}, t in {1,5,10}, q in {0.05, 0.10}. The default here is a
# reduced version of it; pass your own grid to widen it.
DEFAULT_GRID = dict(J=[1, 5], k_tilde=[1, 3], t_max=[5.0, 10.0], q=[0.05, 0.10])


def tune(train, grid=None, k=5, val_frac=0.2, cfg=None, seed=0):
    cfg = cfg or LPConfig()
    grid = grid or DEFAULT_GRID
    n_val = int(val_frac * len(train))
    fit_part, val_part = train[:-n_val], train[-n_val:]
    X, y = split_xy(fit_part)
    s = target_score(X, y, cfg=cfg, seed=seed)          # one score per tuning run, shared by the grid
    y_val = val_part[:, -1]

    keys = list(grid)
    best = None
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values), k=k)
        pred, _ = fit_predict(fit_part, val_part, s, params, cfg=cfg, seed=seed)
        err = rmse(y_val, pred)
        if best is None or err < best[0]:
            best = (err, params)
    return best[1], best[0]


def tune_and_run(train, test, grid=None, cfg=None, seed=0):
    params, val_rmse = tune(train, grid=grid, cfg=cfg, seed=seed)
    y_pred, info = run_dga(train, test, params, cfg=cfg, seed=seed)
    return y_pred, {"params": params, "val_rmse": val_rmse, **info}
