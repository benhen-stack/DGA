"""DGA end to end: score -> density-guided augmentation -> LP forecaster."""
import numpy as np

from utils.data import split_xy

from .augment import density_guided_augmentation
from .lp import LPConfig, MultiScaleLP
from .scores import target_score

DEFAULT_PARAMS = dict(q=0.05, k=5, k_tilde=3, J=5, t_max=5.0)


def fit_predict(train, test, score, params, cfg=None, seed=0, decimals=2):
    """Augment `train` with Algorithm 2 guided by `score`, fit the LP forecaster on it, predict `test`.

    train, test : window matrices (n, window); the last column is the target.
    score       : (len(train),) density score of the training rows (low = sparse).
    """
    cfg = cfg or LPConfig()
    p = {**DEFAULT_PARAMS, **params}
    rng = np.random.default_rng(seed)
    Z_new, info = density_guided_augmentation(
        train, score, q=p["q"], k=p["k"], k_tilde=p["k_tilde"], J=p["J"], t_max=p["t_max"], rng=rng)
    if decimals is not None:
        Z_new = Z_new.round(decimals)
    Z = np.vstack([train, Z_new])
    X, y = split_xy(Z)
    lp = MultiScaleLP(cfg, seed=seed).fit(X, y.reshape(-1, 1))
    y_pred = lp.predict(split_xy(test)[0])[:, 0]
    return y_pred, {"n_synthetic": len(Z_new), "best_level": lp.best_level, **info}


def run_dga(train, test, params=None, cfg=None, seed=0):
    """Target-score the training windows, then augment and forecast."""
    cfg = cfg or LPConfig()
    X, y = split_xy(train)
    s = target_score(X, y, cfg=cfg, seed=seed)
    return fit_predict(train, test, s, params or {}, cfg=cfg, seed=seed)
