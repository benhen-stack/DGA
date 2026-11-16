"""Target Score s(x) for DGA (Section 3.2 of the paper).

The multi-scale LP regresses the target function y = f(X) on the training set. Its leave-one-out residuals are
turned into one error vector per point and ranked by an Isolation Forest. The score therefore highlights regions
of X where f is hard to approximate.

Convention: LOW score = hard / sparse region. The augmentation set X_B is the bottom-q quantile of s.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from .lp import LPConfig, MultiScaleLP


def error_vectors(D, level):
    """Per-point error vectors: for each level 0..level, the sum over output dimensions of |z-score| of the residual.

    D: list of (n, d) residual matrices of the pyramid. Returns (n, level + 1).
    """
    cols = []
    for l in range(level + 1):
        d = np.asarray(D[l], dtype=np.float64)
        sd = d.std(0)
        z = np.abs(d - d.mean(0)) / np.where(sd > 0, sd, 1.0)      # constant columns contribute 0
        cols.append(z.sum(1))
    return np.stack(cols, 1)


def target_score(X, y, cfg=None, seed=0, n_estimators=500):
    """Return s(x) for every row of X (low = hard to approximate).

    X: (n, m) inputs. y: (n,) target.
    """
    cfg = cfg or LPConfig()
    lp = MultiScaleLP(cfg, seed=seed).fit(X, np.asarray(y, dtype=np.float32).reshape(-1, 1))
    E = error_vectors(lp.D, lp.best_level)
    forest = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(E)
    return forest.decision_function(E)
