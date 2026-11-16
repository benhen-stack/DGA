"""Density-guided augmentation (Algorithm 2 of the paper).

Sampling mechanism
------------------
For fixed z_i, z_j and t the synthetic point is deterministic, so repeating the operation J times with the same
t would return J identical points. Algorithm 2 therefore draws a FRESH, independent step length for every
synthetic point:

    t_{i,l,j} ~ Uniform(t_min, t_max)      i.i.d. over i in X_B, l = 1..k_tilde, j = 1..J

The neighbours are also random (k_tilde of the k nearest, without replacement), so the randomness of Algorithm 2
is exactly {neighbour choice, step length}; both come from one seeded Generator, which makes a run reproducible.
t_max plays the role of the sparsity level: t_max <= 1 stays inside the segment [z_i, z_j], t_max > 1 extrapolates
outward from the dense area.

Direction (Eqs. 10-11), with the paper's convention LOW score = sparse:
    s(x_i) <= s(x_j):  z_new = z_i + (z_i - z_j) * t      (x_i is the sparser point, step away from x_j)
    otherwise       :  z_new = z_i + (z_j - z_i) * t      (step toward the sparser x_j)
For regression the step is applied to z = [x, f(x)], so the function value is synthesised together with x.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def density_guided_augmentation(Z, s, *, q, k, k_tilde, J, t_max, t_min=0.0, rng=None, n_input_cols=None):
    """Algorithm 2.

    Z : (n, m+1) rows z_i = [x_i, f(x_i)]. Neighbours are searched in the input space x (the first
        `n_input_cols` columns; default all but the last).
    s : (n,) score, low = sparse.
    q : fraction of points forming X_B (bottom-q quantile of s).
    k, k_tilde, J : neighbours searched, neighbours used (k_tilde <= k), synthetic points per (i, neighbour).
    t_max, t_min  : t ~ Uniform(t_min, t_max), drawn independently for every synthetic point.
    rng : np.random.Generator or an int seed.

    Returns (Z_new, info) with Z_new of shape (|X_B| * k_tilde * J, m+1).
    """
    rng = np.random.default_rng(rng)
    Z, s = np.asarray(Z, dtype=float), np.asarray(s, dtype=float)
    n = len(Z)
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    if k_tilde > k:
        raise ValueError("k_tilde must not exceed k")
    if k >= n:
        raise ValueError("k must be smaller than the number of points")
    if not t_max > t_min >= 0:
        raise ValueError("need 0 <= t_min < t_max")
    d_in = Z.shape[1] - 1 if n_input_cols is None else n_input_cols

    # X_B: bottom-q quantile of the score
    B = np.flatnonzero(s <= np.quantile(s, q))

    # k nearest neighbours of every x_i in X_B (the point itself is excluded)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[:, :d_in])
    nbrs = nn.kneighbors(Z[B, :d_in], return_distance=False)
    nbrs = np.stack([row[row != i][:k] for row, i in zip(nbrs, B)])          # (|B|, k)

    # RandomChoice: k_tilde of the k neighbours, without replacement
    pick = np.argsort(rng.random(nbrs.shape), axis=1)[:, :k_tilde]
    chosen = np.take_along_axis(nbrs, pick, axis=1)                          # (|B|, k_tilde)

    # one row per synthetic point: (i, j_l) repeated J times
    I = np.repeat(B, k_tilde * J)
    Jn = np.repeat(chosen, J, axis=1).reshape(-1)
    t = rng.uniform(t_min, t_max, size=len(I))                               # fresh t for EVERY point

    away = (s[I] <= s[Jn])[:, None]                                          # Eq. (10) vs Eq. (11)
    step = np.where(away, Z[I] - Z[Jn], Z[Jn] - Z[I])
    Z_new = Z[I] + step * t[:, None]
    return Z_new, {"base_idx": I, "neighbor_idx": Jn, "t": t, "n_boundary": len(B)}
