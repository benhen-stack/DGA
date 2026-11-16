"""Multi-scale Laplacian Pyramid (LP) with chunked kernels.

Kernels: row-normalised Gaussian exp(-d^2 / sigma^2), sigma halved at every level, zero diagonal on the
training set (leave-one-out). Nothing of size N x N is stored: every kernel block is rebuilt from a GEMM over
a chunk of rows, so memory is O(chunk * N).

The model approximates a target Y from inputs X.
    * Data score   : Y = X   (self reconstruction, f(X) = X)
    * Target score : Y = f(X) (regression function, e.g. the next value of a window)
When Y is omitted it defaults to X, which is the original self-reconstruction behaviour.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LPConfig:
    levels: int = 21                 # number of pyramid levels (sigma_0 ... sigma_0 / 2^(levels-1))
    coarse_level: int = 0            # level whose approximation is stored as F_coarse
    chunk: int = 2048                # rows per kernel block
    eps_sample: int = 10_000         # rows used to estimate sigma_0
    eps_factor: float = 10.0         # sigma_0 = eps_factor * max_i min_{j != i} ||x_i - x_j||
    # Rows whose raw kernel mass is below this are zeroed (sklearn's normalize() treats row sums below
    # 10*eps(float64) as zero). Set to 0 for the exact mathematical limit.
    min_row_mass: float = field(default_factory=lambda: 10 * float(np.finfo(np.float64).eps))


class MultiScaleLP:
    def __init__(self, cfg=None, seed=0):
        self.cfg, self.seed = cfg or LPConfig(), seed

    # ---- helpers
    def _d2(self, q, qsq):
        return np.maximum(qsq[:, None] + self.Xsq[None] - 2.0 * (q @ self.X.T), 0.0)

    def _kern(self, D2, sigma):
        """Row-normalised Gaussian exp(-d^2/sigma^2), computed with a row-min shift so float32 never underflows.
        Rows whose raw kernel mass is below min_row_mass are zeroed (see LPConfig)."""
        z = D2 * np.float32(1.0 / sigma ** 2)
        zmin = z.min(1, keepdims=True)
        K = np.exp(zmin - z)
        s = K.sum(1, keepdims=True)
        K /= s
        cut = self.cfg.min_row_mass
        if cut > 0:
            K[((np.log(s) - zmin) < np.log(cut))[:, 0]] = 0.0     # raw mass = exp(-zmin) * s
        return K

    def _epsilon(self):
        N, rng = len(self.X), np.random.default_rng(self.seed)
        idx = rng.choice(N, min(N, self.cfg.eps_sample), replace=False)
        mins = []
        for a in range(0, len(idx), self.cfg.chunk):
            ii = idx[a:a + self.cfg.chunk]
            D2 = self._d2(self.X[ii], self.Xsq[ii])
            D2[np.arange(len(ii)), ii] = np.inf
            mins.append(D2.min(1))
        return float(np.sqrt(np.concatenate(mins).max()) * self.cfg.eps_factor)

    # ---- fit
    def fit(self, X, Y=None):
        """Fit the pyramid. X: (N, m) inputs the kernel is built on. Y: (N, d) target (default: X)."""
        c = self.cfg
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.Xsq = (self.X ** 2).sum(1)
        self.self_reconstruction = Y is None
        Y = self.X if Y is None else np.asarray(Y, dtype=np.float32)
        self.Y = np.ascontiguousarray(Y.reshape(len(Y), -1))
        N = len(self.X)
        self.sigmas = [self._epsilon() / 2 ** l for l in range(c.levels)]
        D, F = [], None
        self.F_coarse = None
        for l in range(c.levels):
            V = self.Y if l == 0 else D[-1]
            A = np.empty_like(self.Y)
            for a in range(0, N, c.chunk):
                b = min(a + c.chunk, N)
                D2 = self._d2(self.X[a:b], self.Xsq[a:b])
                D2[np.arange(b - a), np.arange(a, b)] = np.inf      # leave-one-out
                A[a:b] = self._kern(D2, self.sigmas[l]) @ V
            F = A if F is None else F + A
            D.append(self.Y - F)
            if l == c.coarse_level:
                self.F_coarse = F.copy()
        self.D = D
        self.S_train = np.stack([np.linalg.norm(d, axis=1) for d in D], 1)     # per-row residual norm per level
        self.mu_s, self.sd_s = self.S_train.mean(0), self.S_train.std(0) + 1e-8
        return self

    @property
    def level_errors(self):
        """Leave-one-out Frobenius error ||D_l|| at every level."""
        return np.sqrt((self.S_train ** 2).sum(0))

    @property
    def best_level(self):
        """Level with the smallest leave-one-out error (the 'optimal iteration' of the pyramid)."""
        return int(np.argmin(self.level_errors))

    # ---- out-of-sample extension
    def predict(self, Q, level=None):
        """Approximation of Y at the query points Q using levels 0..level (default: best_level)."""
        c = self.cfg
        level = self.best_level if level is None else level
        out = []
        for a in range(0, len(Q), c.chunk):
            q = np.ascontiguousarray(Q[a:a + c.chunk], dtype=np.float32)
            D2 = self._d2(q, (q ** 2).sum(1))
            F = self._kern(D2, self.sigmas[0]) @ self.Y
            for l in range(1, level + 1):
                F = F + self._kern(D2, self.sigmas[l]) @ self.D[l - 1]
            out.append(F)
        return np.concatenate(out)

    def transform(self, Q):
        """Self-reconstruction only. Returns (raw signatures (n, levels), coarse approximations (n, m))."""
        if not self.self_reconstruction:
            raise ValueError("transform() is defined for self-reconstruction (Y = X); use predict() instead.")
        c, S, Fc = self.cfg, [], []
        for a in range(0, len(Q), c.chunk):
            q = np.ascontiguousarray(Q[a:a + c.chunk], dtype=np.float32)
            D2 = self._d2(q, (q ** 2).sum(1))
            F = self._kern(D2, self.sigmas[0]) @ self.X
            s = [np.linalg.norm(q - F, axis=1)]
            fc = F if c.coarse_level == 0 else None
            for l in range(1, c.levels):
                F = F + self._kern(D2, self.sigmas[l]) @ self.D[l - 1]
                s.append(np.linalg.norm(q - F, axis=1))
                if l == c.coarse_level:
                    fc = F
            S.append(np.stack(s, 1))
            Fc.append(fc)
        return np.concatenate(S), np.concatenate(Fc)

    def std(self, S):
        return ((S - self.mu_s) / self.sd_s).astype(np.float32)
