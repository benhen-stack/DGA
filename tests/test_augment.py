import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dga import LPConfig, MultiScaleLP, density_guided_augmentation, target_score   # noqa: E402


@pytest.fixture
def cloud():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] ** 2 + 0.1 * rng.normal(size=120)
    Z = np.column_stack([X, y])
    s = np.linalg.norm(X, axis=1)[::-1]          # arbitrary score, low = sparse
    return Z, s


KW = dict(q=0.1, k=5, k_tilde=2, J=4, t_max=3.0)


def test_output_size(cloud):
    Z, s = cloud
    Z_new, info = density_guided_augmentation(Z, s, rng=0, **KW)
    assert len(Z_new) == info["n_boundary"] * KW["k_tilde"] * KW["J"]
    assert Z_new.shape[1] == Z.shape[1]


def test_J_copies_are_not_identical(cloud):
    """With a fixed t the J repetitions of a pair would coincide; t is drawn per point, so they must differ."""
    Z, s = cloud
    Z_new, info = density_guided_augmentation(Z, s, rng=0, **KW)
    I, Jn = info["base_idx"], info["neighbor_idx"]
    for i, j in set(zip(I, Jn)):
        group = Z_new[(I == i) & (Jn == j)]
        assert len(group) == KW["J"]
        assert len(np.unique(group.round(12), axis=0)) == KW["J"]


def test_t_is_drawn_per_point_in_range(cloud):
    Z, s = cloud
    _, info = density_guided_augmentation(Z, s, rng=0, **KW)
    t = info["t"]
    assert len(np.unique(t)) == len(t)
    assert t.min() >= 0 and t.max() <= KW["t_max"]


def test_reproducible_and_seed_dependent(cloud):
    Z, s = cloud
    a, _ = density_guided_augmentation(Z, s, rng=7, **KW)
    b, _ = density_guided_augmentation(Z, s, rng=7, **KW)
    c, _ = density_guided_augmentation(Z, s, rng=8, **KW)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_direction_rule(cloud):
    """Eq. 10 (away from the neighbour) when s_i <= s_j, Eq. 11 (toward it) otherwise."""
    Z, _ = cloud
    s = np.random.default_rng(3).random(len(Z))      # unstructured score so both branches occur
    Z_new, info = density_guided_augmentation(Z, s, rng=0, **{**KW, "q": 0.5})
    I, Jn, t = info["base_idx"], info["neighbor_idx"], info["t"]
    away = s[I] <= s[Jn]
    expected = np.where(away[:, None], Z[I] + (Z[I] - Z[Jn]) * t[:, None], Z[I] + (Z[Jn] - Z[I]) * t[:, None])
    np.testing.assert_allclose(Z_new, expected)
    assert away.any() and (~away).any()


def test_boundary_set_is_bottom_quantile(cloud):
    Z, s = cloud
    _, info = density_guided_augmentation(Z, s, rng=0, **KW)
    assert set(info["base_idx"]) == set(np.flatnonzero(s <= np.quantile(s, KW["q"])))


def test_invalid_arguments(cloud):
    Z, s = cloud
    with pytest.raises(ValueError):
        density_guided_augmentation(Z, s, **{**KW, "k_tilde": 9})
    with pytest.raises(ValueError):
        density_guided_augmentation(Z, s, **{**KW, "t_max": 0.0})


def test_lp_regression_and_self_reconstruction(cloud):
    Z, _ = cloud
    X, y = Z[:, :-1], Z[:, -1]
    cfg = LPConfig(levels=8, eps_factor=2.0)
    reg = MultiScaleLP(cfg).fit(X, y.reshape(-1, 1))
    assert reg.predict(X[:5]).shape == (5, 1)
    self_rec = MultiScaleLP(cfg).fit(X)
    S, Fc = self_rec.transform(X[:5])
    assert S.shape == (5, 8) and Fc.shape == (5, X.shape[1])
    with pytest.raises(ValueError):
        reg.transform(X)


def test_target_score_shape(cloud):
    Z, _ = cloud
    s = target_score(Z[:, :-1], Z[:, -1], cfg=LPConfig(levels=6, eps_factor=2.0))
    assert s.shape == (len(Z),) and np.isfinite(s).all()
