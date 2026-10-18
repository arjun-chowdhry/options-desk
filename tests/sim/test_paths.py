"""Tests for path simulation, against theoretical moments.

GBM: mean and variance of S_T, normal log returns, the discounted price is a martingale at
every date, and the exact scheme gives the same answer on a coarse or fine grid. Heston: the
mean of the variance follows ``theta + (v0 - theta) e^{-kappa t}``, the discounted price is a
martingale, the first-step shocks have correlation rho, variance never goes negative, and with
no vol of vol it collapses to GBM. Plus seed reproducibility and input checks.

Statistical checks use a fixed seed and a tolerance of 4 standard errors, so they are
deterministic and would only fail by chance about once in 15,000 seeds.
"""

import numpy as np
import pytest

from optdesk.sim.paths import gbm_paths, heston_paths, time_grid

S0, MU, SIGMA, T = 100.0, 0.05, 0.25, 1.0
HESTON = {"v0": 0.09, "kappa": 3.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}  # Feller holds


def _std_tol(n):
    """Relative tolerance for a sample std: 4 standard errors, where SE = sigma / sqrt(2n)."""
    return 4 / np.sqrt(2 * n)


def _z(sample, expected):
    """How many standard errors the sample mean is from the expected value."""
    sample = np.asarray(sample)
    return (sample.mean(axis=0) - expected) / (sample.std(axis=0, ddof=1) / np.sqrt(len(sample)))


def test_time_grid():
    np.testing.assert_allclose(time_grid(1.0, 4), [0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        time_grid(0.0, 4)
    with pytest.raises(ValueError):
        time_grid(1.0, 0)


def test_gbm_terminal_mean_and_variance_match_theory():
    S_T = gbm_paths(S0, MU, SIGMA, T, 1, 400_000, rng=1).S[:, -1]
    mean = S0 * np.exp(MU * T)
    var = S0**2 * np.exp(2 * MU * T) * (np.exp(SIGMA**2 * T) - 1)
    assert abs(_z(S_T, mean)) < 4
    assert S_T.var() == pytest.approx(var, rel=0.02)


def test_gbm_log_returns_are_normal_with_the_right_moments():
    p = gbm_paths(S0, MU, SIGMA, T, 1, 400_000, rng=2)
    x = np.log(p.S[:, -1] / S0)
    assert x.mean() == pytest.approx((MU - 0.5 * SIGMA**2) * T, abs=4 * SIGMA / np.sqrt(4e5))
    assert x.std() == pytest.approx(SIGMA * np.sqrt(T), rel=_std_tol(len(x)))
    standardized = (x - x.mean()) / x.std()
    assert abs(np.mean(standardized**3)) < 0.02  # skew of a normal
    assert np.mean(standardized**4) == pytest.approx(3.0, abs=0.05)  # kurtosis of a normal


def test_gbm_discounted_price_is_a_martingale_at_every_date():
    r, q = 0.04, 0.01
    p = gbm_paths(S0, r - q, SIGMA, T, 20, 100_000, rng=3)
    discounted = p.S * np.exp(-(r - q) * p.times)  # stock with dividends reinvested, in PV
    assert np.all(np.abs(_z(discounted[:, 1:], S0)) < 4)


def test_gbm_is_exact_so_grid_size_does_not_change_the_distribution():
    coarse = gbm_paths(S0, MU, SIGMA, T, 1, 200_000, rng=4).S[:, -1]
    fine = gbm_paths(S0, MU, SIGMA, T, 250, 200_000, rng=5).S[:, -1]
    se = np.sqrt(coarse.var() / len(coarse) + fine.var() / len(fine))
    assert abs(coarse.mean() - fine.mean()) < 4 * se
    assert np.quantile(coarse, 0.01) == pytest.approx(np.quantile(fine, 0.01), rel=0.01)


def test_gbm_variance_view_is_constant_sigma_squared():
    p = gbm_paths(S0, MU, SIGMA, T, 10, 5, rng=6)
    assert p.S.shape == p.v.shape == (5, 11)
    assert np.all(p.v == SIGMA**2)
    assert np.all(p.S[:, 0] == S0)


def test_seed_reproduces_paths_and_different_seeds_differ():
    a = gbm_paths(S0, MU, SIGMA, T, 10, 50, rng=7)
    b = gbm_paths(S0, MU, SIGMA, T, 10, 50, rng=np.random.default_rng(7))
    c = gbm_paths(S0, MU, SIGMA, T, 10, 50, rng=8)
    np.testing.assert_array_equal(a.S, b.S)
    assert not np.allclose(a.S, c.S)
    h1 = heston_paths(S0, mu=MU, T=T, n_steps=10, n_paths=50, rng=9, **HESTON)
    h2 = heston_paths(S0, mu=MU, T=T, n_steps=10, n_paths=50, rng=9, **HESTON)
    np.testing.assert_array_equal(h1.S, h2.S)
    np.testing.assert_array_equal(h1.v, h2.v)


def test_heston_mean_variance_follows_mean_reversion():
    """E[v_t] = theta + (v0 - theta) e^{-kappa t}, decaying from 0.09 towards 0.04."""
    p = heston_paths(S0, mu=MU, T=T, n_steps=250, n_paths=50_000, rng=10, **HESTON)
    expected = HESTON["theta"] + (HESTON["v0"] - HESTON["theta"]) * np.exp(
        -HESTON["kappa"] * p.times
    )
    for i in (25, 50, 125, 250):
        # 4 standard errors, plus 0.5% for the small bias of the Euler scheme
        se = p.v[:, i].std() / np.sqrt(p.v.shape[0])
        assert abs(p.v[:, i].mean() - expected[i]) < 4 * se + 0.005 * expected[i]


def test_heston_discounted_price_is_a_martingale():
    r, q = 0.04, 0.01
    p = heston_paths(S0, mu=r - q, T=T, n_steps=100, n_paths=100_000, rng=11, **HESTON)
    discounted = p.S[:, [10, 50, 100]] * np.exp(-(r - q) * p.times[[10, 50, 100]])
    assert np.all(np.abs(_z(discounted, S0)) < 4)


def test_heston_price_and_variance_shocks_have_correlation_rho():
    dt = 1 / 252
    p = heston_paths(S0, mu=MU, T=dt, n_steps=1, n_paths=200_000, rng=12, **HESTON)
    dlog_s = np.log(p.S[:, 1] / S0)
    dv = p.v[:, 1] - p.v[:, 0]  # v0 is far from zero, so truncation never bites on step 1
    assert np.corrcoef(dlog_s, dv)[0, 1] == pytest.approx(HESTON["rho"], abs=0.01)


def test_heston_variance_never_negative_even_when_feller_fails():
    params = {**HESTON, "kappa": 1.0, "xi": 1.0}  # 2 kappa theta = 0.08 < xi^2 = 1
    p = heston_paths(S0, mu=MU, T=T, n_steps=100, n_paths=5_000, rng=13, **params)
    assert np.all(p.v >= 0)
    assert np.mean(p.v == 0) > 0.01  # the raw Euler variance did hit zero: truncation was needed
    assert np.all(np.isfinite(p.S)) and np.all(p.S > 0)


def test_heston_with_no_vol_of_vol_is_gbm():
    theta = 0.0625  # 25% vol
    params = {"v0": theta, "kappa": 2.0, "theta": theta, "xi": 0.0, "rho": -0.5}
    p = heston_paths(S0, mu=MU, T=T, n_steps=50, n_paths=200_000, rng=14, **params)
    assert np.all(p.v == theta)
    x = np.log(p.S[:, -1] / S0)
    assert abs(_z(x, (MU - 0.5 * theta) * T)) < 4
    assert x.std() == pytest.approx(np.sqrt(theta * T), rel=_std_tol(len(x)))


@pytest.mark.parametrize(
    "kwargs",
    [{"S0": 0.0}, {"v0": -0.01}, {"rho": 1.5}, {"xi": -0.1}],
)
def test_heston_rejects_bad_inputs(kwargs):
    args = {"S0": S0, "mu": MU, "T": T, "n_steps": 5, "n_paths": 5, **HESTON, **kwargs}
    with pytest.raises(ValueError):
        heston_paths(**args)


def test_gbm_rejects_bad_inputs():
    with pytest.raises(ValueError):
        gbm_paths(-1.0, MU, SIGMA, T, 5, 5)
    with pytest.raises(ValueError):
        gbm_paths(S0, MU, -0.1, T, 5, 5)
