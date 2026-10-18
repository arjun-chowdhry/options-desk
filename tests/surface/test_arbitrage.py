"""Tests for the static-arbitrage checks.

- ``g(k)`` is checked against an independent route to the density: Breeden-Litzenberger,
  the second strike-derivative of Black-Scholes call prices, by finite differences.
- Known answers: Axel Vogt's SVI slice (Gatheral & Jacquier 2014) has butterfly arbitrage
  near k = 0.9 and must be flagged; a flat smile has g = 1 everywhere; the synthetic SSVI
  surface is arbitrage-free by construction and must pass.
- Calendar: hand-built slices that cross at a known k are flagged exactly there.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.pricing.black_scholes import bs_price
from optdesk.surface.arbitrage import (
    butterfly_check,
    calendar_check,
    check_surface,
    g_function,
)
from optdesk.surface.svi import PARAM_NAMES, SVIParams
from optdesk.surface.synthetic import true_slices

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)  # T = 1


@pytest.mark.parametrize(
    "params",
    [SVIParams(0.01, 0.1, -0.5, 0.02, 0.15), SVIParams(0.002, 0.03, -0.8, 0.03, 0.05), VOGT],
    ids=["smooth", "sharp_equity", "vogt"],
)
def test_density_formula_matches_breeden_litzenberger(params):
    """g/sqrt(2 pi w) exp(-d2^2/2) equals K d2C/dK2 (density per unit of k), F = 1, r = 0."""
    k = np.linspace(-0.5, 1.2, 35)
    K = np.exp(k)

    def call(strike):
        vol = params.implied_vol(np.log(strike), 1.0)
        return bs_price(1.0, strike, 1.0, 0.0, vol)

    h = 1e-4 * K
    density_K = (call(K + h) - 2 * call(K) + call(K - h)) / h**2
    w = params.total_variance(k)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    formula = g_function(k, params) / np.sqrt(2 * np.pi * w) * np.exp(-(d2**2) / 2)
    np.testing.assert_allclose(formula, K * density_K, rtol=1e-4, atol=1e-6)


def test_vogt_example_is_flagged():
    report = butterfly_check(VOGT)
    assert not report.ok
    assert report.lee_ok  # wings are fine; the problem is in the body of the smile
    assert report.min_g < -0.03
    assert 0.8 < report.k_at_min < 0.95
    assert len(report.bad_ranges) == 1
    start, end = report.bad_ranges[0]
    assert 0.6 < start < 0.7 and 1.2 < end < 1.3
    assert np.all(g_function(np.linspace(start, end, 50), VOGT) < 0)


def test_flat_smile_has_g_equal_one():
    flat = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)  # w' = w'' = 0
    np.testing.assert_allclose(g_function(np.linspace(-1, 1, 11), flat) - 1, 0, atol=0)
    assert butterfly_check(flat).ok


def test_true_synthetic_surface_is_arbitrage_free():
    T = np.array([0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    slices = true_slices(T)
    for params in slices:
        report = butterfly_check(params)
        assert report.ok and report.min_g > 0 and report.bad_ranges == []
    assert calendar_check(list(zip(T, slices, strict=True))).ok


def test_lee_violation_flagged():
    steep = SVIParams(a=0.01, b=1.5, rho=-0.6, m=0.0, sigma=0.1)  # b (1 + |rho|) = 2.4
    report = butterfly_check(steep)
    assert not report.lee_ok and not report.ok


def test_negative_variance_flagged():
    report = butterfly_check(SVIParams(a=-0.02, b=0.1, rho=0.0, m=0.0, sigma=0.1))
    assert not report.ok
    assert report.min_g == -np.inf  # undefined density where w <= 0
    # w(k) = -0.02 + 0.1 sqrt(k^2 + 0.01) < 0 for |k| < sqrt(0.03) = 0.173: one flagged range.
    edge = np.sqrt(0.03)
    assert any(abs(lo + edge) < 2e-3 and abs(hi - edge) < 2e-3 for lo, hi in report.bad_ranges)
    # Where w is small but positive the density is negative too (the (w')^2 / 4w term blows
    # up), so there are more flagged ranges either side.
    assert len(report.bad_ranges) == 3


def test_calendar_flags_crossed_slices_where_they_cross():
    """Same b, m, sigma; only a and rho differ, so w2 - w1 = 0.002 + 0.02 k: crosses at -0.1."""
    near = SVIParams(a=0.010, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    far = SVIParams(a=0.012, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    report = calendar_check([(0.5, far), (0.25, near)])  # unsorted on purpose
    assert not report.ok
    row = report.pairs.iloc[0]
    assert (row["T1"], row["T2"]) == (0.25, 0.5)
    (start, end), *_ = row["bad_ranges"]
    assert start == pytest.approx(-1.5)  # left end of the grid
    assert end == pytest.approx(-0.1, abs=1e-3)
    assert row["min_gap"] == pytest.approx(0.002 + 0.02 * -1.5)
    assert row["k_at_min"] == pytest.approx(-1.5)


def test_calendar_ok_and_single_slice():
    p = SVIParams(0.01, 0.1, -0.5, 0.0, 0.1)
    bigger = SVIParams(0.02, 0.1, -0.5, 0.0, 0.1)  # same shape, more variance everywhere
    assert calendar_check([(0.25, p), (0.5, bigger)]).ok
    assert not calendar_check([(0.25, bigger), (0.5, p)]).ok
    assert calendar_check([(0.25, p)]).ok and calendar_check([(0.25, p)]).pairs.empty


def _table(slices):
    """A fits-style table from (T, params) pairs, shuffled: order shouldn't matter."""
    rows = [{"expiry": i, "T": T, **vars(p)} for i, (T, p) in enumerate(slices)]
    return pd.DataFrame(rows).sample(frac=1, random_state=0)


def test_check_surface_flags_calendar_on_the_later_slice():
    a_values = [0.01, 0.02, 0.015]  # third slice dips under the second
    out = check_surface(
        _table(
            [(0.25 * (i + 1), SVIParams(a, 0.1, -0.5, 0.0, 0.1)) for i, a in enumerate(a_values)]
        )
    )
    assert list(out["T"]) == [0.25, 0.5, 0.75]
    assert list(out["calendar_ok"]) == [True, True, False]
    assert out["butterfly_ok"].all()
    assert set(PARAM_NAMES) <= set(out.columns) and "min_g" in out and "lee_ok" in out


def test_check_surface_flags_butterfly_on_the_bad_slice():
    high = SVIParams(0.3, 0.1, -0.5, 0.0, 0.1)  # above Vogt everywhere: calendar is fine
    out = check_surface(_table([(1.0, VOGT), (2.0, high)]))
    assert list(out["butterfly_ok"]) == [False, True]
    assert out["calendar_ok"].all()
    assert out.loc[0, "min_g"] < -0.03


def test_fitted_synthetic_surface_passes(market_fits):
    out = check_surface(market_fits)
    assert out["butterfly_ok"].all() and out["calendar_ok"].all() and out["lee_ok"].all()
