"""Tests for the swappable surface sources, against answers known exactly.

- The analytic SSVI surface equals the synthetic market's raw SVI slices at every expiry.
- A surface built from SVI slices reproduces each slice exactly, interpolates total variance
  linearly in T, holds each strike's vol fixed outside the fitted range, and stays
  calendar-arbitrage-free (w non-decreasing in T) when the slices are.
- Every source satisfies the ``VolSurface`` protocol.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.surface.source import (
    FlatSurface,
    SSVISurface,
    SVISurface,
    VolSurface,
    fitted_surface,
    synthetic_surface,
)
from optdesk.surface.svi import PARAM_NAMES
from optdesk.surface.synthetic import atm_vol, true_slices

K_GRID = np.linspace(-0.4, 0.3, 29)
T_SLICES = np.array([0.05, 0.1, 0.25, 0.5, 1.0])


@pytest.fixture(scope="module")
def truth_table():
    rows = [
        {"T": T, **dict(zip(PARAM_NAMES, vars(p).values(), strict=True))}
        for T, p in zip(T_SLICES, true_slices(T_SLICES), strict=True)
    ]
    return pd.DataFrame(rows)


def test_every_source_is_a_vol_surface(truth_table):
    for s in (FlatSurface(0.2), synthetic_surface(), fitted_surface(truth_table)):
        assert isinstance(s, VolSurface)


def test_flat_surface_broadcasts():
    vol = FlatSurface(0.2).implied_vol(K_GRID[:, None], T_SLICES[None, :])
    assert vol.shape == (len(K_GRID), len(T_SLICES))
    assert np.all(vol == 0.2)
    assert FlatSurface(0.2).implied_vol(0.0, 1.0) == 0.2


@pytest.mark.parametrize("T", T_SLICES)
def test_ssvi_surface_matches_synthetic_slices(T):
    expected = true_slices(T)[0].implied_vol(K_GRID, T)
    np.testing.assert_allclose(SSVISurface().implied_vol(K_GRID, T), expected, rtol=1e-12)


def test_ssvi_atm_vol_is_the_term_structure():
    T = np.array([0.02, 0.3, 2.0])
    np.testing.assert_allclose(SSVISurface().implied_vol(0.0, T), atm_vol(T), rtol=1e-12)


def test_svi_surface_reproduces_each_slice(truth_table):
    s = fitted_surface(truth_table)
    for T, p in zip(T_SLICES, true_slices(T_SLICES), strict=True):
        np.testing.assert_allclose(s.implied_vol(K_GRID, T), p.implied_vol(K_GRID, T), rtol=1e-12)


def test_svi_surface_is_linear_in_total_variance_between_slices(truth_table):
    s = fitted_surface(truth_table)
    w_lo = s.total_variance(K_GRID, 0.25)
    w_hi = s.total_variance(K_GRID, 0.5)
    np.testing.assert_allclose(s.total_variance(K_GRID, 0.3125), 0.75 * w_lo + 0.25 * w_hi)


def test_svi_surface_holds_vol_fixed_outside_the_fitted_range(truth_table):
    s = fitted_surface(truth_table)
    np.testing.assert_allclose(s.implied_vol(K_GRID, 0.01), s.implied_vol(K_GRID, 0.05))
    np.testing.assert_allclose(s.implied_vol(K_GRID, 3.0), s.implied_vol(K_GRID, 1.0))


def test_svi_surface_has_no_calendar_arbitrage(truth_table):
    s = fitted_surface(truth_table)
    T = np.linspace(0.01, 1.5, 200)
    w = s.total_variance(K_GRID[:, None], T[None, :])
    assert np.all(np.diff(w, axis=1) >= -1e-15)


def test_svi_surface_broadcasts_k_against_T(truth_table):
    s = fitted_surface(truth_table)
    k = np.array([[-0.1], [0.0]])
    T = np.array([0.07, 0.3, 0.8])
    vol = s.implied_vol(k, T)
    assert vol.shape == (2, 3)
    assert vol[0, 1] == pytest.approx(float(s.implied_vol(-0.1, 0.3)))


def test_single_slice_surface_has_constant_vol_per_strike(truth_table):
    s = fitted_surface(truth_table.iloc[[2]])
    np.testing.assert_allclose(s.implied_vol(K_GRID, 0.1), s.implied_vol(K_GRID, 0.9))


def test_svi_surface_rejects_bad_slices(truth_table):
    with pytest.raises(ValueError, match="increasing"):
        SVISurface(T=np.array([0.5, 0.25]), slices=tuple(true_slices([0.5, 0.25])))
    with pytest.raises(ValueError, match="one SVI slice"):
        SVISurface(T=np.array([]), slices=())
