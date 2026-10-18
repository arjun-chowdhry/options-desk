"""Tests for raw SVI: the formula, its derivatives, and parameter recovery by the fitter.

Recovery is tested at three levels of realism:
1. Vols computed straight from known parameters: the fit must return those parameters to
   ~1e-8, for equity-, FX- and commodity-shaped smiles and the Vogt slice.
2. The exact synthetic chain run through the real cleaner: the same, for every expiry.
3. The realistic American chain (noise, rounding, early exercise): the fitted smile must be
   within 0.1 vol points of the truth everywhere, and the parameters close from 1 month out.
   (For 1-2 week expiries, very different parameter sets draw almost the same smile over
   the few strikes quoted, so there we only check the smile.)
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.pricing.greeks import vega
from optdesk.surface.svi import (
    PARAM_NAMES,
    SVIParams,
    fit_chain,
    fit_svi,
    ssvi_slice,
    svi_derivatives,
    svi_total_variance,
)

SLICES = {  # (params, T, quoted k range)
    "equity_1m": (SVIParams(0.0008, 0.012, -0.6, 0.01, 0.03), 1 / 12, (-0.2, 0.08)),
    "symmetric_fx": (SVIParams(0.01, 0.05, 0.0, 0.0, 0.2), 0.5, (-0.4, 0.4)),
    "commodity_call_skew": (SVIParams(0.02, 0.1, 0.4, 0.05, 0.15), 1.0, (-0.5, 0.6)),
    "vogt_negative_a": (SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153), 1.0, (-1.0, 1.5)),
}
slices = pytest.mark.parametrize("params, T, k_range", SLICES.values(), ids=SLICES.keys())


def _as_array(p: SVIParams) -> np.ndarray:
    return np.array([getattr(p, name) for name in PARAM_NAMES])


# --- The formula -------------------------------------------------------------------------------


def test_hand_values():
    p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.1, sigma=0.2)
    # At k = m the square root is just sigma: w = a + b sigma.
    assert p.total_variance(0.1) == pytest.approx(0.04 + 0.4 * 0.2, abs=1e-15)
    # k = 0.4: x = 0.3, sqrt(0.09 + 0.04) = sqrt(0.13).
    expected = 0.04 + 0.4 * (-0.4 * 0.3 + np.sqrt(0.13))
    assert p.total_variance(0.4) == pytest.approx(expected, abs=1e-15)
    assert p.implied_vol(0.4, 0.5) == pytest.approx(np.sqrt(expected / 0.5), abs=1e-15)


def test_vectorized_and_matches_function():
    p = SVIParams(0.01, 0.1, -0.5, 0.0, 0.1)
    k = np.linspace(-1, 1, 7)
    np.testing.assert_array_equal(
        p.total_variance(k), svi_total_variance(k, 0.01, 0.1, -0.5, 0, 0.1)
    )
    assert p.total_variance(k).shape == (7,)


@slices
def test_derivatives_match_finite_differences(params, T, k_range):
    k = np.linspace(*k_range, 25)
    h = 1e-4 * params.sigma  # sigma sets the curvature scale; a sharp smile needs a small step
    w, dw, d2w = svi_derivatives(k, params)
    np.testing.assert_allclose(w, params.total_variance(k), rtol=1e-14)
    # Central differences, one derivative at a time: w' from w, then w'' from w'.
    fd1 = (params.total_variance(k + h) - params.total_variance(k - h)) / (2 * h)
    fd2 = (svi_derivatives(k + h, params)[1] - svi_derivatives(k - h, params)[1]) / (2 * h)
    np.testing.assert_allclose(dw, fd1, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(d2w, fd2, rtol=1e-7)
    assert np.all(d2w > 0)  # every SVI slice is convex in k


@slices
def test_min_variance_and_wing_slopes(params, T, k_range):
    k = np.linspace(-20, 20, 400001)
    w = params.total_variance(k)
    assert params.min_variance == pytest.approx(w.min(), abs=1e-9)
    left, right = params.wing_slopes
    assert (w[0] - w[1]) / (k[1] - k[0]) == pytest.approx(left, rel=1e-3)
    assert (w[-1] - w[-2]) / (k[-1] - k[-2]) == pytest.approx(right, rel=1e-3)


def test_implied_vol_nan_where_variance_negative():
    p = SVIParams(a=-0.05, b=0.1, rho=0.0, m=0.0, sigma=0.1)  # w(0) = -0.04
    vols = p.implied_vol(np.array([0.0, 2.0]), 1.0)
    assert np.isnan(vols[0]) and vols[1] > 0


def test_ssvi_slice_matches_ssvi_formula():
    theta, rho, phi = 0.03, -0.7, 4.0
    k = np.linspace(-1, 1, 41)
    direct = theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    np.testing.assert_allclose(ssvi_slice(theta, rho, phi).total_variance(k), direct, rtol=1e-13)
    assert ssvi_slice(theta, rho, phi).total_variance(0.0) == pytest.approx(theta, rel=1e-14)


def test_from_mapping_round_trip():
    p = SVIParams(0.01, 0.1, -0.5, 0.02, 0.3)
    assert SVIParams.from_mapping(pd.Series(vars(p))) == p
    assert SVIParams.from_mapping(vars(p)) == p


# --- Fitting: exact data -----------------------------------------------------------------------


@slices
@pytest.mark.parametrize("weighted", [False, True], ids=["equal", "vega"])
def test_fit_recovers_known_parameters(params, T, k_range, weighted):
    k = np.linspace(*k_range, 40)
    iv = params.implied_vol(k, T)
    weights = vega(1.0, np.exp(k), T, 0.0, iv) if weighted else None
    fit = fit_svi(k, iv, T, weights)
    assert fit.success, fit.message
    np.testing.assert_allclose(_as_array(fit.params), _as_array(params), rtol=1e-7, atol=1e-9)
    assert fit.rmse < 1e-9 and fit.max_error < 1e-9
    assert fit.n == 40 and fit.T == T


def test_fit_chain_recovers_every_expiry_of_exact_chain(exact_market, exact_clean):
    fits = fit_chain(exact_clean).set_index("expiry")
    truth = exact_market.truth.set_index("expiry")
    assert list(fits.index) == list(truth.index)
    assert fits["success"].all()
    np.testing.assert_allclose(fits[PARAM_NAMES], truth[PARAM_NAMES], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(fits["forward"], truth["forward"], rtol=1e-12)
    assert (fits["rmse"] < 1e-9).all()


# --- Fitting: realistic noisy data -------------------------------------------------------------


def _smile_error(fit_row, truth_row, k):
    """Largest vol gap between fit and truth over ``k`` (fit's own forward; truth's forward)."""
    fitted = SVIParams.from_mapping(fit_row).implied_vol(k, fit_row["T"])
    k_true = k + np.log(fit_row["forward"] / truth_row["forward"])
    true = SVIParams.from_mapping(truth_row).implied_vol(k_true, truth_row["T"])
    return np.abs(fitted - true).max()


def _check_noisy_recovery(clean, fits, truth):
    fits, truth = fits.set_index("expiry"), truth.set_index("expiry")
    assert fits["success"].all()
    for expiry, row in fits.iterrows():
        quoted = clean.loc[clean["expiry"] == expiry, "k"]
        k = np.linspace(quoted.min(), quoted.max(), 200)
        assert _smile_error(row, truth.loc[expiry], k) < 0.001, expiry  # 0.1 vol points
        assert row["rmse"] < 0.001  # residuals at the noise level, not model misfit
    later = fits[fits["T"] > 0.08]  # 1 month and beyond: parameters well identified
    true = truth.loc[later.index]
    for name, tol in (("a", 0.25), ("b", 0.06), ("sigma", 0.06)):
        np.testing.assert_array_less(np.abs(later[name] / true[name] - 1), tol, err_msg=name)
    np.testing.assert_array_less(np.abs(later["rho"] - true["rho"]), 0.1)
    np.testing.assert_array_less(np.abs(later["m"] - true["m"]), 0.005)


def test_fit_recovers_surface_from_realistic_american_chain(market, market_clean, market_fits):
    _check_noisy_recovery(market_clean, market_fits, market.truth)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fit_recovery_holds_across_noise_seeds(seed):
    """Fresh noise each time (European pricing keeps this fast)."""
    from optdesk.data.clean import clean_chain
    from optdesk.surface.synthetic import synthetic_chain

    market = synthetic_chain(american=False, seed=seed)
    clean = clean_chain(market.raw, market.curve)
    _check_noisy_recovery(clean, fit_chain(clean), market.truth)


def test_vega_weighting_resists_a_bad_wing_quote():
    """One deep-OTM quote 3 vol points off: vega weights barely let it move the ATM vol."""
    params, T, (k_lo, k_hi) = SLICES["equity_1m"]
    k = np.linspace(k_lo, k_hi, 30)
    iv = params.implied_vol(k, T)
    iv[0] += 0.03  # the deepest put: a stale or fat-fingered wing quote
    weights = vega(1.0, np.exp(k), T, 0.0, iv)
    atm = params.implied_vol(0.0, T)
    equal_err = abs(fit_svi(k, iv, T).params.implied_vol(0.0, T) - atm)
    vega_err = abs(fit_svi(k, iv, T, weights).params.implied_vol(0.0, T) - atm)
    assert vega_err < equal_err / 3


# --- Input handling ----------------------------------------------------------------------------


def test_too_few_quotes_raises():
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi([-0.1, 0.0, 0.1, 0.2], [0.2, 0.18, 0.17, 0.17], 0.5)


def test_fit_chain_skips_thin_expiries_and_checks_weighting(exact_clean):
    first = exact_clean["expiry"].min()
    thin = pd.concat(
        [
            exact_clean[exact_clean["expiry"] != first],
            exact_clean[exact_clean["expiry"] == first][:4],
        ]
    )
    fits = fit_chain(thin)
    assert first not in set(fits["expiry"])
    assert fits["T"].is_monotonic_increasing
    with pytest.raises(ValueError, match="weighting"):
        fit_chain(exact_clean, weighting="spread")
