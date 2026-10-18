"""Tests for the implied-vol solver.

The core check is the round trip: price an option at a known vol, invert, and recover that
vol. On top of that: a textbook value, NaN outside the no-arbitrage bounds, both solver paths
(Newton and the Brent fallback) agreeing, and hard cases (deep ITM, extreme vols).
"""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import vega
from optdesk.pricing.implied_vol import _brent, implied_vol

# --- Textbook value ----------------------------------------------------------------------------


def test_hull_implied_vol_example():
    """Hull, Options, Futures and Other Derivatives, ch. 15: a European call with S=21, K=20,
    r=10%, T=0.25 trading at 1.875 has an implied vol of about 23.5%."""
    assert implied_vol(1.875, 21, 20, 0.25, 0.10) == pytest.approx(0.235, abs=5e-4)


# --- Round trip (property-based) ---------------------------------------------------------------

spots = st.floats(1.0, 1000.0)
moneyness = st.floats(0.3, 3.0)
expiries = st.floats(7 / 365, 5.0)
rates = st.floats(-0.02, 0.15)
yields = st.floats(0.0, 0.10)
vols = st.floats(0.02, 3.0)


@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols, call=st.booleans())
def test_round_trip_recovers_vol(S, m, T, r, q, sigma, call):
    """price(sigma) -> implied_vol -> sigma, whenever the vol is actually identifiable.

    If vega is tiny, a price rounding error of 1e-14 already moves the vol a lot, so no solver
    can recover it to 1e-8; we skip those cases with ``assume``.
    """
    K = S * m
    assume(vega(S, K, T, r, sigma, q) > 1e-4 * S)
    price = bs_price(S, K, T, r, sigma, q, call)
    assert implied_vol(price, S, K, T, r, q, call) == pytest.approx(sigma, abs=1e-8)


@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols)
def test_call_and_put_give_the_same_vol(S, m, T, r, q, sigma):
    """Put-call parity means a call and a put on the same strike must imply the same vol."""
    K = S * m
    assume(vega(S, K, T, r, sigma, q) > 1e-4 * S)
    c = bs_price(S, K, T, r, sigma, q, True)
    p = bs_price(S, K, T, r, sigma, q, False)
    assert implied_vol(c, S, K, T, r, q, True) == pytest.approx(
        implied_vol(p, S, K, T, r, q, False), abs=1e-8
    )


# --- Solver paths and hard cases --------------------------------------------------------------

HARD_CASES = {
    "atm": dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2),
    "deep_itm_call": dict(S=100.0, K=50.0, T=0.5, r=0.03, q=0.01, sigma=0.25),
    "deep_otm_call": dict(S=100.0, K=200.0, T=1.0, r=0.03, q=0.0, sigma=0.4),
    "tiny_vol": dict(S=100.0, K=101.0, T=1.0, r=0.0, q=0.0, sigma=0.01),
    "huge_vol": dict(S=100.0, K=100.0, T=0.5, r=0.02, q=0.0, sigma=5.0),
    "one_week": dict(S=100.0, K=103.0, T=7 / 365, r=0.05, q=0.0, sigma=0.3),
}


@pytest.mark.parametrize("case", HARD_CASES.values(), ids=HARD_CASES.keys())
@pytest.mark.parametrize("call", [True, False], ids=["call", "put"])
@pytest.mark.parametrize("max_iter", [50, 0], ids=["newton", "brent_only"])
def test_hard_cases_on_both_solver_paths(case, call, max_iter):
    """max_iter=0 skips Newton entirely, so every option goes through the Brent fallback."""
    args = {k: v for k, v in case.items() if k != "sigma"}
    price = bs_price(**args, sigma=case["sigma"], call=call)
    iv = implied_vol(price, **args, call=call, max_iter=max_iter)
    assert iv == pytest.approx(case["sigma"], abs=1e-8)


def test_deep_itm_call_keeps_precision():
    """An ITM call whose time value is ~1e-6 of its price still inverts accurately, because
    the solver works on the equivalent OTM put."""
    S, K, T, r, sigma = 100.0, 60.0, 0.25, 0.05, 0.3
    price = bs_price(S, K, T, r, sigma, call=True)
    time_value = price - (S - K * np.exp(-r * T))
    assert time_value < 1e-4 * price
    assert implied_vol(price, S, K, T, r) == pytest.approx(sigma, abs=1e-7)


# --- No-arbitrage bounds -----------------------------------------------------------------------


def test_nan_outside_no_arbitrage_bounds():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.02
    pv_s, pv_k = S * np.exp(-q * T), K * np.exp(-r * T)
    call_lo, put_lo = pv_s - pv_k, 0.0  # this call is ITM on the forward, the put is OTM

    for bad_call in [call_lo - 0.01, call_lo, pv_s, pv_s + 1.0, -1.0, np.nan]:
        assert np.isnan(implied_vol(bad_call, S, K, T, r, q, True)), bad_call
    for bad_put in [put_lo - 0.01, put_lo, pv_k, pv_k + 1.0]:
        assert np.isnan(implied_vol(bad_put, S, K, T, r, q, False)), bad_put

    # Just inside the bounds a vol does exist.
    assert np.isfinite(implied_vol(call_lo + 0.5, S, K, T, r, q, True))
    assert np.isfinite(implied_vol(pv_s - 0.5, S, K, T, r, q, True))


def test_brent_fallback_returns_nan_when_it_cannot_bracket():
    """The fallback's two safety exits: a price below what a near-zero vol gives (no time
    value to fit), and one that would need a vol above the 10,000 % search cap."""
    assert np.isnan(_brent(1e-12, 100.0, 100.0, 1.0, 0.0, 0.0, True))
    # With T = 1e-4, even sigma = 100 only prices this ATM call at about 38.
    assert np.isnan(_brent(50.0, 100.0, 100.0, 1e-4, 0.0, 0.0, True))


def test_nan_at_expiry():
    assert np.isnan(implied_vol(5.0, 100.0, 95.0, 0.0, 0.05))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        implied_vol(5.0, -100.0, 95.0, 1.0, 0.05)


# --- Broadcasting ------------------------------------------------------------------------------


def test_vectorized_chain_with_mixed_valid_and_invalid_quotes():
    """A whole smile at once: one invalid quote becomes NaN without spoiling the others."""
    S, T, r, q = 100.0, 0.5, 0.04, 0.01
    K = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    true_vol = np.array([0.32, 0.26, 0.22, 0.20, 0.21])  # a typical equity skew
    call = K >= S  # OTM calls above spot, OTM puts below
    prices = bs_price(S, K, T, r, true_vol, q, call)
    prices[1] = -1.0
    iv = implied_vol(prices, S, K, T, r, q, call)
    assert iv.shape == (5,)
    assert np.isnan(iv[1])
    np.testing.assert_allclose(np.delete(iv, 1), np.delete(true_vol, 1), atol=1e-8)


def test_broadcast_grid_and_scalar_output():
    K = np.array([90.0, 100.0, 110.0])
    sigma = np.array([[0.15], [0.4]])
    prices = bs_price(100.0, K, 1.0, 0.03, sigma)
    iv = implied_vol(prices, 100.0, K, 1.0, 0.03)
    assert iv.shape == (2, 3)
    np.testing.assert_allclose(iv, np.broadcast_to(sigma, (2, 3)), atol=1e-8)
    assert np.ndim(implied_vol(10.0, 100.0, 100.0, 1.0, 0.03)) == 0
