"""Tests for the Black-Scholes-Merton pricer.

Every test checks against an answer we know independently of the code: textbook values,
put-call parity, no-arbitrage price bounds, or exact limiting cases.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from optdesk.pricing.black_scholes import bs_price, d1_d2, forward_price

# --- Textbook values ---------------------------------------------------------------------------


def test_hull_example_no_dividends():
    """Hull, Options, Futures and Other Derivatives, Example 15.6:
    S=42, K=40, r=10%, sigma=20%, T=0.5 -> d1=0.7693, d2=0.6278, call=4.76, put=0.81."""
    d1, d2 = d1_d2(42, 40, 0.5, 0.10, 0.20)
    assert d1 == pytest.approx(0.7693, abs=1e-4)
    assert d2 == pytest.approx(0.6278, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, call=True) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, call=False) == pytest.approx(0.8086, abs=1e-4)


def test_hull_example_index_with_dividend_yield():
    """Hull, Example 17.1: index call, S=930, K=900, r=8%, q=3%, sigma=20%, T=2/12 -> 51.83."""
    assert bs_price(930, 900, 2 / 12, 0.08, 0.20, q=0.03) == pytest.approx(51.83, abs=0.01)


def test_haug_generalized_bs_put():
    """Haug, The Complete Guide to Option Pricing Formulas (Merton 1973 example):
    put with S=100, K=95, T=0.5, r=10%, q=5%, sigma=20% -> 2.4648."""
    assert bs_price(100, 95, 0.5, 0.10, 0.20, q=0.05, call=False) == pytest.approx(2.4648, abs=1e-4)


# --- No-arbitrage relationships (property-based: Hypothesis tries hundreds of random inputs) ---

spots = st.floats(1.0, 1000.0)
moneyness = st.floats(0.2, 5.0)  # K / S, so strikes stay in a sensible range around spot
expiries = st.floats(1 / 365, 5.0)
rates = st.floats(-0.02, 0.15)
yields = st.floats(0.0, 0.10)
vols = st.floats(0.01, 2.0)


@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols)
def test_put_call_parity(S, m, T, r, q, sigma):
    """C - P = S e^{-qT} - K e^{-rT} for any inputs. Holds by arbitrage, whatever the model."""
    K = S * m
    c = bs_price(S, K, T, r, sigma, q, call=True)
    p = bs_price(S, K, T, r, sigma, q, call=False)
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(rhs, abs=1e-9 * max(S, K))


@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols)
def test_price_bounds(S, m, T, r, q, sigma):
    """Model-free bounds (Hull ch. 11, with dividend yield):
    max(Se^{-qT} - Ke^{-rT}, 0) <= C <= Se^{-qT}  and  max(Ke^{-rT} - Se^{-qT}, 0) <= P <= Ke^{-rT}.
    """
    K = S * m
    pv_s, pv_k = S * np.exp(-q * T), K * np.exp(-r * T)
    tol = 1e-9 * max(S, K)
    c = bs_price(S, K, T, r, sigma, q, call=True)
    p = bs_price(S, K, T, r, sigma, q, call=False)
    assert max(pv_s - pv_k, 0.0) - tol <= c <= pv_s + tol
    assert max(pv_k - pv_s, 0.0) - tol <= p <= pv_k + tol


# --- Limiting cases ----------------------------------------------------------------------------


def test_at_expiry_price_is_intrinsic():
    S, K = 100.0, np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(bs_price(S, K, 0.0, 0.05, 0.3), [20.0, 0.0, 0.0])
    np.testing.assert_allclose(bs_price(S, K, 0.0, 0.05, 0.3, call=False), [0.0, 0.0, 20.0])


def test_zero_vol_is_discounted_forward_intrinsic():
    """With no volatility the asset ends at the forward for sure: price = DF * max(F - K, 0)."""
    S, T, r, q = 100.0, 1.0, 0.05, 0.02
    F = forward_price(S, T, r, q)
    K = np.array([90.0, F, 110.0])
    DF = np.exp(-r * T)
    np.testing.assert_allclose(bs_price(S, K, T, r, 0.0, q), DF * np.maximum(F - K, 0.0))
    np.testing.assert_allclose(
        bs_price(S, K, T, r, 0.0, q, call=False), DF * np.maximum(K - F, 0.0)
    )


def test_tiny_vol_approaches_zero_vol_limit():
    """The sigma -> 0 limit is continuous, so the special case doesn't create a jump."""
    args = dict(S=100.0, K=95.0, T=1.0, r=0.05, q=0.02)
    assert bs_price(sigma=1e-8, **args) == pytest.approx(bs_price(sigma=0.0, **args), abs=1e-10)


def test_huge_vol_call_approaches_discounted_spot():
    """As sigma -> infinity the call is worth the whole (dividend-discounted) asset, the put
    the whole discounted strike."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    assert bs_price(S, K, T, r, 50.0, q) == pytest.approx(S * np.exp(-q * T), rel=1e-6)
    assert bs_price(S, K, T, r, 50.0, q, call=False) == pytest.approx(K * np.exp(-r * T), rel=1e-6)


# --- Broadcasting and input handling -----------------------------------------------------------


def test_broadcasting_matches_scalar_calls():
    """A vector of strikes against a column of vols gives the same grid as pricing one by one."""
    K = np.array([90.0, 100.0, 110.0])
    sigma = np.array([[0.1], [0.3]])
    call = np.array([True, False, True])
    grid = bs_price(100.0, K, 0.5, 0.03, sigma, 0.01, call)
    assert grid.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            expected = bs_price(100.0, K[j], 0.5, 0.03, sigma[i, 0], 0.01, call[j])
            assert grid[i, j] == pytest.approx(expected, rel=1e-14)


def test_scalar_inputs_return_scalar():
    assert np.ndim(bs_price(100, 100, 1, 0.05, 0.2)) == 0


@pytest.mark.parametrize(
    "bad",
    [dict(S=-1.0), dict(S=0.0), dict(K=0.0), dict(T=-0.1), dict(sigma=-0.2)],
)
def test_invalid_inputs_raise(bad):
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    args.update(bad)
    with pytest.raises(ValueError):
        bs_price(**args)
