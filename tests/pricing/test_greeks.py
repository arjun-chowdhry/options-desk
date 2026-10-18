"""Tests for the closed-form Greeks.

Each Greek is checked against an independent answer: a finite-difference bump of the price (or
of a lower-order Greek), textbook values, relationships implied by put-call parity, and the
Black-Scholes PDE itself.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import (
    charm,
    charm_per_day,
    delta,
    gamma,
    rho,
    theta,
    theta_per_day,
    vanna,
    vega,
    vega_per_vol_point,
    volga,
    volga_per_vol_point,
)

ALL_GREEKS = [delta, gamma, vega, theta, rho, vanna, volga, charm]

# A spread of market setups: at/in/out of the money, short/long dated, with and without
# dividends, and a negative rate.
CASES = {
    "atm_1y": dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02),
    "itm_call_3m": dict(S=100.0, K=80.0, T=0.25, r=0.03, sigma=0.35, q=0.0),
    "otm_call_2y": dict(S=100.0, K=130.0, T=2.0, r=0.01, sigma=0.15, q=0.03),
    "neg_rate_high_vol": dict(S=50.0, K=45.0, T=0.1, r=-0.005, sigma=0.60, q=0.01),
}

# Bump size per input, chosen so truncation error (~h^2) and rounding error (~eps/h) are both
# far below the test tolerance.
BUMP = {"S": 1e-5, "sigma": 1e-5, "T": 1e-6, "r": 1e-5}  # S bump is relative to spot


def central_diff(fn, case, wrt, call):
    """(f(x + h) - f(x - h)) / 2h, bumping only the input named ``wrt``."""
    h = BUMP[wrt] * (case["S"] if wrt == "S" else 1.0)
    up = {**case, wrt: case[wrt] + h}
    dn = {**case, wrt: case[wrt] - h}
    return (fn(**up, call=call) - fn(**dn, call=call)) / (2 * h)


cases = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
call_put = pytest.mark.parametrize("call", [True, False], ids=["call", "put"])


# --- Every Greek vs a finite difference --------------------------------------------------------

# (greek, function it differentiates, input it differentiates with respect to, sign).
# Sign -1 for theta and charm: they measure time *passing*, i.e. minus the T-derivative.
FD_TABLE = [
    (delta, bs_price, "S", 1),
    (vega, bs_price, "sigma", 1),
    (theta, bs_price, "T", -1),
    (rho, bs_price, "r", 1),
    (gamma, delta, "S", 1),
    (vanna, delta, "sigma", 1),
    (vanna, vega, "S", 1),  # same Greek via the other mixed partial: d/dS(vega) = d/dsigma(delta)
    (volga, vega, "sigma", 1),
    (charm, delta, "T", -1),
]


@cases
@call_put
@pytest.mark.parametrize(
    "greek, base, wrt, sign",
    FD_TABLE,
    ids=[f"{g.__name__}=d{b.__name__}/d{w}" for g, b, w, _ in FD_TABLE],
)
def test_greek_matches_finite_difference(greek, base, wrt, sign, case, call):
    expected = sign * central_diff(base, case, wrt, call)
    assert greek(**case, call=call) == pytest.approx(expected, rel=1e-6, abs=1e-8)


@cases
def test_gamma_matches_second_difference_of_price(case):
    """Tie gamma straight to the price, not only to our own delta."""
    S, h = case["S"], case["S"] * 1e-3
    up, mid, dn = (bs_price(**{**case, "S": s}) for s in (S + h, S, S - h))
    assert gamma(**case) == pytest.approx((up - 2 * mid + dn) / h**2, rel=1e-4)


# --- Textbook values ---------------------------------------------------------------------------


def test_hull_greeks_example():
    """Hull, Options, Futures and Other Derivatives, ch. 19 running example: a call with
    S=49, K=50, r=5%, sigma=20%, T=20 weeks (0.3846) has delta 0.522, gamma 0.066, vega 12.1,
    theta -4.31 per year (-0.0118 per day), rho 8.91."""
    args = dict(S=49.0, K=50.0, T=0.3846, r=0.05, sigma=0.20)
    assert delta(**args) == pytest.approx(0.522, abs=5e-4)
    assert gamma(**args) == pytest.approx(0.066, abs=5e-4)
    assert vega(**args) == pytest.approx(12.1, abs=0.05)
    assert theta(**args) == pytest.approx(-4.31, abs=5e-3)
    assert theta_per_day(**args) == pytest.approx(-0.0118, abs=5e-5)
    assert rho(**args) == pytest.approx(8.91, abs=5e-3)


def test_unit_helpers_rescale():
    args = CASES["atm_1y"]
    assert vega_per_vol_point(**args) == pytest.approx(vega(**args) / 100)
    assert volga_per_vol_point(**args) == pytest.approx(volga(**args) / 10_000)
    assert theta_per_day(**args, call=False) == pytest.approx(theta(**args, call=False) / 365)
    assert charm_per_day(**args, call=False) == pytest.approx(charm(**args, call=False) / 365)


# --- Relationships that must hold for any inputs (Hypothesis) ---------------------------------

spots = st.floats(1.0, 1000.0)
moneyness = st.floats(0.2, 5.0)
expiries = st.floats(1 / 365, 5.0)
rates = st.floats(-0.02, 0.15)
yields = st.floats(0.0, 0.10)
vols = st.floats(0.01, 2.0)


@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols)
def test_greeks_respect_put_call_parity(S, m, T, r, q, sigma):
    """Differentiate C - P = S e^{-qT} - K e^{-rT}: the call and put Greeks must differ by the
    Greeks of that forward position. Calls and puts are coded separately, so this is a real
    check."""
    K = S * m
    args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    pv_s, pv_k = S * np.exp(-q * T), K * np.exp(-r * T)
    tol = 1e-9 * max(S, K)

    def diff(g):
        return g(**args, call=True) - g(**args, call=False)

    assert diff(delta) == pytest.approx(np.exp(-q * T), abs=1e-12)
    assert diff(theta) == pytest.approx(q * pv_s - r * pv_k, abs=tol)
    assert diff(rho) == pytest.approx(K * T * np.exp(-r * T), abs=tol * max(T, 1.0))
    assert diff(charm) == pytest.approx(q * np.exp(-q * T), abs=1e-12)


@cases
@call_put
def test_black_scholes_pde(case, call):
    """Every Black-Scholes price solves dV/dt + (r - q) S delta + 1/2 sigma^2 S^2 gamma = r V,
    with dV/dt = theta. This ties theta, delta, gamma and the price together."""
    S, r, q, sigma = case["S"], case["r"], case["q"], case["sigma"]
    V = bs_price(**case, call=call)
    lhs = (
        theta(**case, call=call)
        + (r - q) * S * delta(**case, call=call)
        + 0.5 * sigma**2 * S**2 * gamma(**case, call=call)
    )
    assert lhs == pytest.approx(r * V, abs=1e-10 * S)


# --- Sign and shape sanity --------------------------------------------------------------------


@cases
def test_long_option_is_long_gamma_and_vega(case):
    for call in (True, False):
        assert gamma(**case, call=call) > 0
        assert vega(**case, call=call) > 0


def test_volga_positive_in_wings_and_near_zero_atm():
    """Volga ~ vega * d1 * d2 / sigma: positive when d1, d2 share a sign (the wings)."""
    base = dict(S=100.0, T=1.0, r=0.0, sigma=0.2)
    K_atm = 100.0 * np.exp(0.5 * 0.2**2)  # makes d1 = 0 (with r = q = 0), so volga = 0
    assert volga(K=60.0, **base) > 0
    assert volga(K=160.0, **base) > 0
    assert abs(volga(K=K_atm, **base)) < 0.01 * volga(K=60.0, **base)


def test_broadcasting_matches_scalar_calls():
    K = np.array([90.0, 100.0, 110.0])
    sigma = np.array([[0.1], [0.3]])
    call = np.array([True, False, True])
    for g in ALL_GREEKS:
        grid = g(100.0, K, 0.5, 0.03, sigma, 0.01, call)
        assert grid.shape == (2, 3), g.__name__
        for i in range(2):
            for j in range(3):
                expected = g(100.0, K[j], 0.5, 0.03, sigma[i, 0], 0.01, call[j])
                assert grid[i, j] == pytest.approx(expected, rel=1e-14), g.__name__


# --- Zero vol / at expiry -----------------------------------------------------------------------


def test_at_expiry_delta_is_step_and_nan_at_strike():
    K = np.array([80.0, 100.0, 120.0])
    np.testing.assert_array_equal(delta(100.0, K, 0.0, 0.05, 0.3), [1.0, np.nan, 0.0])
    np.testing.assert_array_equal(delta(100.0, K, 0.0, 0.05, 0.3, call=False), [0.0, np.nan, -1.0])


def test_at_expiry_second_order_greeks_vanish_off_the_kink():
    K = np.array([80.0, 100.0, 120.0])
    for g in (gamma, vega, vanna, volga, charm):
        np.testing.assert_array_equal(g(100.0, K, 0.0, 0.05, 0.3), [0.0, np.nan, 0.0], g.__name__)


def test_zero_vol_greeks_match_the_forward_payoff():
    """With sigma = 0 an in-the-money call is just a forward: V = S e^{-qT} - K e^{-rT}. So
    delta = e^{-qT}, rho = K T e^{-rT}, theta = q S e^{-qT} - r K e^{-rT}."""
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.02
    args = dict(S=S, K=K, T=T, r=r, sigma=0.0, q=q)
    assert delta(**args) == pytest.approx(np.exp(-q * T))
    assert rho(**args) == pytest.approx(K * T * np.exp(-r * T))
    assert theta(**args) == pytest.approx(q * S * np.exp(-q * T) - r * K * np.exp(-r * T))
    assert charm(**args) == pytest.approx(q * np.exp(-q * T))
    assert gamma(**args) == 0.0
    # Out of the money, a zero-vol option is worth nothing and has no sensitivities.
    for g in ALL_GREEKS:
        assert g(**{**args, "K": 120.0}) == 0.0, g.__name__


def test_tiny_vol_greeks_approach_zero_vol_limit():
    args = dict(S=100.0, K=90.0, T=1.0, r=0.05, q=0.02)
    for g in (delta, theta, rho, charm):
        assert g(sigma=1e-6, **args) == pytest.approx(g(sigma=0.0, **args), abs=1e-9), g.__name__
