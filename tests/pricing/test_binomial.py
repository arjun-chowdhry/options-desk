"""Tests for the CRR binomial tree.

European prices are checked against Black-Scholes (convergence at the O(1/N) rate, and exact
put-call parity on the tree). American prices are checked against a textbook value and the
no-arbitrage facts: never below European or intrinsic, and an American call on a
non-dividend stock is never exercised early. The early-exercise premium is checked against a
fine-tree reference and the cases where it must be exactly zero.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optdesk.pricing.binomial import binomial_price, early_exercise_premium
from optdesk.pricing.black_scholes import bs_price

CASES = {
    "atm_1y": dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02),
    "itm_call_3m": dict(S=100.0, K=80.0, T=0.25, r=0.03, sigma=0.35, q=0.0),
    "otm_call_2y": dict(S=100.0, K=130.0, T=2.0, r=0.01, sigma=0.15, q=0.03),
    "neg_rate_high_vol": dict(S=50.0, K=45.0, T=0.1, r=-0.005, sigma=0.60, q=0.01),
}
cases = pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
call_put = pytest.mark.parametrize("call", [True, False], ids=["call", "put"])


def european(steps, **kwargs):
    return binomial_price(**kwargs, american=False, steps=steps)


# --- European: converges to Black-Scholes ----------------------------------------------------


@cases
@call_put
def test_european_error_shrinks_like_one_over_n(case, call):
    """CRR's error is O(1/N): error * N stays bounded as N grows 64-fold. (It oscillates, so
    we bound it rather than demand it fall at every step.)"""
    bs = bs_price(**case, call=call)
    for N in (50, 200, 800, 3200):
        assert abs(european(N, **case, call=call) - bs) * N < 3.0, N


@cases
@call_put
def test_european_matches_black_scholes_to_a_hundredth_of_a_cent(case, call):
    """Averaging N and N + 1 steps cancels most of the odd/even oscillation."""
    avg = 0.5 * (european(2000, **case, call=call) + european(2001, **case, call=call))
    assert avg == pytest.approx(bs_price(**case, call=call), abs=1e-4 * case["S"])


@cases
def test_put_call_parity_holds_exactly_on_the_tree(case):
    """The risk-neutral p makes the tree's forward exactly S e^{(r-q)T}, so C - P equals
    S e^{-qT} - K e^{-rT} to rounding, at any N, not just in the limit."""
    S, K, T, r, q = (case[k] for k in ("S", "K", "T", "r", "q"))
    for N in (1, 7, 100):
        c = european(N, **case, call=True)
        p = european(N, **case, call=False)
        assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10 * S)


# --- American ----------------------------------------------------------------------------------


def test_hull_american_put_five_steps():
    """Hull, Options, Futures and Other Derivatives, ch. 21: American put with S=50, K=50,
    r=10%, sigma=40%, T=5 months on a 5-step tree is worth 4.49."""
    assert binomial_price(50, 50, 5 / 12, 0.10, 0.40, call=False, steps=5) == pytest.approx(
        4.49, abs=5e-3
    )


spots = st.floats(1.0, 1000.0)
moneyness = st.floats(0.3, 3.0)
expiries = st.floats(1 / 365, 5.0)
rates = st.floats(0.0, 0.15)
yields = st.floats(0.0, 0.10)
vols = st.floats(0.05, 1.5)


@settings(max_examples=50, deadline=None)
@given(S=spots, m=moneyness, T=expiries, r=rates, q=yields, sigma=vols, call=st.booleans())
def test_american_at_least_european_and_intrinsic(S, m, T, r, q, sigma, call):
    """The right to exercise early can't have negative value, and an American option can
    always be exercised immediately, so it's worth at least its intrinsic value."""
    K = S * m
    args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, call=call)
    amer = binomial_price(**args, steps=200)
    tol = 1e-10 * max(S, K)
    assert amer >= european(200, **args) - tol
    assert amer >= (max(S - K, 0.0) if call else max(K - S, 0.0)) - tol


@cases
def test_american_put_beats_black_scholes_put(case):
    """With r > 0 a deep-ish ITM put is worth exercising early, so the American put exceeds
    the European (Black-Scholes) one by more than the tree's discretization error."""
    args = {**case, "K": case["S"] * 1.2, "r": 0.05}
    amer = binomial_price(**args, call=False, steps=1000)
    assert amer > bs_price(**args, call=False) + 1e-3


@cases
def test_american_call_without_dividends_equals_european(case):
    """Merton: with q = 0 and r >= 0 it never pays to exercise a call early (you'd give up
    time value and pay the strike sooner), so American = European on the same tree."""
    args = {**case, "q": 0.0, "r": max(case["r"], 0.0)}
    assert binomial_price(**args, call=True, steps=500) == pytest.approx(
        european(500, **args, call=True), abs=1e-12 * args["S"]
    )


def test_american_call_with_big_dividend_yield_exercises_early():
    args = dict(S=100.0, K=70.0, T=1.0, r=0.01, sigma=0.2, q=0.10)
    assert binomial_price(**args, steps=500) > european(500, **args) + 1e-2


def test_very_deep_itm_american_put_is_intrinsic():
    """Spot far below strike with positive rates: exercise now and earn interest on K."""
    assert binomial_price(10.0, 100.0, 1.0, 0.05, 0.2, call=False) == pytest.approx(90.0)


def test_american_put_converges():
    args = dict(S=50.0, K=50.0, T=5 / 12, r=0.10, sigma=0.40, call=False)
    assert binomial_price(**args, steps=2000) == pytest.approx(
        binomial_price(**args, steps=4000), abs=1e-3
    )


# --- Edges and shapes --------------------------------------------------------------------------


def test_at_expiry_is_intrinsic():
    K = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(binomial_price(100.0, K, 0.0, 0.05, 0.3), [20.0, 0.0, 0.0])
    np.testing.assert_allclose(
        binomial_price(100.0, K, 0.0, 0.05, 0.3, call=False), [0.0, 0.0, 20.0]
    )


def test_broadcasting_matches_scalar_calls():
    K = np.array([90.0, 100.0, 110.0])
    sigma = np.array([[0.1], [0.3]])
    call = np.array([True, False, True])
    grid = binomial_price(100.0, K, 0.5, 0.03, sigma, 0.01, call, steps=100)
    assert grid.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            expected = binomial_price(100.0, K[j], 0.5, 0.03, sigma[i, 0], 0.01, call[j], steps=100)
            assert grid[i, j] == pytest.approx(expected, rel=1e-13)
    assert np.ndim(binomial_price(100, 100, 1, 0.05, 0.2, steps=10)) == 0


@pytest.mark.parametrize(
    "bad",
    [
        dict(steps=0),
        dict(sigma=0.0),
        dict(S=-1.0),
        dict(sigma=0.01, r=0.5, steps=1),  # drift beats vol over one huge step: p > 1
    ],
)
def test_invalid_inputs_raise(bad):
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, steps=50)
    args.update(bad)
    with pytest.raises(ValueError):
        binomial_price(**args)


# --- Early-exercise premium ----------------------------------------------------------------


def test_premium_zero_for_call_without_dividends():
    """Merton: never exercise a call early on a non-dividend stock, so the premium is 0."""
    K = np.array([80.0, 100.0, 120.0])
    eep = early_exercise_premium(100.0, K, 1.0, 0.05, 0.2, 0.0, True)
    np.testing.assert_allclose(eep, 0.0, atol=1e-12)


def test_premium_zero_for_put_at_zero_rates():
    """With r = q = 0 there's no interest to earn on K by exercising, so a put waits too."""
    K = np.array([80.0, 100.0, 120.0])
    eep = early_exercise_premium(100.0, K, 1.0, 0.0, 0.2, 0.0, False)
    np.testing.assert_allclose(eep, 0.0, atol=1e-12)


def test_put_premium_positive_and_grows_in_the_money():
    """With r > 0 the right to take K early is worth more the deeper the put is in the money."""
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    eep = early_exercise_premium(100.0, K, 1.0, 0.05, 0.2, 0.0, False)
    assert np.all(eep > 0)
    assert np.all(np.diff(eep) > 0)


def test_call_premium_positive_with_high_dividend():
    """With q > r a deep ITM call is worth exercising to collect the stock's yield."""
    assert early_exercise_premium(100.0, 70.0, 1.0, 0.01, 0.2, 0.08, True) > 0.1


def test_premium_matches_fine_tree_reference():
    """SPY-like puts, 1 month to 1 year: within ~1.5 cents of a 4,000-step reference."""
    S, r, q = 580.0, 0.04, 0.013
    T = np.repeat([1 / 12, 0.25, 1.0], 4)
    K = S * np.exp(np.tile([-0.2, -0.05, 0.0, 0.1], 3))
    sigma = 0.2

    def fine(american):
        return (
            sum(
                binomial_price(S, K, T, r, sigma, q, False, american=american, steps=n)
                for n in (4000, 4001)
            )
            / 2
        )

    reference = fine(True) - fine(False)
    eep = early_exercise_premium(S, K, T, r, sigma, q, False)
    np.testing.assert_allclose(eep, reference, atol=0.015)
    assert reference.max() > 1.0  # the test covers premiums that matter, not just zeros
