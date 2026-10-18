"""Tests for rate conversions and the interpolated rate curve.

Known answers: a hand-computed T-bill example, the defining identity of each conversion (the
continuous rate must reproduce the quoted instrument's price), and the interpolation rules
(exact at knots, linear in between, flat beyond the ends).
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from optdesk.data.rates import (
    RateCurve,
    bond_yield_to_continuous,
    discount_yield_to_continuous,
    treasury_curve,
)

# --- Quote conversions -------------------------------------------------------------------------


def test_tbill_hand_example():
    """A 91-day bill priced at 98 per 100 has discount yield 2 * 360/91 % (Hull's convention).
    Continuous rate: -ln(0.98) / (91/365) = 8.1033%."""
    d = 0.02 * 360 / 91
    assert discount_yield_to_continuous(d, 91) == pytest.approx(0.0810328370428, abs=1e-12)


@given(d=st.floats(0.0, 0.2), days=st.integers(1, 364))
def test_discount_yield_reproduces_bill_price(d, days):
    """Discounting at the continuous rate must give back the bill price 1 - d * days/360."""
    r = discount_yield_to_continuous(d, days)
    assert np.exp(-r * days / 365) == pytest.approx(1 - d * days / 360, rel=1e-12)


def test_bond_yield_hand_example():
    """4% semiannual: 2 ln(1.02) = 3.9605%. Continuous is always below the compounded quote."""
    assert bond_yield_to_continuous(0.04) == pytest.approx(0.0396052545924, abs=1e-12)


@given(y=st.floats(-0.01, 0.2), T=st.floats(0.1, 30.0), freq=st.sampled_from([1, 2, 4, 12]))
def test_bond_yield_growth_matches(y, T, freq):
    """(1 + y/freq)^(freq T) must equal e^(rT): same growth, two compounding conventions."""
    r = bond_yield_to_continuous(y, freq)
    assert np.exp(r * T) == pytest.approx((1 + y / freq) ** (freq * T), rel=1e-12)


def test_conversions_broadcast():
    assert discount_yield_to_continuous([0.01, 0.02, 0.03]).shape == (3,)
    assert bond_yield_to_continuous(np.full((2, 2), 0.04)).shape == (2, 2)


# --- Rate curve --------------------------------------------------------------------------------

CURVE = RateCurve(tenors=(0.25, 5.0, 10.0), rates=(0.05, 0.04, 0.045))


def test_curve_exact_at_knots():
    np.testing.assert_allclose(CURVE([0.25, 5.0, 10.0]), [0.05, 0.04, 0.045], atol=0)


def test_curve_linear_between_knots():
    """Halfway from 0.25y to 5y the rate is halfway between 5% and 4%."""
    assert CURVE((0.25 + 5.0) / 2) == pytest.approx(0.045, abs=1e-15)
    assert CURVE(7.5) == pytest.approx(0.0425, abs=1e-15)


def test_curve_flat_beyond_ends():
    assert CURVE(0.01) == 0.05
    assert CURVE(30.0) == 0.045


def test_discount_factor():
    T = np.array([0.1, 1.0, 3.0])
    np.testing.assert_allclose(CURVE.discount_factor(T), np.exp(-CURVE(T) * T), rtol=1e-15)
    assert CURVE.discount_factor(0.0) == 1.0


def test_curve_broadcasts_and_returns_scalars():
    assert CURVE(np.ones((3, 2))).shape == (3, 2)
    assert np.ndim(CURVE(1.0)) == 0


def test_flat_curve():
    flat = RateCurve.flat(0.03)
    np.testing.assert_allclose(flat([0.01, 1.0, 50.0]), 0.03)


def test_dict_round_trip():
    assert RateCurve.from_dict(CURVE.to_dict()) == CURVE


@pytest.mark.parametrize(
    "tenors, rates",
    [
        ((), ()),
        ((1.0, 2.0), (0.05,)),
        ((2.0, 1.0), (0.05, 0.04)),
        ((1.0, 1.0), (0.05, 0.04)),
        ((0.0, 1.0), (0.05, 0.04)),
        ((1.0,), (np.nan,)),
    ],
    ids=["empty", "length_mismatch", "decreasing", "duplicate", "zero_tenor", "nan_rate"],
)
def test_curve_rejects_bad_input(tenors, rates):
    with pytest.raises(ValueError):
        RateCurve(tenors=tenors, rates=rates)


# --- Treasury curve from Yahoo quotes ----------------------------------------------------------


def test_treasury_curve_converts_each_quote():
    curve = treasury_curve({"^TNX": 4.0, "^IRX": 5.0, "^FVX": 3.8})
    assert curve.tenors == pytest.approx((91 / 365, 5.0, 10.0))
    assert curve.rates == pytest.approx(
        (
            float(discount_yield_to_continuous(0.05)),
            float(bond_yield_to_continuous(0.038)),
            float(bond_yield_to_continuous(0.04)),
        )
    )


def test_treasury_curve_skips_missing_quotes():
    curve = treasury_curve({"^IRX": 5.0, "^FVX": float("nan")})
    assert curve.tenors == pytest.approx((91 / 365,))


def test_treasury_curve_errors():
    with pytest.raises(ValueError, match="no usable"):
        treasury_curve({"^IRX": float("nan")})
    with pytest.raises(ValueError, match="unknown"):
        treasury_curve({"^GSPC": 5000.0})
