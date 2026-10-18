"""Tests for the synthetic market: the answer key must be right before it can grade the fit.

Checks: the true slices are the SSVI surface we claim (ATM variance, no-arbitrage
conditions), exact European quotes equal Black-Scholes at the true vol, American quotes are
worth at least European ones, quotes look like a real screen (penny ticks, zero bids under a
cent, the fetch_chain columns), and a seed reproduces the chain exactly.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.data.fetch import RAW_COLUMNS
from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.surface import synthetic
from optdesk.surface.svi import SVIParams
from optdesk.surface.synthetic import atm_vol, synthetic_chain, true_slices


def test_ssvi_parameters_satisfy_no_arbitrage_conditions():
    assert synthetic.SSVI_ETA * (1 + abs(synthetic.SSVI_RHO)) <= 2  # no butterfly arbitrage
    assert 0 < synthetic.SSVI_GAMMA <= 0.5  # no calendar arbitrage


def test_true_slices_have_the_stated_atm_vol_and_rising_variance():
    T = np.array([0.02, 0.1, 0.5, 1.0, 3.0])
    slices = true_slices(T)
    atm_var = np.array([p.total_variance(0.0) for p in slices])
    np.testing.assert_allclose(atm_var, atm_vol(T) ** 2 * T, rtol=1e-13)
    assert np.all(np.diff(atm_var) > 0)
    assert all(p.rho == synthetic.SSVI_RHO for p in slices)


def test_smile_levels_look_like_spy():
    """~14% ATM at one week, puts richer than calls, skew ~ -1 vol point per 1% at 1 month."""
    (week,) = true_slices(7 / 365)
    assert 0.13 < week.implied_vol(0.0, 7 / 365) < 0.15
    (month,) = true_slices(1 / 12)
    skew = (month.implied_vol(0.01, 1 / 12) - month.implied_vol(-0.01, 1 / 12)) / 0.02
    assert -1.3 < skew < -0.7
    assert month.implied_vol(-0.05, 1 / 12) > month.implied_vol(0.05, 1 / 12)


def test_exact_european_quotes_are_black_scholes(exact_market):
    raw, truth = exact_market.raw, exact_market.truth.set_index("expiry")
    assert list(raw.columns) == RAW_COLUMNS
    for expiry, group in raw.groupby("expiry"):
        row = truth.loc[expiry]
        assert row["forward"] == pytest.approx(
            forward_price(synthetic.SPOT, row["T"], row["r"], synthetic.DIV_YIELD), rel=1e-14
        )
        vol = SVIParams.from_mapping(row).implied_vol(
            np.log(group["strike"] / row["forward"]), row["T"]
        )
        call = (group["option_type"] == "call").to_numpy()
        model = bs_price(
            synthetic.SPOT, group["strike"], row["T"], row["r"], vol, synthetic.DIV_YIELD, call
        )
        mid = (group["bid"] + group["ask"]) / 2
        priced = (group["bid"] > 0).to_numpy()  # zero-bid wings have a lopsided "mid"
        np.testing.assert_allclose(mid[priced], model[priced], rtol=1e-12, atol=1e-12)


def test_american_quotes_at_least_european():
    expiries = ["2027-06-17"]
    am = synthetic_chain(american=True, noise=0, tick=None, expiries=expiries).raw
    eu = synthetic_chain(american=False, noise=0, tick=None, expiries=expiries).raw
    gap = (am["bid"] + am["ask"]) / 2 - (eu["bid"] + eu["ask"]) / 2
    priced = (eu["bid"] > 0) & (eu["ask"] > 0.05)
    assert (gap[priced] > -0.01).all()  # equal up to the 400-step tree's ~half-cent error...
    deep_puts = (am["option_type"] == "put") & (am["strike"] > 650)
    assert (gap[deep_puts] > 0.5).all()  # ...and ITM puts carry a real premium


def test_quotes_look_like_a_screen(market):
    raw = market.raw
    for col in ("bid", "ask"):
        cents = raw[col] * 100
        np.testing.assert_allclose(cents, np.round(cents), atol=1e-6)  # penny ticks
    assert (raw["ask"] > raw["bid"]).all()
    assert (raw["bid"] == 0).any()  # far wings: nobody bids
    assert (raw.loc[raw["bid"] == 0, "ask"] <= 0.03).all()
    assert raw["last_trade"].max() <= raw["asof"].iloc[0]
    assert raw["contract"].is_unique and raw["contract"].str.match(r"SPY\d{6}[CP]\d{8}$").all()
    weekly = raw[raw["expiry"] == raw["expiry"].min()]
    assert np.diff(np.unique(weekly["strike"])).max() == 1.0  # $1 strikes near term


def test_seed_reproduces_the_chain():
    one = synthetic_chain(american=False, seed=7).raw
    pd.testing.assert_frame_equal(one, synthetic_chain(american=False, seed=7).raw)
    other = synthetic_chain(american=False, seed=8).raw
    assert not np.allclose(one["bid"], other["bid"])
