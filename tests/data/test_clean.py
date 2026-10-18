"""Tests for chain cleaning: quote filters, time to expiry, parity forward, OTM selection, IV.

The saved chain is generated from Black-Scholes with a known forward and vol for every contract
(see ``tests/fixtures/make_spy_fixture.py``), so the cleaner must recover those exactly. Its
planted bad quotes must be flagged, and nothing else.

The American-option tests use the synthetic SPY market from ``optdesk.surface.synthetic``:
quotes priced with early exercise on a CRR tree from a known vol surface. De-Americanizing
must recover that surface's vols and the true forward; skipping it must show the known bias.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from optdesk.data.clean import (
    CLEAN_COLUMNS,
    clean_chain,
    european_equivalent,
    parity_forward,
    quote_flags,
    year_fraction,
)
from optdesk.data.rates import RateCurve
from optdesk.pricing.binomial import binomial_price
from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.pricing.implied_vol import implied_vol
from optdesk.surface.svi import SVIParams
from optdesk.surface.synthetic import synthetic_chain

ASOF = pd.Timestamp("2026-10-16 19:45", tz="UTC")  # 3:45pm New York (EDT, UTC-4)

# --- Time to expiry ----------------------------------------------------------------------------


def test_year_fraction_one_week():
    """To Friday 23 Oct 4pm EDT = 20:00 UTC: 7 days and 15 minutes."""
    T = year_fraction(ASOF, pd.Timestamp("2026-10-23"))
    assert T == pytest.approx((7 * 1440 + 15) / (365 * 1440), abs=1e-15)


def test_year_fraction_across_daylight_saving():
    """US clocks go back on 1 Nov 2026, so 4pm on 20 Nov is 21:00 UTC: 35 days 1h15m."""
    T = year_fraction(ASOF, pd.Timestamp("2026-11-20"))
    assert T == pytest.approx((35 * 1440 + 75) / (365 * 1440), abs=1e-15)


def test_year_fraction_vectorized_and_same_day():
    expiries = pd.Series(pd.to_datetime(["2026-10-16", "2026-10-23"]))
    T = year_fraction(pd.Series([ASOF, ASOF]), expiries)
    np.testing.assert_allclose(T, [15 / (365 * 1440), (7 * 1440 + 15) / (365 * 1440)], rtol=1e-12)


def test_year_fraction_matches_fixture(truth):
    """The fixture's answer key was built with the same convention."""
    for expiry, info in truth["expiries"].items():
        assert year_fraction(ASOF, pd.Timestamp(expiry)) == pytest.approx(info["T"], abs=1e-15)


# --- Quote filters -----------------------------------------------------------------------------


def _flagged(raw, flags, column):
    return set(raw.loc[flags[column], "contract"])


def test_flags_catch_exactly_the_planted_defects(raw_chain, truth):
    flags = quote_flags(raw_chain)
    for kind in ("crossed", "stale"):
        assert _flagged(raw_chain, flags, kind) == {truth["defects"][kind]}, kind
    assert flags["zero_bid"].sum() == (raw_chain["bid"] == 0).sum() > 0

    # Besides the planted one, "wide" also catches genuine wing quotes: options worth a few
    # cents with a 1-cent half-spread, i.e. a spread over 50% of the mid.
    wide = _flagged(raw_chain, flags, "wide")
    assert truth["defects"]["wide"] in wide
    wings = raw_chain[raw_chain["contract"].isin(wide - {truth["defects"]["wide"]})]
    assert len(wings) > 0
    assert (wings["ask"] < 0.10).all()


def test_flag_rules_on_handmade_quotes():
    asof = ASOF
    raw = pd.DataFrame(
        {
            "bid": [1.00, 0.00, np.nan, 1.10, 1.00, 0.10, 1.00, 1.00],
            "ask": [1.10, 0.05, 1.00, 1.00, 1.00, 0.30, 1.10, 1.10],
            "asof": asof,
            "last_trade": [asof] * 6 + [asof - pd.Timedelta(days=6), pd.NaT],
        }
    )
    flags = quote_flags(raw)
    # ok, zero bid, missing bid, crossed, locked (ok), wide (spread 0.2 / mid 0.2 = 100%),
    # stale (6 days), never traded (stale)
    assert flags["zero_bid"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert flags["crossed"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert flags["wide"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert flags["stale"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_flag_thresholds_are_adjustable():
    raw = pd.DataFrame(
        {"bid": [1.0], "ask": [1.5], "asof": ASOF, "last_trade": ASOF - pd.Timedelta(days=10)}
    )
    assert not quote_flags(raw)["wide"].iloc[0]  # 40% spread is under the 50% default
    assert quote_flags(raw, max_rel_spread=0.3)["wide"].iloc[0]
    assert quote_flags(raw)["stale"].iloc[0]  # 10 days without a trade
    assert not quote_flags(raw, max_stale_days=None)["stale"].iloc[0]


# --- Parity forward ----------------------------------------------------------------------------


@given(
    S=st.floats(10.0, 1000.0),
    T=st.floats(1 / 365, 3.0),
    r=st.floats(-0.01, 0.10),
    q=st.floats(0.0, 0.08),
    sigma=st.floats(0.05, 1.0),
)
def test_parity_forward_recovers_true_forward(S, T, r, q, sigma):
    """From exact European prices, every strike gives F = S e^((r-q)T); so does the median."""
    F = float(forward_price(S, T, r, q))
    strikes = F * np.linspace(0.8, 1.2, 9)
    calls = bs_price(S, strikes, T, r, sigma, q, True)
    puts = bs_price(S, strikes, T, r, sigma, q, False)
    assert parity_forward(strikes, calls, puts, np.exp(-r * T)) == pytest.approx(F, rel=1e-9)


def test_parity_forward_ignores_a_bad_quote():
    """A mispriced pair far from the money isn't among the nearest strikes; one near the money
    is outvoted by the median."""
    S, T, r, sigma = 100.0, 0.5, 0.04, 0.2
    F = float(forward_price(S, T, r))
    strikes = np.arange(80.0, 121.0, 5.0)
    calls = bs_price(S, strikes, T, r, sigma, 0.0, True)
    puts = bs_price(S, strikes, T, r, sigma, 0.0, False)
    calls[strikes == 120.0] += 5.0  # far wing
    calls[strikes == 100.0] += 0.5  # near the money
    assert parity_forward(strikes, calls, puts, np.exp(-r * T)) == pytest.approx(F, rel=1e-9)
    # A plain average over all strikes would have been pulled more than 50 cents off:
    assert abs(np.mean(strikes + (calls - puts) * np.exp(r * T)) - F) > 0.5


def test_parity_forward_empty():
    assert np.isnan(parity_forward([], [], [], 0.99))


# --- Full cleaning on the saved chain ----------------------------------------------------------


@pytest.fixture
def clean(raw_chain, curve):
    return clean_chain(raw_chain, curve)


def test_clean_columns_and_order(clean):
    assert list(clean.columns) == CLEAN_COLUMNS
    assert clean.equals(clean.sort_values(["expiry", "strike"]).reset_index(drop=True))


def test_forward_recovered_per_expiry(clean, truth):
    got = clean.groupby(clean["expiry"].dt.strftime("%Y-%m-%d"))["forward"].first()
    for expiry, info in truth["expiries"].items():
        assert got[expiry] == pytest.approx(info["forward"], rel=1e-10), expiry


def test_rate_and_time_columns(clean, curve, truth):
    np.testing.assert_allclose(clean["r"], curve(clean["T"].to_numpy()), rtol=0)
    for expiry, info in truth["expiries"].items():
        rows = clean[clean["expiry"] == pd.Timestamp(expiry)]
        np.testing.assert_allclose(rows["T"], info["T"], rtol=1e-14)


def test_only_otm_one_option_per_strike(clean):
    calls = clean[clean["option_type"] == "call"]
    puts = clean[clean["option_type"] == "put"]
    assert (calls["strike"] >= calls["forward"]).all()
    assert (puts["strike"] < puts["forward"]).all()
    assert not clean.duplicated(subset=["expiry", "strike"]).any()
    assert (clean["k"] < 0).eq(clean["option_type"] == "put").all()


def test_implied_vols_match_the_truth(clean, source_chain):
    """Mids are exact model prices, so the solver must give back the generating vol."""
    true_vol = source_chain.set_index("contractSymbol")["impliedVolatility"]
    np.testing.assert_allclose(clean["iv"], true_vol.loc[clean["contract"]], atol=1e-8)


def test_log_moneyness_and_mid(clean):
    np.testing.assert_allclose(clean["k"], np.log(clean["strike"] / clean["forward"]))
    np.testing.assert_allclose(clean["mid"], (clean["bid"] + clean["ask"]) / 2)
    np.testing.assert_allclose(clean["rel_spread"], (clean["ask"] - clean["bid"]) / clean["mid"])


def test_bad_quotes_are_gone(clean, truth):
    assert (clean["bid"] > 0).all()
    assert (clean["rel_spread"] <= 0.5).all()
    assert not set(truth["defects"].values()) & set(clean["contract"])


def test_every_good_otm_quote_survives(raw_chain, clean, truth):
    """Nothing is lost beyond the filters: count the OTM, unflagged contracts by hand."""
    F = (
        raw_chain["expiry"]
        .dt.strftime("%Y-%m-%d")
        .map({e: info["forward"] for e, info in truth["expiries"].items()})
    )
    otm = np.where(
        raw_chain["option_type"] == "call", raw_chain["strike"] >= F, raw_chain["strike"] < F
    )
    good = ~quote_flags(raw_chain).any(axis=1)
    assert len(clean) == (otm & good).sum()


def test_filter_settings_pass_through(raw_chain, curve, truth):
    loose = clean_chain(raw_chain, curve, max_rel_spread=np.inf, max_stale_days=None)
    assert {truth["defects"]["wide"], truth["defects"]["stale"]} <= set(loose["contract"])


# --- Edge cases --------------------------------------------------------------------------------


def test_expiry_without_pairs_is_dropped_with_warning(raw_chain, curve):
    """Remove every put from one expiry: no parity pair, so no forward, so the expiry goes."""
    first = raw_chain["expiry"].min()
    no_puts = raw_chain[~((raw_chain["expiry"] == first) & (raw_chain["option_type"] == "put"))]
    with pytest.warns(UserWarning, match="no call/put pair"):
        out = clean_chain(no_puts, curve)
    assert first not in set(out["expiry"])
    assert out["expiry"].nunique() == raw_chain["expiry"].nunique() - 1


def test_expired_options_are_dropped(raw_chain, curve):
    later = raw_chain.assign(asof=pd.Timestamp("2026-10-24", tz="UTC"))
    later = later.assign(last_trade=later["asof"])  # keep them from being flagged stale
    out = clean_chain(later, curve)
    assert (out["expiry"] > pd.Timestamp("2026-10-23")).all()
    assert (out["T"] > 0).all()


def test_arbitrage_violating_quote_is_dropped(raw_chain, curve):
    """An OTM call quoted above the stock price has no implied vol, so it can't stay."""
    bad = raw_chain.copy()
    i = bad.index[(bad["option_type"] == "call") & (bad["strike"] == 640.0)][-1]
    bad.loc[i, ["bid", "ask"]] = [700.0, 701.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # and it shouldn't upset anything else
        out = clean_chain(bad, curve)
    assert bad.at[i, "contract"] not in set(out["contract"])
    assert out["iv"].notna().all()


def test_flat_curve_and_duplicate_strikes(raw_chain):
    """A repeated (expiry, type, strike) row, as adjusted contracts produce, appears once."""
    doubled = pd.concat([raw_chain, raw_chain.iloc[:5]], ignore_index=True)
    out = clean_chain(doubled, RateCurve.flat(0.04))
    assert not out.duplicated(subset=["expiry", "strike"]).any()


def test_after_hours_chain_warns_and_returns_empty(raw_chain, curve):
    """After the close, Yahoo often shows every bid as 0: nothing survives, and we say why."""
    closed = raw_chain.assign(bid=0.0)
    with pytest.warns(UserWarning, match="market was closed"):
        out = clean_chain(closed, curve)
    assert out.empty
    assert list(out.columns) == CLEAN_COLUMNS


# --- American options: de-Americanizing --------------------------------------------------------


def test_european_equivalent_recovers_the_vol():
    """American prices at a known vol -> strip the premium -> Black-Scholes gives that vol back."""
    S, r, q = 580.0, 0.04, 0.013
    T = np.array([0.25, 0.25, 1.0, 1.0, 1.0])
    K = np.array([560.0, 600.0, 450.0, 580.0, 620.0])
    sigma = np.array([0.19, 0.15, 0.26, 0.18, 0.16])
    call = np.array([False, True, False, False, True])
    american = sum(binomial_price(S, K, T, r, sigma, q, call, steps=n) for n in (1500, 1501)) / 2

    european, premium = european_equivalent(american, S, K, T, r, q, call)
    np.testing.assert_allclose(implied_vol(european, S, K, T, r, q, call), sigma, atol=1e-4)
    np.testing.assert_allclose(european + premium, american, rtol=1e-14)
    # Without the correction, the 1-year near-the-money put's vol is off by over 0.3 points.
    naive = implied_vol(american, S, K, T, r, q, call)
    assert naive[3] - sigma[3] > 0.003


@pytest.fixture(scope="module")
def american_market():
    """Exact (noise-free, unrounded) American quotes from the synthetic SPY surface."""
    return synthetic_chain(american=True, noise=0.0, tick=None)


@pytest.fixture(scope="module")
def american_clean(american_market):
    return clean_chain(american_market.raw, american_market.curve, american=True)


def _true_vols(clean, truth):
    """The surface's vol at each quote's strike, using the true forward."""
    out = np.empty(len(clean))
    for expiry, row in truth.set_index("expiry").iterrows():
        sel = (clean["expiry"] == expiry).to_numpy()
        k_true = np.log(clean.loc[sel, "strike"] / row["forward"])
        out[sel] = SVIParams.from_mapping(row).implied_vol(k_true, row["T"])
    return out


def test_american_cleaning_recovers_forward_and_vols(american_market, american_clean):
    truth = american_market.truth.set_index("expiry")
    forwards = american_clean.groupby("expiry")["forward"].first()
    np.testing.assert_allclose(forwards, truth.loc[forwards.index, "forward"], rtol=5e-5)  # 0.5bp
    error = np.abs(american_clean["iv"] - _true_vols(american_clean, american_market.truth))
    # The floor here is the *generator's* 400-step tree (about a cent on a $580 stock, so
    # ~0.02 vol points on low-vega quotes); de-Americanizing itself adds ~0.005.
    assert error.max() < 5e-4  # 0.05 vol points, every quote, every expiry
    assert error.median() < 1e-4  # typical quote: 0.01 vol points


def test_ignoring_early_exercise_shows_the_bias(american_market, american_clean):
    """Cleaning American quotes as European: forward too low, put vols too high, worse with T."""
    naive = clean_chain(american_market.raw, american_market.curve, american=False)
    truth = american_market.truth.set_index("expiry")
    gap_bp = 1e4 * (naive.groupby("expiry")["forward"].first() / truth["forward"] - 1)
    assert gap_bp.iloc[-1] < -20  # 14-month expiry: forward more than 20bp low
    assert gap_bp.iloc[0] > gap_bp.iloc[-1]  # bias grows with maturity
    long_dated = naive["expiry"] == truth.index[-1]
    error = naive["iv"] - _true_vols(naive, american_market.truth)
    assert error[long_dated].max() > 0.003  # some long-dated vol off by > 0.3 points


def test_eep_column(american_clean, raw_chain, curve):
    assert (american_clean["eep"] >= 0).all()
    puts = american_clean["option_type"] == "put"
    assert american_clean.loc[puts, "eep"].max() > 0.5  # long-dated puts: tens of cents+
    assert american_clean.loc[~puts, "eep"].max() < 0.01  # OTM calls: essentially none
    assert (clean_chain(raw_chain, curve)["eep"] == 0).all()  # European: nothing removed
