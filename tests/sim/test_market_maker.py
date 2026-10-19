"""Tests for the market maker, against answers known in closed form or by construction.

- Quoting formulas: symmetric quotes at zero inventory, the skew moves both quotes against
  the inventory by exactly ``gamma sigma_v^2 H V``, the spread equals the textbook
  Avellaneda-Stoikov spread, tends to ``2/k`` as risk aversion goes to 0, and grows with vol
  of vol (always) and with risk aversion (when ``sigma_v^2 H k^2 >= 1``, as A-S implies).
- Simulation answer keys: Poisson fill counts match their expectation; the three P&L buckets
  add up exactly to the mark-to-market P&L; with no vol moves the inventory bucket is zero;
  with realized vol = implied vol the hedging bucket averages minus the hedge costs; a flat
  surface makes both delta methods identical.
- Behaviour: risk limits bind, the run is reproducible from its seeds, skewing reduces
  inventory variance, and a fitted SVI surface can be swapped in for the analytic one.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.sim.market_maker import (
    Contract,
    QuoteParams,
    fill_intensity,
    half_spread,
    quote_vols,
    reservation_shift,
    run_market_maker,
    simulate_market,
)
from optdesk.surface.source import FlatSurface, fitted_surface, synthetic_surface
from optdesk.surface.svi import PARAM_NAMES
from optdesk.surface.synthetic import DIV_YIELD, SPOT, true_slices

R, H_YEARS, N_STEPS = 0.04, 1 / 12, 21 * 26
CONTRACTS = [
    Contract(525.0, 2 / 12, False),
    Contract(585.0, 2 / 12, True),
    Contract(555.0, 3 / 12, False),
    Contract(615.0, 6 / 12, True),
]


def market(n_paths=200, seed=1, vol_of_vol=0.10, base_vol=0.15, corr=-0.7):
    return simulate_market(
        SPOT, R, DIV_YIELD, base_vol, H_YEARS, N_STEPS, n_paths, seed,
        vol_of_vol=vol_of_vol, spot_vol_corr=corr,
    )  # fmt: skip


@pytest.fixture(scope="module")
def mkt():
    return market()


@pytest.fixture(scope="module")
def pair(mkt):
    """Skewed and unskewed desks on the same market and the same customer random numbers."""
    surf = synthetic_surface()
    return {
        skew: run_market_maker(mkt, CONTRACTS, surf, QuoteParams(skew=skew), rng=2)
        for skew in (False, True)
    }


# --- Quoting formulas ---------------------------------------------------------------------------


def test_quotes_are_symmetric_at_zero_inventory():
    p = QuoteParams()
    bid, ask = quote_vols(0.2, 50_000.0, 0.0, p)
    half = half_spread(50_000.0, p.risk_aversion, p.vol_of_vol, p.horizon, p.fill_decay)
    assert 0.2 - bid == pytest.approx(half, rel=1e-12)
    assert ask - 0.2 == pytest.approx(half, rel=1e-12)


def test_unskewed_quotes_ignore_inventory():
    p = QuoteParams(skew=False)
    np.testing.assert_array_equal(quote_vols(0.2, 5e4, 3e6, p), quote_vols(0.2, 5e4, 0.0, p))


@pytest.mark.parametrize("book_vega", [-2e6, -1e5, 1e5, 2e6])
def test_skew_moves_both_quotes_against_the_inventory(book_vega):
    p = QuoteParams()
    bid0, ask0 = quote_vols(0.2, 5e4, 0.0, p)
    bid, ask = quote_vols(0.2, 5e4, book_vega, p)
    expected = -p.risk_aversion * p.vol_of_vol**2 * p.horizon * book_vega
    assert bid - bid0 == pytest.approx(expected, rel=1e-9)
    assert ask - ask0 == pytest.approx(expected, rel=1e-9)
    assert np.sign(ask - ask0) == -np.sign(book_vega)  # long vega -> lower vols
    assert ask - bid == pytest.approx(ask0 - bid0, rel=1e-12)  # skew doesn't change the width


def test_reservation_shift_value():
    # 1e-4 / $ x (0.10)^2 x 1/252 x $1m per 1.00 vol = 0.00397 (0.4 of a vol point)
    assert reservation_shift(1e6, 1e-4, 0.10, 1 / 252) == pytest.approx(1e-4 * 0.01 / 252 * 1e6)


def test_half_spread_is_the_textbook_avellaneda_stoikov_spread():
    """In 'units of one fill' A-S give spread = g s^2 H + (2/g) ln(1 + g/kappa)."""
    g, sv, H, k, nu = 3e-5, 0.12, 1 / 252, 150.0, 8e4
    sigma_unit, kappa_unit = nu * sv, k / nu  # a fill's price vol, and fill decay per $
    textbook = g * sigma_unit**2 * H + (2 / g) * np.log(1 + g / kappa_unit)
    assert 2 * half_spread(nu, g, sv, H, k) * nu == pytest.approx(textbook, rel=1e-12)


def test_half_spread_tends_to_one_over_k_as_risk_aversion_vanishes():
    assert half_spread(8e4, 0.0, 0.1, 1 / 252, 200.0) == 1 / 200
    assert half_spread(8e4, 1e-12, 0.1, 1 / 252, 200.0) == pytest.approx(1 / 200, rel=1e-6)


def test_spread_grows_with_vol_of_vol():
    sv = np.linspace(0.0, 0.5, 11)
    assert np.all(np.diff(half_spread(8e4, 5e-5, sv, 1 / 252, 200.0)) > 0)


def test_spread_grows_with_risk_aversion_when_inventory_risk_dominates():
    sv, H, k = 0.10, 1 / 252, 200.0
    assert sv**2 * H * k**2 >= 1  # A-S: then d(spread)/d(gamma) > 0 for every gamma
    gammas = np.geomspace(1e-7, 1e-3, 30)
    assert np.all(np.diff(half_spread(8e4, gammas, sv, H, k)) > 0)


def test_fill_intensity_decays_exponentially():
    assert fill_intensity(0.0, 1000.0, 200.0) == 1000.0
    assert fill_intensity(0.005, 1000.0, 200.0) == pytest.approx(1000.0 / np.e)
    assert fill_intensity(-0.005, 1000.0, 200.0) == pytest.approx(1000.0 * np.e)


# --- The simulated market -----------------------------------------------------------------------


def test_market_moments():
    m = market(n_paths=20_000, seed=5, base_vol=0.15, vol_of_vol=0.10, corr=-0.7)
    T = m.times[-1]
    x_T = m.vol_shift[:, -1]
    assert x_T.mean() == pytest.approx(0.0, abs=4 * 0.10 * np.sqrt(T) / np.sqrt(20_000))
    assert x_T.std() == pytest.approx(0.10 * np.sqrt(T), rel=0.03)
    discounted = m.S[:, -1] * np.exp(-(R - DIV_YIELD) * T)
    se = discounted.std() / np.sqrt(len(discounted))
    assert discounted.mean() == pytest.approx(SPOT, abs=4 * se)  # martingale
    log_ret = np.diff(np.log(m.S), axis=1).ravel()
    dx = np.diff(m.vol_shift, axis=1).ravel()
    assert np.corrcoef(log_ret, dx)[0, 1] == pytest.approx(-0.7, abs=0.01)


def test_market_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulate_market(-1, R, 0, 0.2, 1, 10, 5)
    with pytest.raises(ValueError):
        simulate_market(100, R, 0, 0.2, 1, 10, 5, vol_of_vol=-0.1)
    with pytest.raises(ValueError):
        simulate_market(100, R, 0, 0.2, 1, 10, 5, spot_vol_corr=1.5)


# --- Simulation answer keys ---------------------------------------------------------------------


def test_fill_counts_match_poisson_expectation():
    """With gamma = 0 every quote sits exactly 1/k from theo, so fills ~ Poisson(2 n A e^-1 T)."""
    m = market(n_paths=400, vol_of_vol=0.0)
    p = QuoteParams(risk_aversion=0.0, skew=False)
    res = run_market_maker(m, CONTRACTS, FlatSurface(0.15), p, rng=3)
    expected = 2 * len(CONTRACTS) * p.arrival_rate * np.exp(-1) * H_YEARS
    se = np.sqrt(expected / len(res.fills))
    assert res.fills.mean() == pytest.approx(expected, abs=4 * se)
    assert res.fills.var() == pytest.approx(expected, rel=0.2)  # Poisson: variance = mean


def test_pnl_buckets_add_up_exactly(pair):
    for res in pair.values():
        scale = np.abs(res.total).max()
        np.testing.assert_allclose(
            res.spread + res.hedging + res.inventory, res.total, rtol=0, atol=1e-9 * scale
        )
        np.testing.assert_allclose(res.cum_spread[:, -1], res.spread)
        np.testing.assert_allclose(res.cum_inventory[:, -1], res.inventory)


def test_unskewed_spread_capture_is_positive(pair):
    res = pair[False]
    assert np.all(res.spread[res.fills > 0] > 0)
    assert np.all(res.costs >= 0)


def test_no_vol_moves_means_no_inventory_pnl():
    m = market(n_paths=20, vol_of_vol=0.0)
    res = run_market_maker(m, CONTRACTS, synthetic_surface(), QuoteParams(), rng=4)
    np.testing.assert_array_equal(res.inventory, 0.0)


def test_hedging_pnl_is_just_costs_when_realized_equals_implied():
    """Flat 15% surface, spot realizes 15%: the hedged options earn nothing on average."""
    m = market(n_paths=400, vol_of_vol=0.0, base_vol=0.15)
    res = run_market_maker(m, CONTRACTS, FlatSurface(0.15), QuoteParams(), rng=6)
    gross = res.hedging + res.costs
    assert abs(gross.mean()) < 4 * gross.std() / np.sqrt(len(gross))
    assert res.costs.mean() > 0


def test_delta_methods_agree_on_a_flat_surface():
    m = market(n_paths=10)
    a, b = (
        run_market_maker(m, CONTRACTS, FlatSurface(0.15), QuoteParams(delta_method=dm), rng=7)
        for dm in ("sticky_strike", "sticky_moneyness")
    )
    np.testing.assert_allclose(a.delta, b.delta, atol=1e-3)  # shares; FD bump error only
    np.testing.assert_allclose(a.total, b.total, rtol=1e-6)


def test_delta_methods_differ_on_a_skewed_surface(mkt):
    surf = synthetic_surface()
    a, b = (
        run_market_maker(mkt, CONTRACTS, surf, QuoteParams(delta_method=dm), rng=7)
        for dm in ("sticky_strike", "sticky_moneyness")
    )
    assert not np.allclose(a.hedging, b.hedging)
    np.testing.assert_array_equal(a.spread, b.spread)  # same quotes and fills


# --- Behaviour ----------------------------------------------------------------------------------


def test_skewing_reduces_inventory_variance(pair):
    no, yes = pair[False], pair[True]
    assert yes.vega[:, -1].std() < 0.5 * no.vega[:, -1].std()
    assert np.abs(yes.vega).mean() < 0.5 * np.abs(no.vega).mean()
    assert yes.inventory.std() < 0.6 * no.inventory.std()
    assert yes.total.std() < no.total.std()


def test_skew_trades_against_the_book(pair):
    """Long vega -> positive shift (vols quoted lower), short -> negative.

    ``shift[i + 1]`` is set from the book at the start of step i + 1, which is the book recorded
    after step i's trades (``vega[i]``) re-marked at the new spot and vol. The re-mark can flip
    the sign of a book that is barely long or short, hence the 99.9%.
    """
    res = pair[True]
    book, shift = res.vega[:, :-2], res.shift[:, 1:-1]
    has_book = np.abs(book) > 1_000
    assert np.mean(np.sign(shift[has_book]) == np.sign(book[has_book])) > 0.999
    np.testing.assert_array_equal(pair[False].shift, 0.0)


@pytest.mark.parametrize(("field", "limit"), [("vega_limit", 2_000.0), ("gamma_limit", 50_000.0)])
def test_risk_limits_bind(field, limit):
    """The pre-trade check keeps the book inside the limit nearly all the time.

    It never trades out of a breach, so the market moving the Greeks (and the odd step with
    several fills) can push exposure past the limit for a while; we allow 10% of the time.
    """
    m = market(n_paths=100)
    surf = synthetic_surface()
    attr = field.removesuffix("_limit")
    free = run_market_maker(m, CONTRACTS, surf, QuoteParams(skew=False), rng=8)
    capped = run_market_maker(m, CONTRACTS, surf, QuoteParams(skew=False, **{field: limit}), rng=8)
    exposure, free_exposure = np.abs(getattr(capped, attr)), np.abs(getattr(free, attr))
    assert np.median(free_exposure) > 2 * limit  # the limit really is tight
    assert np.mean(exposure > limit) < 0.10
    assert np.median(exposure) < limit
    assert exposure.max() < free_exposure.max() / 5
    assert capped.at_limit.mean() > 0.2 and free.at_limit.max() == 0


def test_run_is_reproducible(mkt):
    surf = synthetic_surface()
    a = run_market_maker(mkt, CONTRACTS, surf, rng=11)
    b = run_market_maker(mkt, CONTRACTS, surf, rng=11)
    c = run_market_maker(mkt, CONTRACTS, surf, rng=12)
    np.testing.assert_array_equal(a.total, b.total)
    np.testing.assert_array_equal(a.vega, b.vega)
    assert not np.array_equal(a.total, c.total)
    m1, m2 = market(n_paths=5, seed=3), market(n_paths=5, seed=3)
    np.testing.assert_array_equal(m1.S, m2.S)


def test_a_fitted_svi_surface_swaps_in_for_the_analytic_one():
    """SVI slices of the true surface every ~2 weeks interpolate back to nearly the same desk."""
    T = np.arange(0.05, 0.75, 0.04)
    table = pd.DataFrame(
        [{"T": t, **dict(zip(PARAM_NAMES, vars(p).values(), strict=True))}
         for t, p in zip(T, true_slices(T), strict=True)]
    )  # fmt: skip
    m = market(n_paths=50)
    a = run_market_maker(m, CONTRACTS, synthetic_surface(), rng=13)
    b = run_market_maker(m, CONTRACTS, fitted_surface(table), rng=13)
    assert np.corrcoef(a.total, b.total)[0, 1] > 0.99
    assert b.total.mean() == pytest.approx(a.total.mean(), rel=0.02)


def test_rejects_bad_setup(mkt):
    surf = synthetic_surface()
    with pytest.raises(ValueError, match="expire"):
        run_market_maker(mkt, [Contract(580.0, 0.05, True)], surf)
    with pytest.raises(ValueError, match="delta_method"):
        run_market_maker(mkt, CONTRACTS, surf, QuoteParams(delta_method="sticky_delta"))
