"""Tests for portfolio Greeks and scenarios, against answers known exactly.

- One position's row is the closed-form Greek x quantity x multiplier, in desk units.
- Put-call parity: long call + short put (same strike, expiry, vol) is a forward, so its
  delta is ``e^{-qT}`` per share and its gamma and vega are zero.
- The scenario grid is zero at zero shock, and its finite differences recover the book's
  delta, gamma and vega (the grid is full revaluation, the Greeks are derivatives).
- The Taylor estimate agrees with full revaluation for small moves and not for big ones.
- Risk is linear in quantity; offsetting positions cancel; the share hedge flattens delta.
- Vols come from the surface at ``ln(K/F)`` unless a position carries its own.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.pricing.greeks import delta, gamma, rho, theta, vanna, vega, volga
from optdesk.pricing.portfolio import (
    PRESETS,
    Position,
    as_frame,
    book_risk,
    delta_hedge_shares,
    position_vols,
    positions_from_frame,
    preset_book,
    scenario_pnl,
    taylor_pnl,
    totals,
)
from optdesk.surface.source import FlatSurface, synthetic_surface

S, R, Q = 580.0, 0.04, 0.013
FLAT = FlatSurface(0.2)
SKEW = synthetic_surface()


def test_single_position_row_is_greek_times_size():
    p = Position(K=600.0, T=0.25, call=True, quantity=-7)
    row = book_risk([p], S, R, Q, FLAT).iloc[0]
    args = (S, 600.0, 0.25, R, 0.2, Q, True)
    size = -7 * 100
    assert row["delta"] == pytest.approx(size * delta(*args))
    assert row["dollar_delta"] == pytest.approx(size * delta(*args) * S)
    assert row["gamma"] == pytest.approx(size * gamma(*args))
    assert row["gamma_1pct"] == pytest.approx(size * gamma(*args) * S**2 / 100)
    assert row["vega"] == pytest.approx(size * vega(*args) / 100)
    assert row["theta"] == pytest.approx(size * theta(*args) / 365)
    assert row["rho"] == pytest.approx(size * rho(*args) / 100)
    assert row["vanna"] == pytest.approx(size * vanna(*args) / 100)
    assert row["volga"] == pytest.approx(size * volga(*args) / 100**2)


def test_call_minus_put_is_a_forward():
    T = 0.5
    book = [Position(590.0, T, True, 1), Position(590.0, T, False, -1)]
    total = totals(book_risk(book, S, R, Q, SKEW))  # same strike -> same surface vol
    assert total["delta"] == pytest.approx(100 * np.exp(-Q * T), rel=1e-10)
    assert total["gamma"] == pytest.approx(0.0, abs=1e-10)
    assert total["vega"] == pytest.approx(0.0, abs=1e-8)
    assert total["volga"] == pytest.approx(0.0, abs=1e-8)
    # and its value is the discounted forward minus strike: S e^{-qT} - K e^{-rT}
    assert total["value"] == pytest.approx(100 * (S * np.exp(-Q * T) - 590 * np.exp(-R * T)))


def test_totals_are_column_sums_and_offsetting_positions_cancel():
    a, b = Position(560.0, 0.2, False, 4), Position(610.0, 0.4, True, -3)
    both = totals(book_risk([a, b], S, R, Q, SKEW))
    separately = totals(book_risk([a], S, R, Q, SKEW)) + totals(book_risk([b], S, R, Q, SKEW))
    np.testing.assert_allclose(both, separately, rtol=1e-12)
    flat_book = [a, Position(a.K, a.T, a.call, -a.quantity)]
    assert np.allclose(totals(book_risk(flat_book, S, R, Q, SKEW)), 0.0, atol=1e-9)


def test_risk_is_linear_in_quantity():
    one = totals(book_risk([Position(580.0, 0.3, True, 1)], S, R, Q, SKEW))
    five = totals(book_risk([Position(580.0, 0.3, True, 5)], S, R, Q, SKEW))
    np.testing.assert_allclose(five, 5 * one, rtol=1e-12)


def test_surface_vol_unless_overridden():
    T, K = 0.25, 540.0
    F = S * np.exp((R - Q) * T)
    vols = position_vols(
        [Position(K, T, False, 1), Position(K, T, False, 1, vol=0.33)], S, R, Q, SKEW
    )
    assert vols[0] == pytest.approx(float(SKEW.implied_vol(np.log(K / F), T)))
    assert vols[1] == 0.33
    assert vols[0] > float(SKEW.implied_vol(0.0, T))  # an OTM put sits up the skew


def test_scenario_is_zero_at_zero_shock():
    book = preset_book("2m call butterfly (95/100/105)", S, R, Q)
    grid = scenario_pnl(book, S, R, Q, SKEW, [-0.05, 0.0, 0.05], [-0.02, 0.0, 0.02])
    assert grid.loc[0.0, 0.0] == 0.0
    assert grid.shape == (3, 3)


def test_scenario_finite_differences_recover_the_greeks():
    book = [Position(560.0, 0.15, False, -10), Position(600.0, 0.4, True, 6)]
    total = totals(book_risk(book, S, R, Q, SKEW))
    h = 1e-4  # relative spot bump
    grid = scenario_pnl(book, S, R, Q, SKEW, [-h, 0.0, h], [0.0])
    down, mid, up = grid.iloc[0]
    dS = S * h
    assert (up - down) / (2 * dS) == pytest.approx(total["delta"], rel=1e-5)
    assert (up - 2 * mid + down) / dS**2 == pytest.approx(total["gamma"], rel=1e-3)
    dv = 1e-4  # vol bump (absolute)
    vgrid = scenario_pnl(book, S, R, Q, SKEW, [0.0], [-dv, 0.0, dv])
    vega_fd = (vgrid.iloc[2, 0] - vgrid.iloc[0, 0]) / (2 * dv) / 100  # $ per vol point
    assert vega_fd == pytest.approx(total["vega"], rel=1e-5)


def test_taylor_matches_small_moves_but_not_large_ones():
    book = preset_book("Long 1m ATM straddle", S, R, Q)
    total = totals(book_risk(book, S, R, Q, SKEW))
    shocks = np.array([0.002, 0.001, -0.15, 0.15])
    full = scenario_pnl(book, S, R, Q, SKEW, shocks, [0.0]).iloc[0].to_numpy()
    approx = taylor_pnl(total, S, shocks).to_numpy()
    error = np.abs(approx - full)
    # a second-order expansion leaves a third-order error: halve the move, error / 8
    assert error[0] / error[1] == pytest.approx(8, rel=0.15)
    np.testing.assert_allclose(approx[:2], full[:2], rtol=5e-3)
    assert np.all(error[2:] > 0.05 * np.abs(full[2:]))  # but a 15% move is far off
    # a pure vol shock: vega dv + 1/2 volga dv^2 tracks the repriced book
    vol_full = scenario_pnl(book, S, R, Q, SKEW, [0.0], [0.01]).iloc[0, 0]
    assert taylor_pnl(total, S, [0.0], 0.01).iloc[0] == pytest.approx(vol_full, rel=1e-3)


def test_share_hedge_flattens_delta_and_is_linear():
    book = preset_book("3m risk reversal (long 105% call, short 95% put)", S, R, Q)
    shares = delta_hedge_shares(book_risk(book, S, R, Q, SKEW))
    assert shares < 0  # a risk reversal is long delta, so the hedge sells stock
    hedged = totals(book_risk(book, S, R, Q, SKEW, shares=shares))
    assert abs(hedged["delta"]) <= 0.5
    stock_only = scenario_pnl([], S, R, Q, SKEW, [-0.1, 0.1], [0.0], shares=200)
    np.testing.assert_allclose(stock_only.iloc[0], [-0.1 * S * 200, 0.1 * S * 200])


def test_presets_have_the_expected_risk_signs():
    def risk(name):
        return totals(book_risk(preset_book(name, S, R, Q), S, R, Q, SKEW))

    straddle = risk("Long 1m ATM straddle")
    assert straddle["gamma"] > 0 and straddle["vega"] > 0 and straddle["theta"] < 0
    assert abs(straddle["delta"]) < 0.1 * 1000  # near flat: 10 calls + 10 puts at the forward
    calendar = risk("ATM calendar (short 1m, long 3m)")
    assert calendar["vega"] > 0 and calendar["gamma"] < 0
    fly = risk("2m call butterfly (95/100/105)")
    assert fly["gamma"] < 0 and fly["vega"] < 0
    assert risk("3m risk reversal (long 105% call, short 95% put)")["delta"] > 0
    assert set(PRESETS) == {
        "Long 1m ATM straddle",
        "3m risk reversal (long 105% call, short 95% put)",
        "ATM calendar (short 1m, long 3m)",
        "2m call butterfly (95/100/105)",
    }


def test_frame_round_trip_and_validation():
    book = [Position(575.0, 30 / 365, True, 10), Position(560.0, 91 / 365, False, -5, vol=0.21)]
    frame = as_frame(book)
    assert list(frame.columns) == ["type", "strike", "days", "quantity", "vol"]
    back = positions_from_frame(frame)
    assert [(p.K, p.call, p.quantity) for p in back] == [(575.0, True, 10), (560.0, False, -5)]
    assert back[0].vol is None and back[1].vol == pytest.approx(0.21)
    assert back[0].T == pytest.approx(30 / 365)

    with_blank = pd.concat(
        [frame, pd.DataFrame([{"type": "call", "strike": 1.0, "days": 1, "quantity": 0}])]
    )
    assert len(positions_from_frame(with_blank)) == 2  # zero quantity is skipped
    with pytest.raises(ValueError, match="call"):
        positions_from_frame(frame.assign(type="straddle"))
    with pytest.raises(ValueError, match="strike"):
        positions_from_frame(frame.assign(strike=np.nan))
    with pytest.raises(ValueError):
        Position(K=-1.0, T=0.1, call=True, quantity=1)


def test_empty_book():
    risk = book_risk([], S, R, Q, SKEW)
    assert risk.empty and (totals(risk) == 0).all()
