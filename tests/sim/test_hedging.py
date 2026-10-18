"""Tests for the delta-hedging engine, against answers known in closed form.

- Realized vol = hedge vol, no costs: mean P&L is zero (within standard error) whatever the
  real-world drift, and the spread shrinks like 1/sqrt(rebalances), matching Derman-Kamal.
- Realized vol != hedge vol: the sign is right (long gamma wins when realized > implied), the
  mean equals both the gamma-P&L formula and BS(realized) - BS(implied), and the formula
  tracks P&L path by path, under GBM and under Heston.
- Accounting: put-call parity of hedged P&Ls (exact with q = 0), a short position is exactly
  the negative of a long one, costs are non-negative, grow with frequency, and reduce P&L by
  exactly what they cost.
"""

import numpy as np
import pytest

from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import delta
from optdesk.sim.hedging import delta_hedge, discrete_hedging_std, frequency_sweep, gamma_pnl
from optdesk.sim.paths import gbm_paths, heston_paths

S0, K, T, R, Q, VOL = 100.0, 100.0, 0.25, 0.04, 0.013, 0.20
N_STEPS, N_PATHS = 441, 5_000  # 63 trading days x 7 hourly marks
FREQS = {"monthly": 147, "weekly": 35, "daily": 7, "hourly": 1}


@pytest.fixture(scope="module")
def fair_paths():
    """Risk-neutral GBM paths that realize exactly the hedge vol."""
    return gbm_paths(S0, R - Q, VOL, T, N_STEPS, N_PATHS, rng=100)


@pytest.fixture(scope="module")
def sweep(fair_paths):
    return frequency_sweep(fair_paths, K, R, VOL, Q, frequencies=FREQS).set_index("frequency")


def test_fair_vol_mean_pnl_is_zero_at_every_frequency(sweep):
    assert np.all(np.abs(sweep["mean"] / sweep["se"]) < 4)
    premium = float(bs_price(S0, K, T, R, VOL, Q))
    assert np.all(np.abs(sweep["mean"]) < 0.01 * premium)


def test_fair_vol_mean_pnl_is_zero_even_with_real_world_drift():
    """The stock's expected return doesn't matter to a hedged option."""
    p = gbm_paths(S0, 0.15, VOL, T, N_STEPS, N_PATHS, rng=101)
    res = delta_hedge(p, K, R, VOL, Q, rebalance_every=7)
    assert abs(res.pnl.mean()) < 4 * res.pnl.std() / np.sqrt(N_PATHS)


def test_pnl_std_shrinks_like_one_over_sqrt_rebalances(sweep):
    slope = np.polyfit(np.log(sweep["n_rebalances"]), np.log(sweep["std"]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)
    assert np.all(np.diff(sweep["std"]) < 0)


@pytest.mark.parametrize("label", ["weekly", "daily", "hourly"])
def test_pnl_std_matches_derman_kamal(sweep, label):
    row = sweep.loc[label]
    expected = discrete_hedging_std(S0, K, T, R, VOL, Q, int(row["n_rebalances"]))
    assert row["std"] == pytest.approx(expected, rel=0.1)


def test_n_rebalances_counts_the_opening_hedge(sweep):
    assert list(sweep["n_rebalances"]) == [3, 13, 63, 441]


@pytest.mark.parametrize("realized", [0.15, 0.25])
def test_wrong_vol_pnl_matches_gamma_formula_and_bs_value_gap(realized):
    """Long gamma: earn 1/2 gamma S^2 (realized var - implied var) dt, summed over the life."""
    p = gbm_paths(S0, R - Q, realized, T, N_STEPS, N_PATHS, rng=102)
    pnl = delta_hedge(p, K, R, VOL, Q).pnl
    formula = gamma_pnl(p, K, R, VOL, Q)
    value_gap = float(bs_price(S0, K, T, R, realized, Q) - bs_price(S0, K, T, R, VOL, Q))

    assert np.sign(pnl.mean()) == np.sign(realized - VOL)
    se = pnl.std() / np.sqrt(N_PATHS)
    assert abs(pnl.mean() - formula.mean()) < 4 * se
    assert abs(pnl.mean() - value_gap) < 4 * se
    # the formula is an expectation over the same paths and much less noisy than P&L
    assert abs(formula.mean() - value_gap) < 2 * formula.std() / np.sqrt(N_PATHS)
    # and it explains P&L path by path, not just on average
    assert np.corrcoef(pnl, formula)[0, 1] > 0.8


def test_gamma_formula_is_zero_when_realized_equals_hedge_vol(fair_paths):
    np.testing.assert_allclose(gamma_pnl(fair_paths, K, R, VOL, Q), 0.0, atol=1e-12)


def test_gamma_formula_tracks_pnl_under_heston():
    """Stochastic vol: the formula still holds with the path's own instantaneous variance."""
    params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    p = heston_paths(S0, mu=R - Q, T=T, n_steps=N_STEPS, n_paths=N_PATHS, rng=103, **params)
    pnl = delta_hedge(p, K, R, VOL, Q).pnl
    formula = gamma_pnl(p, K, R, VOL, Q)
    assert abs(pnl.mean() - formula.mean()) < 4 * pnl.std() / np.sqrt(N_PATHS)
    assert np.corrcoef(pnl, formula)[0, 1] > 0.95


def test_short_option_pnl_is_exactly_minus_long(fair_paths):
    long_ = delta_hedge(fair_paths, K, R, VOL, Q, rebalance_every=7).pnl
    short = delta_hedge(fair_paths, K, R, VOL, Q, rebalance_every=7, position=-1).pnl
    np.testing.assert_allclose(short, -long_, atol=1e-12)


def test_hedged_put_call_parity_is_exact_without_dividends():
    """Call minus put is a forward; its delta is exactly 1, so the hedge never trades."""
    p = gbm_paths(S0, R, VOL, T, 50, 1_000, rng=104)
    for every in (1, 3, 50):
        call = delta_hedge(p, K, R, VOL, 0.0, True, rebalance_every=every).pnl
        put = delta_hedge(p, K, R, VOL, 0.0, False, rebalance_every=every).pnl
        np.testing.assert_allclose(call, put, atol=1e-10)


def test_hedged_put_call_parity_holds_with_dividends():
    p = gbm_paths(S0, R - Q, VOL, T, N_STEPS, 1_000, rng=105)
    call = delta_hedge(p, K, R, VOL, Q, True).pnl
    put = delta_hedge(p, K, R, VOL, Q, False).pnl
    assert np.max(np.abs(call - put)) < 1e-3  # only the step-by-step dividend accrual differs


def test_costs_reduce_pnl_by_exactly_what_they_cost(fair_paths):
    free = delta_hedge(fair_paths, K, R, VOL, Q, rebalance_every=7)
    costly = delta_hedge(fair_paths, K, R, VOL, Q, rebalance_every=7, cost=0.0005)
    assert np.all(costly.costs > 0)
    np.testing.assert_allclose(costly.pnl, free.pnl - costly.costs, atol=1e-12)
    assert costly.pnl.mean() < free.pnl.mean()


def test_opening_hedge_cost_is_cost_times_delta_times_spot():
    """With one rebalance (the opening hedge), the whole cost is the first trade."""
    p = gbm_paths(S0, R - Q, VOL, T, 10, 10, rng=106)
    res = delta_hedge(p, K, R, VOL, Q, rebalance_every=10, cost=0.001)
    np.testing.assert_allclose(res.costs, 0.001 * float(delta(S0, K, T, R, VOL, Q)) * S0)
    assert res.n_rebalances == 1


def test_costs_grow_with_hedging_frequency(fair_paths):
    out = frequency_sweep(fair_paths, K, R, VOL, Q, frequencies=FREQS, cost=0.0005)
    assert np.all(np.diff(out["mean_cost"]) > 0)
    # Leland: costs grow roughly like sqrt(N); monthly to hourly is sqrt(147) ~ 12x
    ratio = out["mean_cost"].iloc[-1] / out["mean_cost"].iloc[0]
    assert 4 < ratio < 15


def test_frequency_sweep_columns(sweep):
    assert {"rebalance_every", "n_rebalances", "mean", "se", "std", "mean_cost", "pnl"} <= set(
        sweep.columns
    )
    assert all(len(p) == N_PATHS for p in sweep["pnl"])


def test_discrete_hedging_std_scales_with_sqrt_n():
    one = discrete_hedging_std(S0, K, T, R, VOL, Q, 1)
    assert discrete_hedging_std(S0, K, T, R, VOL, Q, 100) == pytest.approx(one / 10)


def test_bad_inputs_raise(fair_paths):
    with pytest.raises(ValueError):
        delta_hedge(fair_paths, K, R, VOL, rebalance_every=0)
    with pytest.raises(ValueError):
        delta_hedge(fair_paths, K, R, VOL, cost=-0.01)
