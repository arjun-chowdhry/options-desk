"""Tests for the dashboard's experiment runners, against the M4 answer keys.

Hedging lab
- Realized = implied, no costs: mean P&L is zero (within 4 standard errors) and the std is
  close to Derman-Kamal at the chosen frequency; the std falls as frequency rises.
- Realized above implied: a long option makes money, matching ``BS(realized) - BS(implied)``;
  a short one loses the same. Costs lower the mean. Heston runs and realizes about its
  long-run vol. A seed reproduces the run. Frequencies longer than the option are refused.

Market-making lab
- Buckets add up to the total; skewing narrows the end-of-run vega; the quote curve is
  symmetric at zero inventory and skews down when long vega (for the skewed desk only).
"""

import numpy as np
import pytest

from optdesk.pricing.black_scholes import bs_price
from optdesk.sim.lab import (
    FREQUENCIES,
    HedgeConfig,
    MMConfig,
    hedge_grid,
    heston_params,
    run_hedge_lab,
    run_mm_lab,
)
from optdesk.surface.source import synthetic_surface

FAIR = HedgeConfig(n_paths=2_000, seed=11)


@pytest.fixture(scope="module")
def fair():
    return run_hedge_lab(FAIR)


def test_fair_vol_hedging_has_zero_mean_and_derman_kamal_std(fair):
    s = fair.summary
    assert abs(s["mean"]) < 4 * s["se"]
    assert s["std"] == pytest.approx(s["derman_kamal"], rel=0.15)
    assert s["premium"] == pytest.approx(float(bs_price(100, 100, 0.25, 0.04, 0.2, 0.013)))
    assert s["bs_gap"] == 0.0 and s["gamma_formula_mean"] == pytest.approx(0.0, abs=1e-12)
    assert s["realized_vol"] == pytest.approx(0.2)
    assert fair.sweep["std"].is_monotonic_decreasing
    assert list(fair.sweep["frequency"]) == list(FREQUENCIES)
    assert fair.sample_S.shape == (30, hedge_grid(0.25)[0] + 1)
    assert fair.days[0] == 0 and fair.days[-1] == pytest.approx(63)


def test_wrong_vol_pnl_matches_bs_gap_and_flips_when_short():
    cfg = HedgeConfig(realized_vol=0.30, frequency="hourly", n_paths=2_000, seed=5)
    long = run_hedge_lab(cfg).summary
    assert long["bs_gap"] > 0
    assert abs(long["mean"] - long["bs_gap"]) < 4 * long["se"]
    assert long["gamma_formula_mean"] == pytest.approx(long["bs_gap"], rel=0.1)
    short = run_hedge_lab(HedgeConfig(**{**vars(cfg), "position": -1.0})).summary
    assert short["mean"] == pytest.approx(-long["mean"], rel=1e-10)


def test_costs_lower_the_mean(fair):
    costly = run_hedge_lab(HedgeConfig(**{**vars(FAIR), "cost": 5e-4})).summary
    assert costly["mean_cost"] > 0
    assert costly["mean"] == pytest.approx(fair.summary["mean"] - costly["mean_cost"], rel=1e-9)


def test_heston_realizes_its_long_run_vol():
    params = heston_params(0.2, 0.2, 2.0, 0.5, -0.7)
    assert params == pytest.approx(
        {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    )
    lab = run_hedge_lab(HedgeConfig(model="heston", heston=params, n_paths=1_000, seed=3))
    assert 0.17 < lab.summary["realized_vol"] < 0.22
    assert np.isnan(lab.summary["bs_gap"])


def test_seed_reproduces_and_bad_inputs_are_refused():
    a, b = run_hedge_lab(HedgeConfig(n_paths=200)), run_hedge_lab(HedgeConfig(n_paths=200))
    np.testing.assert_array_equal(a.pnl, b.pnl)
    _, freqs = hedge_grid(10 / 365)  # 7 trading days: weekly fits, monthly does not
    assert "monthly" not in freqs and "hourly" in freqs
    with pytest.raises(ValueError, match="doesn't fit"):
        run_hedge_lab(HedgeConfig(T=10 / 365, frequency="monthly"))
    with pytest.raises(ValueError, match="model"):
        run_hedge_lab(HedgeConfig(model="sabr"))


@pytest.fixture(scope="module")
def mm():
    return run_mm_lab(MMConfig(n_paths=40, days=5, seed=1), synthetic_surface(), 580, 0.04, 0.013)


def test_mm_buckets_add_up_and_skew_narrows_vega(mm):
    s = mm.summary.set_index("strategy")
    np.testing.assert_allclose(
        s["spread_mean"] + s["hedging_mean"] + s["inventory_mean"], s["total_mean"], rtol=1e-9
    )
    no, yes = mm.results["No skew"], mm.results["A-S skew"]
    assert yes.vega[:, -1].std() < no.vega[:, -1].std()
    for label, cum in mm.mean_cum.items():
        assert cum["total"].iloc[-1] == pytest.approx(mm.results[label].total.mean(), rel=1e-9)
    lo, med, hi = mm.vega_bands["A-S skew"]
    assert np.all(lo <= med) and np.all(med <= hi)
    assert mm.days[-1] == pytest.approx(5) and len(mm.contracts) == 12


def test_mm_quote_curve_is_symmetric_at_zero_and_skews_when_long(mm):
    qc = mm.quote_curve.set_index("book_vega")
    zero = qc.loc[0.0]
    assert zero["A-S skew ask"] == pytest.approx(-zero["A-S skew bid"])
    assert zero["No skew ask"] == pytest.approx(zero["A-S skew ask"])
    long = qc.iloc[-1]
    assert long["A-S skew ask"] < zero["A-S skew ask"]  # long vega: lower the vols
    assert long["No skew ask"] == pytest.approx(zero["No skew ask"])


def test_mm_seed_reproduces():
    cfg = MMConfig(n_paths=10, days=2, seed=4)
    a = run_mm_lab(cfg, synthetic_surface(), 580, 0.04, 0.013)
    b = run_mm_lab(cfg, synthetic_surface(), 580, 0.04, 0.013)
    np.testing.assert_array_equal(a.results["A-S skew"].total, b.results["A-S skew"].total)
