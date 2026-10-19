"""Tests for the hedging study and its charts (on a small number of paths, to stay fast).

The numbers themselves are validated in test_hedging.py; here we check the study is wired up
right (every frequency, every realized vol, every cost level, the right answer keys) and that
the command line writes four real PNG files.
"""

import numpy as np
import pytest

from optdesk.data.store import save_snapshot
from optdesk.pricing.black_scholes import forward_price
from optdesk.sim import plot
from optdesk.sim.plot import HedgeSetup, main, run_study
from optdesk.surface.market import synthetic_market
from optdesk.surface.source import synthetic_surface
from optdesk.surface.synthetic import DIV_YIELD, SPOT, synthetic_chain

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def study():
    return run_study(n_paths=400, seed=1)


def test_sweep_covers_every_frequency_with_derman_kamal(study):
    s = study.sweep
    assert list(s["frequency"]) == list(plot.FREQUENCIES)
    assert s["n_rebalances"].is_monotonic_increasing
    assert s["derman_kamal"].is_monotonic_decreasing
    assert s.iloc[-1]["n_rebalances"] == plot.N_STEPS  # hourly hedges every grid step


def test_wrong_vol_table_has_the_right_answer_keys(study):
    w = study.wrong_vol
    gbm = w[w["model"] == "GBM"].set_index("realized_vol")
    assert list(gbm.index) == study.setup.realized_vols
    fair = gbm.loc[study.setup.hedge_vol]
    assert fair["bs_value_gap"] == 0.0 and abs(fair["gamma_formula"]) < 1e-12
    assert np.all(np.sign(gbm["gamma_formula"]) == np.sign(gbm.index - study.setup.hedge_vol))
    heston = w[w["model"] == "Heston"].iloc[0]
    assert np.isnan(heston["bs_value_gap"])
    assert 0.15 < heston["realized_vol"] < 0.25
    assert set(study.scatter) == {"GBM, realized 25%", "Heston, stochastic vol"}


def test_cost_table_has_every_cost_and_frequency(study):
    c = study.costs
    assert sorted(c["cost_bp"].unique()) == plot.COSTS_BP
    assert len(c) == len(plot.COSTS_BP) * len(plot.FREQUENCIES)
    assert np.all(c.loc[c["cost_bp"] == 0, "mean_cost"] == 0)
    assert np.all(c.loc[c["cost_bp"] > 0, "mean_cost"] > 0)


def test_default_setup_is_the_original_experiment():
    """Spot 100, 20% vol: realized vols 10%...30%, scatter at 25%, Heston at 20% long-run."""
    m = HedgeSetup()
    assert (m.S0, m.K, m.r, m.q, m.hedge_vol) == (100.0, 100.0, 0.04, 0.013, 0.20)
    assert m.realized_vols == [0.10, 0.15, 0.20, 0.25, 0.30]
    assert m.scatter_vol == 0.25
    assert m.heston["v0"] == pytest.approx(0.04) and m.heston["theta"] == pytest.approx(0.04)


def test_pnl_scales_with_spot():
    """The problem is homogeneous: double S and K and every dollar P&L doubles, path by path."""
    base = run_study(n_paths=50, seed=4)
    double = run_study(n_paths=50, seed=4, setup=HedgeSetup(S0=200.0, K=200.0))
    np.testing.assert_allclose(double.sweep["std"], 2 * base.sweep["std"], rtol=1e-9)
    np.testing.assert_allclose(double.sweep["derman_kamal"], 2 * base.sweep["derman_kamal"])
    np.testing.assert_allclose(
        double.wrong_vol["sim_mean"], 2 * base.wrong_vol["sim_mean"], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(double.costs["mean_cost"], 2 * base.costs["mean_cost"], rtol=1e-9)


def test_setup_from_a_market_reads_the_surface():
    """On the synthetic market, the 3-month ATM vol is the true SSVI vol at that strike."""
    market = synthetic_market()
    m = HedgeSetup.from_market(market)
    assert m.S0 == m.K == SPOT
    assert m.q == pytest.approx(DIV_YIELD)
    k = np.log(SPOT / forward_price(SPOT, 0.25, m.r, m.q))
    assert m.hedge_vol == pytest.approx(float(synthetic_surface().implied_vol(k, 0.25)), abs=1e-3)
    assert m.realized_vols[2] == round(m.hedge_vol, 4)


def test_seed_reproduces_the_study():
    a, b = run_study(n_paths=50, seed=7), run_study(n_paths=50, seed=7)
    np.testing.assert_array_equal(a.sweep["std"], b.sweep["std"])
    np.testing.assert_array_equal(a.wrong_vol["sim_mean"], b.wrong_vol["sim_mean"])


def test_main_writes_four_pngs(tmp_path, capsys):
    assert main(["--paths", "200", "--seed", "3", "--out", str(tmp_path)]) == 0
    names = {
        "hedge_frequency.png",
        "hedge_pnl_histograms.png",
        "gamma_pnl.png",
        "hedge_costs.png",
    }
    assert {p.name for p in tmp_path.iterdir()} == names
    for name in names:
        assert (tmp_path / name).read_bytes()[:8] == PNG_SIGNATURE
    out = capsys.readouterr().out
    assert "Realized vol = implied vol" in out and "Transaction costs" in out


def test_main_runs_on_a_saved_snapshot(tmp_path, capsys):
    market = synthetic_chain(american=False, noise=0.0, tick=None)
    save_snapshot(market.raw, market.curve, tmp_path / "snaps")
    args = ["--ticker", "SPY", "--date", "2026-10-16", "--european"]
    args += ["--root", str(tmp_path / "snaps"), "--paths", "20", "--out", str(tmp_path / "img")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Market: SPY 2026-10-16, spot 580.00" in out
    assert len(list((tmp_path / "img").iterdir())) == 4
