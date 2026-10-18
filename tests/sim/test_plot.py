"""Tests for the hedging study and its charts (on a small number of paths, to stay fast).

The numbers themselves are validated in test_hedging.py; here we check the study is wired up
right (every frequency, every realized vol, every cost level, the right answer keys) and that
the command line writes four real PNG files.
"""

import numpy as np
import pytest

from optdesk.sim import plot
from optdesk.sim.plot import main, run_study

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
    assert list(gbm.index) == plot.REALIZED_VOLS
    fair = gbm.loc[plot.HEDGE_VOL]
    assert fair["bs_value_gap"] == 0.0 and abs(fair["gamma_formula"]) < 1e-12
    assert np.all(np.sign(gbm["gamma_formula"]) == np.sign(gbm.index - plot.HEDGE_VOL))
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
