"""Tests for the market-making study and its charts (few paths, to stay fast).

The numbers are validated in test_market_maker.py; here we check the study is wired up
(both strategies on the same market, every risk aversion, the listed contracts), that the
command line writes four real PNGs, and that a saved snapshot can replace the synthetic
surface.
"""

import numpy as np
import pytest

from optdesk.data.store import save_snapshot
from optdesk.sim import mm_plot
from optdesk.sim.mm_plot import main, run_study, study_contracts
from optdesk.surface.synthetic import synthetic_chain

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FIGURES = {"mm_attribution.png", "mm_inventory.png", "mm_path.png", "mm_frontier.png"}


@pytest.fixture(scope="module")
def study():
    return run_study(n_paths=40, seed=1)


def test_contracts_are_otm_puts_and_calls():
    contracts = study_contracts(580.0, 0.04, 0.013)
    assert len(contracts) == len(mm_plot.TENORS) * len(mm_plot.MONEYNESS)
    for c in contracts:
        F = 580.0 * np.exp((0.04 - 0.013) * c.T)
        assert c.call == (c.K >= F)
        assert c.K % 5 == 0


def test_study_has_both_strategies_and_the_sweep(study):
    assert list(study.summary["strategy"]) == list(mm_plot.STRATEGIES)
    assert list(study.sweep["risk_aversion"]) == mm_plot.RISK_AVERSIONS
    s = study.summary.set_index("strategy")
    total = s["spread_mean"] + s["hedging_mean"] + s["inventory_mean"]
    np.testing.assert_allclose(total, s["total_mean"], rtol=1e-9)
    # same customers and market for both desks: skewing only changes the quotes
    no, yes = study.results["No skew"], study.results["A-S skew"]
    assert yes.vega[:, -1].std() < no.vega[:, -1].std()


def test_seed_reproduces_the_study():
    a, b = run_study(n_paths=10, seed=7), run_study(n_paths=10, seed=7)
    np.testing.assert_array_equal(a.summary["total_mean"], b.summary["total_mean"])


def test_main_writes_four_pngs(tmp_path, capsys):
    assert main(["--paths", "20", "--seed", "3", "--out", str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.iterdir()} == FIGURES
    for name in FIGURES:
        assert (tmp_path / name).read_bytes()[:8] == PNG_SIGNATURE
    out = capsys.readouterr().out
    assert "P&L by bucket" in out and "Risk-aversion sweep" in out


def test_main_runs_on_a_saved_snapshot_surface(tmp_path, capsys):
    market = synthetic_chain(american=False, noise=0.0, tick=None)
    save_snapshot(market.raw, market.curve, tmp_path / "snaps")
    args = ["--ticker", "SPY", "--date", "2026-10-16", "--european"]
    args += ["--root", str(tmp_path / "snaps"), "--paths", "10", "--out", str(tmp_path / "img")]
    assert main(args) == 0
    assert "SPY 2026-10-16 SVI fit" in capsys.readouterr().out
    assert {p.name for p in (tmp_path / "img").iterdir()} == FIGURES
