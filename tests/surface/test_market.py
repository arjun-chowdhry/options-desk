"""Tests for the swappable market sources (synthetic market and saved snapshots).

- The synthetic market prices on the true SSVI surface, with the known spot and dividend
  yield, and its displayed fit recovers that surface closely and arbitrage-free.
- A snapshot of exact European quotes, saved and loaded back, recovers the same market:
  spot, the rate on the curve, the dividend yield implied by the parity forward, and a
  fitted surface matching the truth.
- ``available_sources`` lists the synthetic market first, then every saved snapshot; with no
  saved snapshots, the committed demo snapshot comes first.
- The demo snapshot is small, fits cleanly, and is what a failed source falls back to.
"""

from pathlib import Path

import numpy as np
import pytest

from optdesk.data.store import save_snapshot
from optdesk.surface.market import (
    SYNTHETIC,
    MarketSource,
    available_sources,
    carry_from_fits,
    demo_source,
    load_market,
    load_market_or_demo,
    snapshot_market,
    synthetic_market,
)
from optdesk.surface.source import SSVISurface, SVISurface, synthetic_surface
from optdesk.surface.synthetic import DIV_YIELD, SPOT

K_GRID = np.linspace(-0.15, 0.08, 12)


@pytest.fixture(scope="module")
def synthetic():
    return synthetic_market()


@pytest.fixture
def snapshot_root(tmp_path, exact_market):
    save_snapshot(exact_market.raw, exact_market.curve, tmp_path)
    return tmp_path


def test_synthetic_market_prices_on_the_truth(synthetic):
    assert isinstance(synthetic.surface, SSVISurface)
    assert synthetic.synthetic and synthetic.spot == SPOT and synthetic.q == DIV_YIELD
    assert synthetic.r == pytest.approx(float(synthetic.curve(0.25)))
    assert synthetic.atm_vol(0.5) == pytest.approx(float(synthetic_surface().implied_vol(0, 0.5)))
    # the displayed fit (from noisy, American, rounded quotes) recovers the truth
    assert synthetic.fits["rmse"].max() < 0.005
    assert synthetic.fits["butterfly_ok"].all() and synthetic.fits["calendar_ok"].all()
    assert len(synthetic.clean) > 300


def test_vol_at_reads_the_smile_at_log_moneyness(synthetic):
    T, K = 0.25, 540.0
    F = SPOT * np.exp((synthetic.r - synthetic.q) * T)
    assert synthetic.vol_at(K, T) == pytest.approx(
        float(synthetic.surface.implied_vol(np.log(K / F), T))
    )
    assert synthetic.vol_at(F, T) == pytest.approx(synthetic.atm_vol(T))


def test_snapshot_recovers_spot_rates_and_surface(snapshot_root, exact_market):
    m = snapshot_market("SPY", "2026-10-16", snapshot_root, american=False)
    assert isinstance(m.surface, SVISurface) and not m.synthetic and not m.american
    assert m.label == "SPY 2026-10-16" and m.key == "SPY 2026-10-16|european"
    assert m.spot == SPOT
    T = m.fits["T"].iloc[(m.fits["T"] - 0.25).abs().argmin()]
    assert m.r == pytest.approx(float(exact_market.curve(T)), abs=1e-12)
    assert m.q == pytest.approx(DIV_YIELD, abs=2e-4)  # backed out of the parity forward
    truth = synthetic_surface()
    for t in m.fits["T"]:
        np.testing.assert_allclose(
            m.surface.implied_vol(K_GRID, t), truth.implied_vol(K_GRID, t), atol=2e-3
        )


def test_carry_from_fits_inverts_the_forward():
    import pandas as pd

    from optdesk.data.rates import RateCurve

    curve = RateCurve.flat(0.03)
    fits = pd.DataFrame({"T": [0.1, 0.26, 0.9], "forward": [0, 100 * np.exp(0.02 * 0.26), 0]})
    r, q = carry_from_fits(fits, 100.0, curve)
    assert r == pytest.approx(0.03) and q == pytest.approx(0.01)


def test_available_sources_lists_synthetic_then_snapshots(snapshot_root, tmp_path):
    assert available_sources(tmp_path / "missing") == [SYNTHETIC]
    sources = available_sources(snapshot_root)
    assert sources[0] == SYNTHETIC
    assert sources[1:] == [MarketSource("snapshot", "SPY", "2026-10-16")]
    assert sources[1].label == "SPY 2026-10-16"


def test_load_market_dispatches(snapshot_root):
    m = load_market(MarketSource("snapshot", "SPY", "2026-10-16"), snapshot_root, american=False)
    assert m.label == "SPY 2026-10-16"
    with pytest.raises(ValueError, match="unknown"):
        load_market(MarketSource("crystal-ball"))


# --- demo snapshot (data/demo/, committed for hosting) ------------------------------------

DEMO_ROOT = Path(__file__).parents[2] / "data" / "demo"
DEMO = MarketSource("demo", "SPY", "2026-10-22")


@pytest.fixture(scope="module")
def demo_market():
    return load_market(DEMO, demo_root=DEMO_ROOT)


def test_demo_snapshot_is_committed_small_and_fits(demo_market):
    files = list(DEMO_ROOT.glob("*/*.parquet"))
    assert len(files) == 1 and files[0].stat().st_size < 1_000_000  # small enough for git
    assert demo_source(DEMO_ROOT) == DEMO
    assert demo_market.label == "SPY 2026-10-22" and not demo_market.synthetic
    fits = demo_market.fits
    assert len(fits) >= 20 and fits["T"].max() > 1.0  # covers the 1y ATM vol the app shows
    assert fits["rmse"].max() < 0.005  # under half a vol point on every slice
    assert 0.0 < demo_market.r < 0.10 and -0.02 < demo_market.q < 0.05
    assert 0.05 < demo_market.atm_vol(1 / 12) < 0.60


def test_demo_is_listed_first_only_without_saved_snapshots(snapshot_root, tmp_path):
    assert available_sources(tmp_path / "missing", DEMO_ROOT) == [DEMO, SYNTHETIC]
    assert DEMO.label == "Demo: SPY 2026-10-22"
    assert available_sources(snapshot_root, DEMO_ROOT)[0] == SYNTHETIC  # local data wins
    assert DEMO not in available_sources(snapshot_root, DEMO_ROOT)
    assert available_sources(tmp_path / "missing", tmp_path / "no-demo") == [SYNTHETIC]
    assert demo_source(None) is None


def test_failed_source_falls_back_to_the_demo(tmp_path, demo_market):
    corrupt = tmp_path / "SPY" / "2026-10-01.parquet"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"not a parquet file")
    broken = MarketSource("snapshot", "SPY", "2026-10-01")
    loaded = load_market_or_demo(broken, tmp_path, demo_root=DEMO_ROOT)
    assert loaded.demo_date == "2026-10-22"
    assert loaded.error.startswith("SPY 2026-10-01: ")
    assert loaded.market.spot == demo_market.spot
    with pytest.raises(Exception):  # noqa: B017 - no demo to fall back to: the error surfaces
        load_market_or_demo(broken, tmp_path, demo_root=tmp_path / "no-demo")


def test_successful_loads_report_no_fallback(snapshot_root):
    ok = load_market_or_demo(
        MarketSource("snapshot", "SPY", "2026-10-16"),
        snapshot_root,
        american=False,
        demo_root=DEMO_ROOT,
    )
    assert ok.demo_date is None and ok.error is None
    demo = load_market_or_demo(DEMO, snapshot_root, demo_root=DEMO_ROOT)
    assert demo.demo_date == "2026-10-22" and demo.error is None
    with pytest.raises(ValueError, match="demo_root"):
        load_market(DEMO)
