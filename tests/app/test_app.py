"""Smoke tests for the Streamlit dashboard, run headless with ``streamlit.testing.v1.AppTest``.

The math is tested elsewhere; here we check that the app runs end to end without errors on
the synthetic market, that every tab renders its charts, that the widgets drive the
simulations, and that a saved snapshot shows up in the data-source dropdown and loads.
Network access is blocked, as in the other suites.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from optdesk.data.store import save_snapshot
from optdesk.surface.synthetic import synthetic_chain

MAIN = Path(__file__).parents[2] / "src" / "optdesk" / "app" / "main.py"
TIMEOUT = 180  # seconds; the first run builds and fits the synthetic market


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No network, and an empty snapshot folder unless a test saves one.

    Loopback stays open: on Windows, asyncio (inside Streamlit) wires its event loop with a
    socket connected to 127.0.0.1. Anything else, e.g. yfinance, is refused.
    """
    connect = socket.socket.connect

    def local_only(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
            return connect(sock, address, *args, **kwargs)
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setenv("OPTDESK_SNAPSHOT_ROOT", str(tmp_path / "snapshots"))


@pytest.fixture(scope="module")
def app():
    """One app run shared by the read-only checks (it takes a while to build)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPTDESK_SNAPSHOT_ROOT", str(Path("does-not-exist")))
        at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
        at.run()
    return at


def _texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def test_app_runs_without_errors_on_the_synthetic_market(app):
    assert not app.exception, app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == [
        "Surface",
        "Hedging simulator",
        "Market maker",
        "Portfolio Greeks",
    ]
    assert app.title[0].value == "Options desk: Synthetic SPY"
    assert app.sidebar.selectbox[0].value.label == "Synthetic SPY (SSVI)"


def test_every_tab_draws_its_charts(app):
    # 2 surface + 5 hedging + 5 market maker + 4 portfolio
    charts = app.get("plotly_chart")
    assert len(charts) == 16
    labels = {m.label for m in app.metric}
    for expected in (
        "Static arbitrage",
        "Mean P&L",
        "Derman-Kamal std",
        "P&L std, A-S skew",
        "Vega",
        "Theta",
    ):
        assert expected in labels


def test_surface_metrics_show_a_clean_fit(app):
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Static arbitrage"] == "None found"
    assert metrics["Expiries fitted"] == "8"


def test_hedging_widgets_drive_the_simulation():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    before = {m.label: m.value for m in at.metric}["Mean P&L"]
    model = next(r for r in at.radio if r.label == "Price model")
    model.set_value("Heston (stochastic vol)")
    next(s for s in at.select_slider if s.label == "Paths").set_value(500)
    next(b for b in at.button if b.label == "Run simulation").click()
    at.run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["BS(realized) − BS(implied)"] == "n/a (Heston)"
    assert metrics["Mean P&L"] != before


def test_portfolio_delta_hedge_flattens_delta():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    preset = next(s for s in at.selectbox if s.label == "Start from")
    preset.set_value("3m risk reversal (long 105% call, short 95% put)")
    at.run()
    unhedged = {m.label: m.value for m in at.metric}["Delta"]
    next(c for c in at.checkbox if c.label == "Delta-hedge with shares").check()
    at.run()
    assert not at.exception, at.exception
    hedged = {m.label: m.value for m in at.metric}["Delta"]
    assert unhedged != "0 sh" and hedged in ("0 sh", "-0 sh")


def test_a_saved_snapshot_appears_and_loads(tmp_path):
    market = synthetic_chain(american=False, noise=0.0, tick=None)
    save_snapshot(market.raw, market.curve, tmp_path / "snapshots")
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    source = at.sidebar.selectbox[0]
    assert source.options == ["Synthetic SPY (SSVI)", "SPY 2026-10-16"]  # as displayed
    source.select_index(1)
    at.run()
    next(c for c in at.sidebar.checkbox if c.label.startswith("American")).uncheck()
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Options desk: SPY 2026-10-16"
    assert {m.label: m.value for m in at.metric}["Static arbitrage"] == "None found"
