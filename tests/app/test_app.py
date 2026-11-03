"""Smoke tests for the Streamlit dashboard, run headless with ``streamlit.testing.v1.AppTest``.

The math is tested elsewhere; here we check that the app runs end to end without errors on
the synthetic market, that every tab renders its charts (and only when it is open: tabs are
lazy), that widget values and book edits survive switching tabs, that the widgets drive the
simulations, that a saved snapshot shows up in the data-source dropdown and loads, and that
the committed demo snapshot is used (with a banner) when there are no snapshots or one fails.
Network access is blocked for every test (``tests/conftest.py``); loopback stays open for
Streamlit's event loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from optdesk.data.store import save_snapshot
from optdesk.pricing.portfolio import PRESETS
from optdesk.surface.synthetic import synthetic_chain

MAIN = Path(__file__).parents[2] / "src" / "optdesk" / "app" / "main.py"
DEMO_ROOT = Path(__file__).parents[2] / "data" / "demo"
TIMEOUT = 180  # seconds; the first run builds and fits the synthetic market


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """An empty snapshot folder unless a test saves one (the network is blocked globally),
    and no demo snapshot unless a test points at it, so the default stays synthetic."""
    monkeypatch.setenv("OPTDESK_SNAPSHOT_ROOT", str(tmp_path / "snapshots"))
    monkeypatch.setenv("OPTDESK_DEMO_ROOT", str(tmp_path / "no-demo"))


@pytest.fixture(scope="module")
def app():
    """One app run shared by the read-only checks (it takes a while to build)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPTDESK_SNAPSHOT_ROOT", str(Path("does-not-exist")))
        mp.setenv("OPTDESK_DEMO_ROOT", str(Path("does-not-exist")))
        at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
        at.run()
    return at


def _open(at: AppTest, tab: str) -> None:
    """Click a tab: the tabs are keyed (``key="tab"``), so their state is the open label."""
    at.session_state["tab"] = tab
    at.run()


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


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
    # lazy tabs: opening the app runs the Surface tab alone
    assert len(app.get("plotly_chart")) == 2
    assert "Mean P&L" not in _metrics(app)


TAB_CONTENT = {  # charts drawn, and a metric only that tab shows
    "Surface": (2, "Static arbitrage"),
    "Hedging simulator": (5, "Derman-Kamal std"),
    "Market maker": (5, "P&L std, A-S skew"),
    "Portfolio Greeks": (4, "Vanna"),
}


def test_each_tab_draws_its_charts_only_when_open():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    for tab, (n_charts, own_metric) in TAB_CONTENT.items():
        _open(at, tab)
        assert not at.exception, at.exception
        assert len(at.get("plotly_chart")) == n_charts, tab
        labels = set(_metrics(at))
        assert own_metric in labels, tab
        others = {m for t, (_, m) in TAB_CONTENT.items() if t != tab}
        assert not labels & others, (tab, labels & others)  # closed tabs didn't run


def test_widget_values_survive_switching_tabs():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    _open(at, "Hedging simulator")
    next(s for s in at.slider if s.label == "Days to expiry").set_value(120)
    next(b for b in at.button if b.label == "Run simulation").click()
    at.run()
    premium = _metrics(at)["Premium"]
    _open(at, "Surface")
    _open(at, "Hedging simulator")
    assert not at.exception, at.exception
    assert next(s for s in at.slider if s.label == "Days to expiry").value == 120
    assert _metrics(at)["Premium"] == premium  # same option, not the 91-day default


def test_book_edits_survive_switching_tabs():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    _open(at, "Portfolio Greeks")
    value = float(_metrics(at)["Book value"].replace("$", "").replace(",", ""))
    last = f"book::{next(iter(PRESETS))}::Synthetic SPY|american::last"
    _open(at, "Surface")
    # what the page stores after a user doubles every position in the table
    book = at.session_state[last].copy()
    book["quantity"] *= 2
    at.session_state[last] = book
    _open(at, "Portfolio Greeks")
    assert not at.exception, at.exception
    doubled = float(_metrics(at)["Book value"].replace("$", "").replace(",", ""))
    assert doubled == pytest.approx(2 * value, abs=1.0)  # the edited book, not the preset


def test_surface_metrics_show_a_clean_fit(app):
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Static arbitrage"] == "None found"
    assert metrics["Expiries fitted"] == "8"


def test_hedging_widgets_drive_the_simulation():
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    _open(at, "Hedging simulator")
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
    _open(at, "Portfolio Greeks")
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


def _infos(at: AppTest) -> list[str]:
    return [i.value for i in at.info]


def test_no_snapshots_opens_on_the_demo_with_a_banner(monkeypatch):
    monkeypatch.setenv("OPTDESK_DEMO_ROOT", str(DEMO_ROOT))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].options == ["Demo: SPY 2026-10-22", "Synthetic SPY (SSVI)"]
    assert at.title[0].value == "Options desk: SPY 2026-10-22"
    assert "Demo snapshot from 2026-10-22" in _infos(at)
    assert not at.warning
    assert len(at.get("plotly_chart")) == 2  # the Surface tab; the others run when opened


def test_a_broken_snapshot_falls_back_to_the_demo(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTDESK_DEMO_ROOT", str(DEMO_ROOT))
    corrupt = tmp_path / "snapshots" / "SPY" / "2026-10-01.parquet"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a parquet file")
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.selectbox[0].select_index(1)  # "SPY 2026-10-01", the broken file
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Options desk: SPY 2026-10-22"
    assert "Demo snapshot from 2026-10-22" in _infos(at)
    assert at.warning[0].value.startswith("Couldn't load SPY 2026-10-01")
