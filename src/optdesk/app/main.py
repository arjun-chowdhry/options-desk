"""Options desk dashboard: ``streamlit run src/optdesk/app/main.py``.

Four tabs (Surface, Hedging simulator, Market maker, Portfolio Greeks) on one market. The
market comes from the sidebar's data-source dropdown: the synthetic SPY market, or any
snapshot saved under ``data/snapshots/`` (set ``OPTDESK_SNAPSHOT_ROOT`` to look elsewhere).
New snapshots show up in the dropdown on their own; no code changes.

Hosted (Streamlit Community Cloud), there are no saved snapshots: ``data/snapshots/`` is
gitignored. The app then opens on the demo snapshot committed in ``data/demo/`` (set
``OPTDESK_DEMO_ROOT`` to look elsewhere), and it also falls back to the demo if the chosen
source fails to load. A banner says so whenever the demo is showing.

This file and ``views/`` only lay out widgets and call tested modules
(``surface.market``, ``sim.lab``, ``pricing.portfolio``).

Keeping it light on a small server
----------------------------------
The free hosting tier has one shared CPU and about 1 GB of memory, so the app avoids work:

- **The SVI fits are cached with ``st.cache_data``** (``fit_quotes``, the slow step: cleaning,
  de-Americanizing, fitting). Data caching stores plain DataFrames and hands each caller a
  copy, so no session can alter another's fits; ``max_entries`` caps memory. The assembled
  market (fits plus surface objects, cheap to build) sits on top in ``st.cache_resource``.
- **Only the open tab runs.** ``st.tabs(..., on_change="rerun")`` reruns the script when the
  user switches tabs, and each tab draws only if it is ``.open``. Opening the app runs the
  Surface tab alone (no market-making simulation, 2 charts sent instead of 16). Widgets keep
  their values while their tab is closed (``persist_state="page"``).
- Each simulation is cached once per set of parameters (``st.cache_data`` in ``views/``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit puts this file's folder on sys.path, not the repo's src/. requirements.txt
# installs the package (`-e .`), but if a server skips that, find optdesk next to this file.
try:
    import optdesk  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from optdesk.app.views import hedging, market_maker, portfolio, surface  # noqa: E402
from optdesk.data.store import DEFAULT_ROOT  # noqa: E402
from optdesk.surface.market import (  # noqa: E402
    LoadedMarket,
    MarketSource,
    available_sources,
    fit_quotes,
    load_market_or_demo,
)

ROOT = Path(os.environ.get("OPTDESK_SNAPSHOT_ROOT", DEFAULT_ROOT))
# Anchored to this file, not the working directory: repo/src/optdesk/app/main.py -> repo/data/demo
DEMO_ROOT = Path(
    os.environ.get("OPTDESK_DEMO_ROOT", Path(__file__).resolve().parents[3] / "data" / "demo")
)
TABS = {
    "Surface": surface,
    "Hedging simulator": hedging,
    "Market maker": market_maker,
    "Portfolio Greeks": portfolio,
}


@st.cache_data(
    show_spinner="Cleaning quotes and fitting SVI (first load only, under a minute)...",
    max_entries=4,
)
def cached_fit(raw, curve, american: bool):
    return fit_quotes(raw, curve, american)


@st.cache_resource(max_entries=4)
def get_market(
    kind: str, ticker: str | None, date: str | None, root: str, demo_root: str, american: bool
) -> LoadedMarket:
    source = MarketSource(kind, ticker, date)
    return load_market_or_demo(source, root, american=american, demo_root=demo_root, fit=cached_fit)


def sidebar() -> LoadedMarket:
    st.sidebar.title("Options desk")
    sources = available_sources(ROOT, DEMO_ROOT)
    source = st.sidebar.selectbox(
        "Data source",
        sources,
        format_func=lambda s: s.label,
        help="Saved snapshots (newest first) and the synthetic market. With no saved "
        "snapshots, a demo SPY snapshot shipped with the repo. Save one with "
        "`python -m optdesk.data.fetch SPY` and it appears here.",
    )
    american = True
    if source.kind in ("snapshot", "demo"):
        american = st.sidebar.checkbox(
            "American exercise (strip early-exercise premium)",
            value=True,
            help="On for SPY and single stocks; off for European index options like ^SPX.",
        )
    loaded = get_market(
        source.kind, source.ticker, source.date, str(ROOT), str(DEMO_ROOT), american
    )
    market = loaded.market

    st.sidebar.metric("Spot", f"{market.spot:,.2f}")
    left, right = st.sidebar.columns(2)
    left.metric("Rate r", f"{100 * market.r:.2f}%")
    right.metric("Div yield q", f"{100 * market.q:.2f}%")
    left.metric("1m ATM vol", f"{100 * market.atm_vol(1 / 12):.1f}%")
    right.metric("1y ATM vol", f"{100 * market.atm_vol(1.0):.1f}%")
    if market.synthetic:
        st.sidebar.info(
            "Synthetic market: a known SSVI surface, so every result can be checked against "
            "the answer. Switch to a saved snapshot when one exists."
        )
    st.sidebar.caption(
        "Every number comes from a tested module in `optdesk`; the app itself holds no math."
    )
    return loaded


def main() -> None:
    st.set_page_config(page_title="Options desk", page_icon="📈", layout="wide")
    loaded = sidebar()
    market = loaded.market
    st.title(f"Options desk: {market.label}")
    if loaded.error:
        st.warning(f"Couldn't load {loaded.error}. Showing the demo snapshot instead.")
    if loaded.demo_date:
        st.info(f"Demo snapshot from {loaded.demo_date}")
    # Lazy tabs: switching tabs reruns the script, and only the open tab's view runs.
    tabs = st.tabs(list(TABS), key="tab", on_change="rerun")
    for tab, view in zip(tabs, TABS.values(), strict=True):
        with tab:
            if tab.open:
                view.render(market)


main()
