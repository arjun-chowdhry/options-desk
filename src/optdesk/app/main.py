"""Options desk dashboard: ``streamlit run src/optdesk/app/main.py``.

Four tabs (Surface, Hedging simulator, Market maker, Portfolio Greeks) on one market. The
market comes from the sidebar's data-source dropdown: the synthetic SPY market, or any
snapshot saved under ``data/snapshots/`` (set ``OPTDESK_SNAPSHOT_ROOT`` to look elsewhere).
New snapshots show up in the dropdown on their own; no code changes.

This file and ``views/`` only lay out widgets and call tested modules
(``surface.market``, ``sim.lab``, ``pricing.portfolio``). Expensive work is cached: the
market is built once per source, and each simulation once per set of parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from optdesk.app.views import hedging, market_maker, portfolio, surface
from optdesk.data.store import DEFAULT_ROOT
from optdesk.surface.market import MarketData, MarketSource, available_sources, load_market

ROOT = Path(os.environ.get("OPTDESK_SNAPSHOT_ROOT", DEFAULT_ROOT))
TABS = ["Surface", "Hedging simulator", "Market maker", "Portfolio Greeks"]


@st.cache_resource(show_spinner="Building the market: cleaning quotes and fitting SVI...")
def get_market(kind: str, ticker: str | None, date: str | None, root: str, american: bool):
    return load_market(MarketSource(kind, ticker, date), root, american=american)


def sidebar() -> MarketData:
    st.sidebar.title("Options desk")
    sources = available_sources(ROOT)
    source = st.sidebar.selectbox(
        "Data source",
        sources,
        format_func=lambda s: s.label,
        help="Synthetic SPY until live snapshots exist. Save one with "
        "`python -m optdesk.data.fetch SPY` and it appears here.",
    )
    american = True
    if source.kind == "snapshot":
        american = st.sidebar.checkbox(
            "American exercise (strip early-exercise premium)",
            value=True,
            help="On for SPY and single stocks; off for European index options like ^SPX.",
        )
    market = get_market(source.kind, source.ticker, source.date, str(ROOT), american)

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
    return market


def main() -> None:
    st.set_page_config(page_title="Options desk", page_icon="📈", layout="wide")
    market = sidebar()
    st.title(f"Options desk: {market.label}")
    tabs = st.tabs(TABS)
    with tabs[0]:
        surface.render(market)
    with tabs[1]:
        hedging.render(market)
    with tabs[2]:
        market_maker.render(market)
    with tabs[3]:
        portfolio.render(market)


main()
