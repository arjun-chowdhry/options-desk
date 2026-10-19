"""Portfolio Greeks tab: build a book, see its risk in desk units and its scenario P&L."""

from __future__ import annotations

import numpy as np
import streamlit as st

from optdesk.app import charts
from optdesk.pricing.portfolio import (
    PRESETS,
    as_frame,
    book_risk,
    delta_hedge_shares,
    positions_from_frame,
    preset_book,
    scenario_pnl,
    taylor_pnl,
    totals,
)
from optdesk.surface.market import MarketData

SPOT_SHOCKS = np.round(np.arange(-0.10, 0.1001, 0.025), 4)
VOL_SHOCKS = np.array([-0.05, -0.025, 0.0, 0.025, 0.05])
LINE_SHOCKS = np.linspace(-0.15, 0.15, 61)
RISK_COLUMNS = {
    "position": "Position",
    "vol": st.column_config.NumberColumn("Vol", format="%.4f"),
    "price": st.column_config.NumberColumn("Price", format="%.2f"),
    "value": st.column_config.NumberColumn("Value $", format="%,.0f"),
    "delta": st.column_config.NumberColumn("Δ shares", format="%,.1f"),
    "dollar_delta": st.column_config.NumberColumn("$Δ", format="%,.0f"),
    "gamma_1pct": st.column_config.NumberColumn("Γ $ per 1%", format="%,.0f"),
    "vega": st.column_config.NumberColumn("Vega $/pt", format="%,.0f"),
    "theta": st.column_config.NumberColumn("Θ $/day", format="%,.0f"),
    "rho": st.column_config.NumberColumn("ρ $ per 1%", format="%,.0f"),
    "vanna": st.column_config.NumberColumn("Vanna Δ/pt", format="%,.1f"),
    "volga": st.column_config.NumberColumn("Volga $/pt²", format="%,.1f"),
}


def render(market: MarketData) -> None:
    st.markdown(
        "Build a book of options (100 shares per contract). Each option is valued at the "
        "surface's vol for its strike and expiry, unless you type a vol in percent. Greeks "
        "are summed in desk units, and the scenario grid **reprices everything** under spot "
        "and parallel vol shocks, so it stays exact where the Greeks' Taylor expansion "
        "doesn't."
    )
    top = st.columns([2, 1])
    preset = top[0].selectbox("Start from", list(PRESETS))
    hedge = top[1].checkbox("Delta-hedge with shares", value=False)

    book = as_frame(preset_book(preset, market.spot, market.r, market.q))
    edited = st.data_editor(
        book,
        num_rows="dynamic",
        key=f"book::{preset}::{market.key}",  # a new preset or market starts a fresh table
        column_config={
            "type": st.column_config.SelectboxColumn("Type", options=["call", "put"]),
            "strike": st.column_config.NumberColumn("Strike", min_value=0.01, format="%.2f"),
            "days": st.column_config.NumberColumn("Days", min_value=1, step=1),
            "quantity": st.column_config.NumberColumn("Contracts (− = short)", step=1),
            "vol": st.column_config.NumberColumn("Vol % (blank = surface)", min_value=0.1),
        },
        hide_index=True,
    )
    try:
        positions = positions_from_frame(edited)
    except ValueError as err:
        st.error(str(err))
        return
    if not positions:
        st.info("Add a position to see its risk.")
        return

    args = (positions, market.spot, market.r, market.q, market.surface)
    shares = delta_hedge_shares(book_risk(*args)) if hedge else 0.0
    risk = book_risk(*args, shares=shares)
    total = totals(risk)

    cols = st.columns(7)
    cols[0].metric("Book value", f"${total['value']:,.0f}")
    cols[1].metric("Delta", f"{total['delta']:,.0f} sh", help=f"${total['dollar_delta']:,.0f}")
    cols[2].metric("Gamma", f"${total['gamma_1pct']:,.0f}", help="$ of delta per 1% move")
    cols[3].metric("Vega", f"${total['vega']:,.0f}", help="$ per vol point")
    cols[4].metric("Theta", f"${total['theta']:,.0f}", help="$ per calendar day")
    cols[5].metric("Rho", f"${total['rho']:,.0f}", help="$ per 1% rate move")
    cols[6].metric("Vanna", f"{total['vanna']:,.1f} sh", help="delta change per vol point")
    st.dataframe(risk[list(RISK_COLUMNS)], column_config=RISK_COLUMNS, hide_index=True)

    grid = scenario_pnl(*args, SPOT_SHOCKS, VOL_SHOCKS, shares=shares)
    line = scenario_pnl(*args, LINE_SHOCKS, [0.0], shares=shares).iloc[0]
    left, right = st.columns(2)
    left.plotly_chart(charts.scenario_heatmap(grid), key="pf_grid")
    right.plotly_chart(
        charts.scenario_lines(line, taylor_pnl(total, market.spot, LINE_SHOCKS)), key="pf_line"
    )
    left, right = st.columns(2)
    left.plotly_chart(
        charts.greek_bars(risk, "vega", "Vega by position ($ per vol point)"), key="pf_vega"
    )
    right.plotly_chart(
        charts.greek_bars(risk, "gamma_1pct", "Gamma by position ($ delta per 1% move)"),
        key="pf_gamma",
    )
