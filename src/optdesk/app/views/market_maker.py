"""Market maker tab: an Avellaneda-Stoikov options desk, skewed vs unskewed, same market."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from optdesk.app import charts
from optdesk.sim.lab import MMConfig, MMLab, run_mm_lab
from optdesk.surface.market import MarketData

SUMMARY_COLUMNS = {
    "spread_mean": "Spread capture",
    "hedging_mean": "Hedging",
    "inventory_mean": "Inventory",
    "total_mean": "Total (mean)",
    "total_std": "Total (std)",
    "mean_over_std": "Mean / std",
    "fills": "Fills",
    "end_vega_std": "End vega std ($/pt)",
    "at_limit": "Steps at a limit",
}


@st.cache_data(show_spinner="Running both desks...", max_entries=20)
def mm_lab(config: MMConfig, market_key: str, _market: MarketData) -> MMLab:
    # market_key identifies the market for the cache; _market itself isn't hashed
    return run_mm_lab(config, _market.surface, _market.spot, _market.r, _market.q)


def render(market: MarketData) -> None:
    st.markdown(
        "A desk quotes 12 SPY options (2, 3 and 6 months; 90-105% strikes) around **theo** "
        "from the surface, takes random customer fills, and delta-hedges every 15 minutes. "
        "What's left is **vega** risk, so it skews quotes by its book vega "
        "(Avellaneda-Stoikov in vol space): long vega → lower both bid and ask vols. Both "
        "desks see the same market and the same customers; only the skew differs."
    )
    with st.form("mm_form"):
        quoting, world, run = st.columns(3)
        with quoting:
            st.markdown("**Quoting**")
            gamma = st.select_slider(
                "Risk aversion γ (per $)",
                [0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                value=5e-5,
                format_func=lambda g: f"{g:g}",
            )
            fill_decay = st.slider("Fill decay k (per 1.00 vol)", 50.0, 500.0, 200.0, 10.0)
            arrival = st.slider("Customer arrivals A (per year, per side)", 200, 3000, 1000, 100)
            trade_size = st.select_slider("Trade size (contracts)", [1, 5, 10, 20, 50], value=10)
        with world:
            st.markdown("**Market**")
            vol_of_vol = st.slider("Vol of vol (vol pts per √year)", 2, 30, 10)
            corr = st.slider("Spot-vol correlation", -0.95, 0.5, -0.7, 0.05)
            hedge_bp = st.slider("Hedge cost (bp)", 0.0, 5.0, 1.0, 0.5)
        with run:
            st.markdown("**Limits and run**")
            vega_limit = st.slider("Vega limit ($k per vol pt)", 5, 100, 25, 5)
            gamma_limit = st.slider("Gamma limit ($k delta per 1%)", 100, 3000, 1000, 100)
            days = st.slider("Trading days", 5, 21, 21)
            n_paths = st.select_slider("Paths", [50, 100, 200, 500], value=200)
            seed = st.number_input("Random seed", min_value=0, value=2026, step=1)
        st.form_submit_button("Run simulation", type="primary")

    config = MMConfig(
        risk_aversion=gamma,
        vol_of_vol=vol_of_vol / 100,
        spot_vol_corr=corr,
        vega_limit=vega_limit * 1_000.0,
        gamma_limit=gamma_limit * 1_000.0,
        trade_size=float(trade_size),
        arrival_rate=float(arrival),
        fill_decay=fill_decay,
        hedge_cost=hedge_bp / 1e4,
        days=days,
        n_paths=n_paths,
        seed=int(seed),
    )
    lab = mm_lab(config, market.key, market)

    s = lab.summary.set_index("strategy")
    no, yes = s.loc["No skew"], s.loc["A-S skew"]
    cols = st.columns(4)
    cols[0].metric("Mean P&L, no skew", f"${no['total_mean']:,.0f}")
    cols[1].metric(
        "Mean P&L, A-S skew",
        f"${yes['total_mean']:,.0f}",
        delta=f"{yes['total_mean'] - no['total_mean']:,.0f} vs no skew",
    )
    cols[2].metric("P&L std, no skew", f"${no['total_std']:,.0f}")
    cols[3].metric(
        "P&L std, A-S skew",
        f"${yes['total_std']:,.0f}",
        delta=f"{yes['total_std'] - no['total_std']:,.0f} vs no skew",
        delta_color="inverse",
    )

    st.plotly_chart(charts.mm_attribution(lab), key="mm_attr")
    left, right = st.columns(2)
    left.plotly_chart(charts.mm_vega(lab), key="mm_vega")
    right.plotly_chart(charts.mm_histogram(lab), key="mm_hist")
    strategy = left.radio("Cumulative P&L for", list(lab.results), index=1, horizontal=True)
    left.plotly_chart(charts.mm_cumulative(lab, strategy), key="mm_cum")
    right.plotly_chart(charts.mm_quotes(lab), key="mm_quotes")

    with st.expander("Summary table and the contracts quoted"):
        st.dataframe(
            s[list(SUMMARY_COLUMNS)].rename(columns=SUMMARY_COLUMNS),
            column_config={
                name: st.column_config.NumberColumn(
                    format="%.3f" if name in ("Mean / std", "Steps at a limit") else "%,.0f"
                )
                for name in SUMMARY_COLUMNS.values()
            },
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"type": "call" if c.call else "put", "strike": c.K, "months": 12 * c.T}
                    for c in lab.contracts
                ]
            ),
            hide_index=True,
            column_config={"months": st.column_config.NumberColumn(format="%.0f")},
        )
