"""Hedging simulator tab: buy or sell an option, delta-hedge it, see what you make."""

from __future__ import annotations

import streamlit as st

from optdesk.app import charts
from optdesk.sim.lab import FREQUENCIES, HedgeConfig, HedgeLab, heston_params, run_hedge_lab
from optdesk.surface.market import MarketData

DAYS_PER_YEAR = 365


@st.cache_data(show_spinner="Simulating and hedging paths...", max_entries=20)
def hedge_lab(config: HedgeConfig) -> HedgeLab:
    return run_hedge_lab(config)


def render(market: MarketData) -> None:
    st.markdown(
        "Buy (or sell) one option at the **implied vol**, then hold the Black-Scholes delta "
        "in shares, rebalancing on a schedule. If the stock moves as much as the vol you paid "
        "for, the hedge replicates the option and P&L is zero, apart from the noise of "
        "hedging at discrete times. If it moves more, long gamma makes money."
    )
    with st.form("hedge_form"):
        option, vols, hedging = st.columns(3)
        with option:
            st.markdown("**Option**")
            kind = st.radio("Type", ["call", "put"], horizontal=True)
            side = st.radio("Position", ["long", "short"], horizontal=True)
            strike = st.number_input(
                "Strike", min_value=1.0, value=float(round(market.spot)), step=1.0
            )
            days = st.slider("Days to expiry", 14, 365, 91)
        with vols:
            st.markdown("**Volatility**")
            hedge_source = st.radio(
                "Implied (hedge) vol",
                ["From the surface", "Custom"],
                horizontal=True,
                help="'From the surface' reads the smile at this strike and expiry.",
            )
            custom_vol = st.slider("Custom implied vol (%)", 5.0, 80.0, 20.0, 0.5)
            model = st.radio(
                "Price model", ["GBM (constant vol)", "Heston (stochastic vol)"], horizontal=True
            )
            realized = st.slider("Realized vol, GBM (%)", 5.0, 80.0, 20.0, 0.5)
            with st.expander("Heston parameters"):
                start_vol = st.slider("Starting vol (%)", 5.0, 60.0, 20.0, 0.5)
                long_vol = st.slider("Long-run vol (%)", 5.0, 60.0, 20.0, 0.5)
                kappa = st.slider("Mean reversion κ (per year)", 0.1, 10.0, 2.0, 0.1)
                xi = st.slider("Vol of variance ξ", 0.05, 1.5, 0.5, 0.05)
                rho = st.slider("Spot-vol correlation ρ", -0.95, 0.95, -0.7, 0.05)
        with hedging:
            st.markdown("**Hedging**")
            frequency = st.select_slider("Rebalance", list(FREQUENCIES), value="daily")
            cost_bp = st.slider("Transaction cost (bp of price, half-spread)", 0.0, 20.0, 0.0)
            n_paths = st.select_slider("Paths", [500, 1_000, 2_000, 5_000], value=2_000)
            seed = st.number_input("Random seed", min_value=0, value=2026, step=1)
        st.form_submit_button("Run simulation", type="primary")

    T = days / DAYS_PER_YEAR
    hedge_vol = market.vol_at(strike, T) if hedge_source == "From the surface" else custom_vol / 100
    config = HedgeConfig(
        model="gbm" if model.startswith("GBM") else "heston",
        S0=market.spot,
        K=strike,
        T=T,
        r=market.r,
        q=market.q,
        call=kind == "call",
        position=1.0 if side == "long" else -1.0,
        hedge_vol=hedge_vol,
        realized_vol=realized / 100,
        heston=heston_params(start_vol / 100, long_vol / 100, kappa, xi, rho),
        cost=cost_bp / 1e4,
        frequency=frequency,
        n_paths=n_paths,
        seed=int(seed),
    )
    try:
        lab = hedge_lab(config)
    except ValueError as err:
        st.error(str(err))
        return

    s = lab.summary
    cols = st.columns(6)
    cols[0].metric("Premium", f"${s['premium']:.2f}", help=f"at {100 * hedge_vol:.2f}% vol")
    cols[1].metric("Mean P&L", f"${s['mean']:.3f}", help=f"± ${2 * s['se']:.3f} (2 SE)")
    cols[2].metric("P&L std", f"${s['std']:.3f}")
    cols[3].metric(
        "Derman-Kamal std",
        f"${s['derman_kamal']:.3f}",
        help="√(π/4)·vega·σ/√N: the expected std when realized = implied and no costs.",
    )
    cols[4].metric(
        "Gamma formula (mean)",
        f"${s['gamma_formula_mean']:.3f}",
        help="Average of ½·Γ·S²·(realized var − implied var)·dt along each path.",
    )
    bs_gap = s["bs_gap"]
    cols[5].metric(
        "BS(realized) − BS(implied)",
        "n/a (Heston)" if bs_gap != bs_gap else f"${bs_gap:.3f}",
        help="The answer key for the mean P&L when spot follows GBM.",
    )
    st.caption(
        f"Implied vol {100 * hedge_vol:.2f}%, realized {100 * s['realized_vol']:.2f}% on "
        f"average; {int(s['n_rebalances'])} hedges; mean cost ${s['mean_cost']:.3f}; "
        f"5th-95th percentile ${s['p5']:.2f} to ${s['p95']:.2f}."
    )

    left, right = st.columns(2)
    left.plotly_chart(charts.hedge_histogram(lab), key="hedge_hist")
    right.plotly_chart(charts.hedge_frequency(lab), key="hedge_freq")
    left.plotly_chart(charts.hedge_scatter(lab), key="hedge_scatter")
    right.plotly_chart(charts.hedge_costs(lab), key="hedge_costs")
    st.plotly_chart(charts.hedge_paths(lab), key="hedge_paths")
    with st.expander("Every frequency on the same paths"):
        st.dataframe(
            lab.sweep.drop(columns=["pnl", "rebalance_every"]),
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ("mean", "se", "std", "mean_cost", "derman_kamal")
            },
        )
