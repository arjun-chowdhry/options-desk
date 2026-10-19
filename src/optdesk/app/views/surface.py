"""Surface tab: the fitted SVI surface in 3D, the smile per expiry, fit quality and arbitrage."""

from __future__ import annotations

import streamlit as st

from optdesk.surface.market import MarketData
from optdesk.surface.plot import smile_figure, surface_figure

FIT_COLUMNS = {
    "expiry": st.column_config.DateColumn("Expiry", format="YYYY-MM-DD"),
    "T": st.column_config.NumberColumn("T (years)", format="%.3f"),
    "forward": st.column_config.NumberColumn("Forward", format="%.2f"),
    "a": st.column_config.NumberColumn("a", format="%.5f"),
    "b": st.column_config.NumberColumn("b", format="%.4f"),
    "rho": st.column_config.NumberColumn("ρ", format="%.3f"),
    "m": st.column_config.NumberColumn("m", format="%.4f"),
    "sigma": st.column_config.NumberColumn("σ", format="%.4f"),
    "rmse": st.column_config.NumberColumn("RMSE (vol)", format="%.4f"),
    "n": st.column_config.NumberColumn("Quotes"),
    "butterfly_ok": st.column_config.CheckboxColumn("No butterfly arb"),
    "calendar_ok": st.column_config.CheckboxColumn("No calendar arb"),
}


def render(market: MarketData) -> None:
    fits, clean = market.fits, market.clean
    st.markdown(
        "Each expiry's smile is fitted with **raw SVI** (5 parameters, vega-weighted), then "
        "checked for static arbitrage: butterfly (Gatheral's g(k) ≥ 0, i.e. no negative "
        "densities) and calendar (total variance never falls with maturity)."
    )
    butterfly_ok = bool(fits["butterfly_ok"].all())
    calendar_ok = bool(fits["calendar_ok"].all())
    cols = st.columns(4)
    cols[0].metric("Expiries fitted", len(fits))
    cols[1].metric("Clean OTM quotes", f"{len(clean):,}")
    cols[2].metric("Worst fit RMSE", f"{100 * fits['rmse'].max():.2f} vol pts")
    cols[3].metric(
        "Static arbitrage",
        "None found" if butterfly_ok and calendar_ok else "Flagged",
        help="Butterfly: "
        + ("OK" if butterfly_ok else "violated")
        + "; calendar: "
        + ("OK" if calendar_ok else "violated"),
    )
    if market.synthetic:
        st.caption(
            "Synthetic market: the simulators price on the true SSVI surface. The fit below "
            "is the pipeline (clean → de-Americanize → SVI) recovering it from noisy, "
            "American, penny-rounded quotes."
        )

    st.plotly_chart(surface_figure(fits, clean).update_layout(height=620), key="surface3d")
    st.plotly_chart(smile_figure(fits, clean), key="smiles")
    with st.expander("SVI parameters and arbitrage checks per expiry"):
        st.dataframe(fits[list(FIT_COLUMNS)], column_config=FIT_COLUMNS, hide_index=True)
