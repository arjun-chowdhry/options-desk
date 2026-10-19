"""Plotly figures for the dashboard. Drawing only: every number arrives precomputed from a
tested module (``sim.lab``, ``pricing.portfolio``); nothing here is computed from scratch.

Colors follow the project palette: blue = model / answer key / the first strategy, orange =
simulation / the second strategy. The scenario heatmap uses a diverging red (loss) to blue
(gain) scale with a neutral gray at zero P&L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optdesk.sim.lab import HedgeLab, MMLab

BLUE, ORANGE = "#2a78d6", "#eb6834"
MUTED = "#8a8984"
STRATEGY_COLORS = {"No skew": BLUE, "A-S skew": ORANGE}
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
BUCKET_LABELS = {"spread": "Spread capture", "hedging": "Hedging", "inventory": "Inventory"}


def _layout(fig: go.Figure, title: str, *, height: int = 380, **kwargs) -> go.Figure:
    layout = {
        "title": {"text": title, "x": 0, "xanchor": "left"},
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 60, "b": 10},
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        "hovermode": "x unified",  # crosshair + one tooltip for line charts; bars override
    }
    fig.update_layout(**{**layout, **kwargs})
    return fig


# --- Hedging -----------------------------------------------------------------------------------


def hedge_histogram(lab: HedgeLab) -> go.Figure:
    s = lab.summary
    fig = go.Figure(
        go.Histogram(
            x=lab.pnl,
            nbinsx=60,
            marker={"color": ORANGE, "line": {"color": "white", "width": 1}},
            name="Simulated P&L",
            hovertemplate="P&L %{x}<br>%{y} paths<extra></extra>",
        )
    )
    fig.add_vline(x=s["mean"], line={"color": ORANGE, "dash": "dash"}, annotation_text="mean")
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    if np.isfinite(s["bs_gap"]):
        fig.add_vline(
            x=s["bs_gap"],
            line={"color": BLUE, "dash": "dot"},
            annotation_text="BS(realized) − BS(hedge)",
            annotation_position="top left",
        )
    return _layout(
        fig,
        f"P&L per option, hedged {lab.config.frequency} ({len(lab.pnl):,} paths)",
        xaxis_title="P&L ($, discounted to today)",
        yaxis_title="Paths",
        hovermode="closest",
        showlegend=False,
    )


def hedge_frequency(lab: HedgeLab) -> go.Figure:
    sw = lab.sweep
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sw["n_rebalances"],
            y=sw["derman_kamal"],
            mode="lines",
            line={"color": BLUE, "width": 2, "dash": "dash"},
            name="Derman-Kamal √(π/4)·vega·σ/√N (fair vol, no costs)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sw["n_rebalances"],
            y=sw["std"],
            mode="lines+markers",
            line={"color": ORANGE, "width": 2},
            marker={"size": 9, "line": {"color": "white", "width": 2}},
            name="Simulated P&L std",
            customdata=np.column_stack([sw["frequency"], sw["mean"], sw["mean_cost"]]),
            hovertemplate="%{customdata[0]}: %{x} hedges<br>std $%{y:.3f}"
            "<br>mean $%{customdata[1]:.3f}<br>cost $%{customdata[2]:.3f}<extra></extra>",
        )
    )
    return _layout(
        fig,
        "Hedging more often shrinks the P&L spread like 1/√N",
        xaxis={"title": "Hedges over the option's life (log)", "type": "log"},
        yaxis={"title": "Std of P&L ($, log)", "type": "log"},
        hovermode="closest",
    )


def hedge_costs(lab: HedgeLab) -> go.Figure:
    sw = lab.sweep
    fig = go.Figure(
        go.Scatter(
            x=sw["n_rebalances"],
            y=sw["mean"],
            mode="lines+markers",
            line={"color": ORANGE, "width": 2},
            marker={"size": 9, "line": {"color": "white", "width": 2}},
            error_y={"type": "data", "array": 2 * sw["se"], "color": MUTED, "thickness": 1},
            name="Mean P&L ± 2 SE",
            customdata=sw["frequency"],
            hovertemplate="%{customdata}: %{x} hedges<br>mean $%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        "Average P&L by frequency: costs grow as you hedge more",
        xaxis={"title": "Hedges over the option's life (log)", "type": "log"},
        yaxis_title="Mean P&L ($)",
        hovermode="closest",
        showlegend=False,
    )


def hedge_paths(lab: HedgeLab) -> go.Figure:
    fig = go.Figure()
    for i, path in enumerate(lab.sample_S):
        fig.add_trace(
            go.Scatter(
                x=lab.days,
                y=path,
                mode="lines",
                line={"color": ORANGE if i == 0 else MUTED, "width": 2 if i == 0 else 1},
                opacity=1.0 if i == 0 else 0.35,
                showlegend=False,
                hoverinfo="skip" if i else None,
                hovertemplate="day %{x:.1f}<br>spot %{y:.2f}<extra>path 1</extra>",
            )
        )
    fig.add_hline(y=lab.config.K, line={"color": BLUE, "dash": "dash"}, annotation_text="strike")
    return _layout(
        fig,
        f"{len(lab.sample_S)} of the simulated spot paths ({lab.config.model.upper()})",
        xaxis_title="Trading day",
        yaxis_title="Spot",
        hovermode="closest",
    )


def hedge_scatter(lab: HedgeLab) -> go.Figure:
    n = min(len(lab.pnl), 3_000)
    x, y = lab.gamma_formula[:n], lab.pnl[:n]
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"color": BLUE, "dash": "dash", "width": 2},
            name="P&L = formula",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={"color": ORANGE, "size": 5, "opacity": 0.5},
            name="One path",
            hovertemplate="formula $%{x:.2f}<br>simulated $%{y:.2f}<extra></extra>",
        )
    )
    return _layout(
        fig,
        "Path by path: P&L vs ½·Γ·S²·(realized var − implied var)·dt",
        xaxis_title="Gamma-P&L formula ($)",
        yaxis_title="Simulated hedged P&L ($)",
        hovermode="closest",
    )


# --- Market maker ------------------------------------------------------------------------------


def mm_attribution(lab: MMLab) -> go.Figure:
    s = lab.summary.set_index("strategy")
    names = list(BUCKET_LABELS.values()) + ["Total"]
    keys = list(BUCKET_LABELS) + ["total"]
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Mean P&L (± 2 SE)", "Std of P&L across paths")
    )
    for label, row in s.iterrows():
        color = STRATEGY_COLORS[label]
        fig.add_trace(
            go.Bar(
                x=names,
                y=[row[f"{k}_mean"] for k in keys],
                error_y={
                    "type": "data",
                    "array": [2 * row[f"{k}_se"] for k in keys],
                    "color": MUTED,
                    "thickness": 1,
                },
                marker={"color": color},
                name=label,
                legendgroup=label,
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=[row[f"{k}_std"] for k in keys],
                marker={"color": color},
                name=label,
                legendgroup=label,
                showlegend=False,
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + label + "</extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(
        fig,
        f"Where the P&L comes from ({lab.config.days} trading days, {lab.config.n_paths} paths)",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        hovermode="closest",
    )


def mm_vega(lab: MMLab) -> go.Figure:
    fig = go.Figure()
    for label, (lo, med, hi) in lab.vega_bands.items():
        color = STRATEGY_COLORS[label]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lab.days, lab.days[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.18,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lab.days,
                y=med,
                mode="lines",
                line={"color": color, "width": 2},
                name=f"{label}: median, 5–95% band",
                legendgroup=label,
                hovertemplate="$%{y:,.0f}",
            )
        )
    for y in (-lab.config.vega_limit, lab.config.vega_limit):
        fig.add_hline(y=y, line={"color": MUTED, "dash": "dash", "width": 1})
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        "Book vega over time (dashed: vega limit)",
        xaxis_title="Trading day",
        yaxis={"title": "$ per vol point", "tickprefix": "$", "tickformat": ",.0f"},
    )


def mm_histogram(lab: MMLab) -> go.Figure:
    fig = go.Figure()
    for label, res in lab.results.items():
        fig.add_trace(
            go.Histogram(
                x=res.total,
                nbinsx=40,
                name=label,
                marker={"color": STRATEGY_COLORS[label]},
                opacity=0.6,
                hovertemplate="$%{x}<br>%{y} paths<extra>" + label + "</extra>",
            )
        )
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        "Distribution of total P&L",
        barmode="overlay",
        xaxis={"title": "Total P&L", "tickprefix": "$", "tickformat": ",.0f"},
        yaxis_title="Paths",
        hovermode="closest",
    )


def mm_cumulative(lab: MMLab, strategy: str) -> go.Figure:
    cum = lab.mean_cum[strategy]
    fig = go.Figure()
    styles = {
        "spread": {"color": BLUE, "dash": "solid"},
        "hedging": {"color": MUTED, "dash": "dot"},
        "inventory": {"color": ORANGE, "dash": "dash"},
    }
    for key, label in BUCKET_LABELS.items():
        fig.add_trace(
            go.Scatter(
                x=lab.days,
                y=cum[key],
                mode="lines",
                line={"width": 2, **styles[key]},
                name=label,
                hovertemplate="$%{y:,.0f}",
            )
        )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        f"Average cumulative P&L by bucket ({strategy})",
        xaxis_title="Trading day",
        yaxis={"tickprefix": "$", "tickformat": ",.0f"},
    )


def mm_quotes(lab: MMLab) -> go.Figure:
    qc = lab.quote_curve
    fig = go.Figure()
    for label, color in STRATEGY_COLORS.items():
        for side, dash in (("ask", "solid"), ("bid", "dash")):
            fig.add_trace(
                go.Scatter(
                    x=qc["book_vega"],
                    y=qc[f"{label} {side}"],
                    mode="lines",
                    line={"color": color, "width": 2, "dash": dash},
                    name=f"{label}: {side}",
                    hovertemplate="%{y:+.2f} vol pts",
                )
            )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        "Quotes vs inventory: long vega → lower both vols",
        xaxis={"title": "Book vega ($ per vol point)", "tickprefix": "$", "tickformat": ",.0f"},
        yaxis_title="Quote minus theo (vol points)",
    )


# --- Portfolio ---------------------------------------------------------------------------------


def scenario_heatmap(grid: pd.DataFrame) -> go.Figure:
    spot = [f"{100 * s:+.1f}%" for s in grid.columns]
    vol = [f"{100 * v:+.0f} pts" for v in grid.index]
    z = grid.to_numpy()
    bound = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=spot,
            y=vol,
            z=z,
            zmin=-bound,
            zmax=bound,
            colorscale=DIVERGING,
            text=[[f"${v:,.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            xgap=2,
            ygap=2,
            colorbar={"title": "P&L", "tickprefix": "$", "tickformat": ",.0f"},
            hovertemplate="spot %{x}, vol %{y}<br>P&L %{text}<extra></extra>",
        )
    )
    return _layout(
        fig,
        "Scenario P&L: full revaluation (red = loss, blue = gain)",
        height=420,
        xaxis_title="Spot shock",
        yaxis_title="Parallel vol shock",
        hovermode="closest",
    )


def scenario_lines(full: pd.Series, approx: pd.Series) -> go.Figure:
    x = 100 * np.asarray(full.index, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=approx.to_numpy(),
            mode="lines",
            line={"color": BLUE, "width": 2, "dash": "dash"},
            name="Greeks: Δ·dS + ½Γ·dS²",
            hovertemplate="$%{y:,.0f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=full.to_numpy(),
            mode="lines",
            line={"color": ORANGE, "width": 2},
            name="Full revaluation",
            hovertemplate="$%{y:,.0f}",
        )
    )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        "P&L vs spot at unchanged vols: where the Greeks stop being enough",
        xaxis={"title": "Spot move (%)", "ticksuffix": "%"},
        yaxis={"tickprefix": "$", "tickformat": ",.0f"},
    )


def greek_bars(risk: pd.DataFrame, column: str, title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=risk["position"],
            y=risk[column],
            marker={"color": [BLUE if v >= 0 else ORANGE for v in risk[column]]},
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    return _layout(
        fig,
        title,
        height=320,
        yaxis={"tickprefix": "$", "tickformat": ",.0f"},
        hovermode="closest",
        showlegend=False,
    )
