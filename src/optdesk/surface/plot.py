"""Plotly figures for the fitted surface: a 3D vol surface and 2D smiles per expiry.

Design choices
--------------
- **Smiles use log-moneyness** ``k = ln(K/F)``, the coordinate the fit uses, so what you see
  is what was fitted. Vol is shown in percent because that's how it's quoted.
- **The 3D surface uses standardized moneyness** ``z = k / (sigma_ATM sqrt(T))``: how many
  ATM standard deviations a strike is from the forward. A 1-week smile spans only a few
  percent of ``k`` while a 1-year smile spans 50%, so on a ``k`` axis the short expiries
  shrink to slivers. In ``z`` every expiry's quotes cover roughly the same range (about -4
  to +2.5), the surface is a full sheet, and skew is comparable across maturities. Traders
  think the same way when they quote by delta.
- **Nothing is drawn beyond the quotes.** Each slice is cut to the range actually quoted for
  it (plus a small margin). SVI happily extrapolates forever, and a surface painted far
  beyond the data invites reading too much into it.
- **Smiles are small multiples,** one panel per expiry, market quotes as dots and the SVI fit
  as a line. Overlaying eight smiles in one chart would make them unreadable.
- Plotly figures are interactive (hover shows strike, vol and fit error) and save to a single
  HTML file that opens in any browser.

Command line
------------
``python -m optdesk.surface.plot`` fits the synthetic market and writes ``surface.html`` and
``smiles.html`` to ``data/plots/``. Add ``--ticker SPY --date 2026-10-19`` to use a saved
snapshot instead. It also prints the fit and arbitrage table. ``--png docs/images/surface.png``
also saves a static image (matplotlib) for the README.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optdesk.surface.svi import SVIParams

__all__ = ["main", "smile_figure", "surface_figure", "surface_png_figure"]

FIT_COLOR = "#2a78d6"  # categorical slot 1 (blue): the model
MARKET_COLOR = "#eb6834"  # categorical slot 2 (orange): the market
SURFACE_SCALE = [  # one-hue sequential blue, light (low vol) to dark (high vol)
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
TEMPLATE = "plotly_white"  # light background with a recessive grid
K_MARGIN = 0.02  # how far past the quoted strikes each slice is drawn (in k)


def _atm_sd(row: pd.Series) -> float:
    """ATM standard deviation of log returns to expiry, ``sqrt(w(0))``."""
    return math.sqrt(max(float(SVIParams.from_mapping(row).total_variance(0.0)), 1e-12))


def _slice_ranges(fits: pd.DataFrame, clean: pd.DataFrame | None) -> list[tuple[float, float]]:
    """``k`` range to draw for each row of ``fits``: the quoted range, or +-3 ATM std devs."""
    ranges = []
    for _, row in fits.iterrows():
        k = None if clean is None else clean.loc[clean["expiry"] == row["expiry"], "k"]
        if k is not None and len(k):
            ranges.append((k.min() - K_MARGIN, k.max() + K_MARGIN))
        else:
            sd = _atm_sd(row)
            ranges.append((-3 * sd, 3 * sd))
    return ranges


def _surface_grid(fits: pd.DataFrame, clean: pd.DataFrame | None, n_z: int):
    """Sorted fits, the ``z`` axis, ATM std devs, ``k`` grid and vol (%) grid, NaN = not drawn."""
    fits = fits.sort_values("T").reset_index(drop=True)
    sd = np.array([_atm_sd(row) for _, row in fits.iterrows()])
    k_ranges = _slice_ranges(fits, clean)
    z_ranges = [(lo / s, hi / s) for (lo, hi), s in zip(k_ranges, sd, strict=True)]
    z = np.linspace(min(lo for lo, _ in z_ranges), max(hi for _, hi in z_ranges), n_z)
    k = np.outer(sd, z)  # log-moneyness at each grid point (rows = expiries)
    vol = np.full_like(k, np.nan)
    for i, row in fits.iterrows():
        lo, hi = z_ranges[i]
        inside = (z >= lo) & (z <= hi)
        vol[i, inside] = 100 * SVIParams.from_mapping(row).implied_vol(k[i, inside], row["T"])
    return fits, z, sd, k, vol


def surface_figure(
    fits: pd.DataFrame, clean: pd.DataFrame | None = None, n_z: int = 121
) -> go.Figure:
    """3D implied-vol surface (percent) over standardized moneyness and expiry.

    ``fits`` is the output of ``svi.fit_chain``. The x axis is ``z = k / sqrt(w(0))`` (ATM
    standard deviations, see module notes). If ``clean`` (the cleaned chain) is given, each
    slice is drawn only over its quoted range and the quotes are shown as dots.
    """
    fits, z, sd, k, vol = _surface_grid(fits, clean, n_z)
    fig = go.Figure(
        go.Surface(
            x=z,
            y=fits["T"],
            z=vol,
            customdata=k,
            colorscale=SURFACE_SCALE,
            colorbar={"title": "Vol (%)"},
            name="SVI fit",
            hovertemplate="%{x:.2f} ATM sd (k = %{customdata:.3f})<br>T = %{y:.3f}y"
            "<br>vol = %{z:.2f}%<extra>SVI</extra>",
        )
    )
    if clean is not None:
        sd_by_expiry = dict(zip(fits["expiry"], sd, strict=True))
        quotes = clean[clean["expiry"].isin(sd_by_expiry)]
        fig.add_trace(
            go.Scatter3d(
                x=quotes["k"] / quotes["expiry"].map(sd_by_expiry),
                y=quotes["T"],
                z=100 * quotes["iv"],
                mode="markers",
                marker={"size": 2.5, "color": MARKET_COLOR},
                name="Market",
                customdata=quotes["strike"],
                hovertemplate="K = %{customdata}<br>%{x:.2f} ATM sd<br>vol = %{z:.2f}%"
                "<extra>Market</extra>",
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Implied volatility surface (SVI fit per expiry)",
        scene={
            "xaxis_title": "ATM std devs  k / (σ_ATM √T)",
            "yaxis_title": "Expiry T (years)",
            "zaxis_title": "Implied vol (%)",
            "camera": {"eye": {"x": 1.0, "y": -1.9, "z": 0.7}},  # short expiries in front
        },
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def smile_figure(fits: pd.DataFrame, clean: pd.DataFrame, cols: int = 3) -> go.Figure:
    """One panel per expiry: market implied vols (dots) and the SVI fit (line), in percent."""
    fits = fits.sort_values("T").reset_index(drop=True)
    rows = math.ceil(len(fits) / cols)
    titles = [
        f"{row['expiry']:%d %b %Y} ({row['T'] * 365:.0f}d), RMSE {100 * row['rmse']:.2f} pts"
        for _, row in fits.iterrows()
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.06)
    for i, (row, (lo, hi)) in enumerate(
        zip((r for _, r in fits.iterrows()), _slice_ranges(fits, clean), strict=True)
    ):
        r, c = divmod(i, cols)
        quotes = clean[clean["expiry"] == row["expiry"]]
        params = SVIParams.from_mapping(row)
        k = np.linspace(lo, hi, 200)
        first = i == 0  # one legend entry per series, not per panel
        fit_error = 100 * (params.implied_vol(quotes["k"], row["T"]) - quotes["iv"])
        fig.add_trace(
            go.Scatter(
                x=quotes["k"],
                y=100 * quotes["iv"],
                mode="markers",
                marker={"size": 8, "color": MARKET_COLOR, "line": {"width": 1, "color": "white"}},
                name="Market (OTM mid)",
                legendgroup="market",
                showlegend=first,
                customdata=np.column_stack([quotes["strike"], fit_error]),
                hovertemplate="K = %{customdata[0]}<br>vol = %{y:.2f}%"
                "<br>fit - market = %{customdata[1]:+.2f} pts<extra></extra>",
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=k,
                y=100 * params.implied_vol(k, row["T"]),
                mode="lines",
                line={"width": 2, "color": FIT_COLOR},
                name="SVI fit",
                legendgroup="fit",
                showlegend=first,
                hovertemplate="k = %{x:.3f}<br>vol = %{y:.2f}%<extra>SVI</extra>",
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_xaxes(title_text="k = ln(K/F)", row=rows)
    fig.update_yaxes(title_text="Implied vol (%)", col=1)
    fig.update_layout(
        template=TEMPLATE,
        title="Volatility smiles by expiry",
        height=320 * rows,
        legend={"orientation": "h", "y": 1.0 + 0.08 / rows, "x": 0, "yanchor": "bottom"},
        hovermode="closest",
    )
    return fig


def surface_png_figure(fits: pd.DataFrame, clean: pd.DataFrame, n_z: int = 121):
    """Static (matplotlib) version for the README: 3D surface left, every smile overlaid right.

    Same grid and coordinates as ``surface_figure``. Plotly can only write PNGs with kaleido and
    a headless Chrome, so the README image is drawn with matplotlib, like the M4 charts. The
    right panel puts all smiles on one ``z`` axis, light (short expiry) to dark (long), with
    the market quotes as dots, so a reader sees the skew flatten with maturity at a glance.
    """
    from matplotlib import rc_context
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    from optdesk.sim.plot import INK, INK_2, STYLE  # one chart style across the README

    fits, z, sd, _, vol = _surface_grid(fits, clean, n_z)
    cmap = LinearSegmentedColormap.from_list("vol", [c for _, c in SURFACE_SCALE])
    shades = cmap(np.linspace(0.2, 1.0, len(fits)))
    with rc_context(STYLE):
        fig = Figure(figsize=(12, 4.8))
        left = fig.add_subplot(1, 2, 1, projection="3d")
        zz, tt = np.meshgrid(z, fits["T"])
        left.plot_surface(
            zz, tt, np.ma.masked_invalid(vol), cmap=cmap, linewidth=0.2, edgecolor="white"
        )
        left.set_xlabel("ATM std devs  k / (σ_ATM √T)", color=INK_2)
        left.set_ylabel("Expiry T (years)", color=INK_2)
        left.set_zlabel("Implied vol (%)", color=INK_2)
        left.view_init(elev=22, azim=-62)  # short expiries in front, like the Plotly camera
        left.set_title("SVI surface", loc="left")

        right = fig.add_subplot(1, 2, 2)
        for i, row in fits.iterrows():
            quotes = clean[clean["expiry"] == row["expiry"]]
            right.plot(z, vol[i], color=shades[i], label=f"{row['T'] * 365:.0f}d")
            z_quotes = quotes["k"] / sd[i]
            right.plot(z_quotes, 100 * quotes["iv"], "o", ms=3, color=shades[i], alpha=0.7)
        right.set_xlabel("ATM std devs  k / (σ_ATM √T)")
        right.set_ylabel("Implied vol (%)")
        right.set_title("Smiles by expiry: SVI fit (lines) vs quotes (dots)")
        right.legend(title="Expiry", ncols=2, loc="upper right")
        title = "Implied volatility surface, SVI fit per expiry"
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
        fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """Fit a surface (synthetic, or a saved snapshot), print the checks and write the plots."""
    from optdesk.data.clean import clean_chain
    from optdesk.surface.arbitrage import check_surface
    from optdesk.surface.svi import fit_chain

    parser = argparse.ArgumentParser(description="Fit SVI per expiry and plot the surface.")
    parser.add_argument("--ticker", help="use a saved snapshot of this ticker (default: synthetic)")
    parser.add_argument("--date", help="snapshot date YYYY-MM-DD (with --ticker)")
    parser.add_argument("--european", action="store_true", help="options are European (^SPX)")
    parser.add_argument("--root", type=Path, default=None, help="snapshot folder (with --ticker)")
    parser.add_argument("--out", type=Path, default=Path("data/plots"), help="output folder")
    parser.add_argument("--png", type=Path, default=None, help="also save a static PNG here")
    args = parser.parse_args(argv)

    if args.ticker:
        from optdesk.data.store import DEFAULT_ROOT, load_snapshot

        if not args.date:
            parser.error("--ticker needs --date")
        raw, curve = load_snapshot(args.ticker, args.date, args.root or DEFAULT_ROOT)
        label = f"{args.ticker.upper()} {args.date}"
    else:
        from optdesk.surface.synthetic import synthetic_chain

        market = synthetic_chain()
        raw, curve = market.raw, market.curve
        label = "synthetic SPY"

    clean = clean_chain(raw, curve, american=not args.european)
    fits = check_surface(fit_chain(clean))
    shown = fits[["expiry", "T", "forward", "a", "b", "rho", "m", "sigma", "rmse", "n"]]
    print(f"SVI fits for {label} ({len(clean)} quotes):")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nArbitrage checks:")
    checks = fits[["expiry", "butterfly_ok", "min_g", "lee_ok", "calendar_ok"]]
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    surface_figure(fits, clean).write_html(args.out / "surface.html", include_plotlyjs=True)
    smile_figure(fits, clean).write_html(args.out / "smiles.html", include_plotlyjs=True)
    print(f"\nwrote {args.out / 'surface.html'} and {args.out / 'smiles.html'}")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        surface_png_figure(fits, clean).savefig(args.png, dpi=150)
        print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
