"""The M4a hedging study: run it, print the validation tables, and save the charts as PNGs.

The experiment
--------------
A 3-month at-the-money call on a SPY-like stock (spot 100, 4% rate, 1.3% dividend yield),
bought at 20% implied vol and delta-hedged at 20%. Prices are simulated on an hourly grid,
63 trading days x 7 hourly marks = 441 steps, and the *same* 5,000 paths are hedged at every
frequency from monthly (3 hedges) to hourly (441), so differences between frequencies come
from hedging, not from simulation luck. Time is trading time: vol is spread evenly over
trading hours and nights and weekends are skipped, the usual simplification.

Four charts (``docs/images/``):

1. ``hedge_frequency.png``: realized vol = implied vol, no costs. The P&L spread shrinks like
   ``1/sqrt(N)`` on top of the Derman-Kamal line, and the mean stays at zero.
2. ``hedge_pnl_histograms.png``: the P&L distribution at four frequencies, same axis.
3. ``gamma_pnl.png``: realized vol != implied vol. Average P&L vs the gamma formula and vs
   ``BS(realized) - BS(implied)``, then path-by-path scatters (GBM, and Heston stochastic vol).
4. ``hedge_costs.png``: with transaction costs, hedging more often cuts risk but costs more.

Why matplotlib here (Plotly for the surface)
--------------------------------------------
These charts are meant to sit in the README and study notes as static images. Matplotlib
writes PNGs with no extra software; Plotly needs kaleido and a headless Chrome to do the same.
The interactive versions belong in the Streamlit app (M5).

Live inputs (``--ticker SPY --date ...``)
---------------------------------------
The same experiment can take its market from a fitted snapshot instead (``HedgeSetup``): live
spot (strike = spot, so still at the money), the 3-month rate and parity-implied dividend
yield, and the hedge vol read off the fitted SVI surface at 3 months and that strike. Only
the market changes; the grid, frequencies and cost levels stay fixed so the two runs compare
like for like. Two things scale with the market rather than staying at fixed numbers:

- *Realized vols* are the hedge vol plus -10, -5, 0, +5 and +10 points, so "realized =
  implied" is always one of them and the wrong-vol chart stays centred.
- *Heston* starts and mean-reverts to the hedge vol (``v0 = theta = sigma_implied^2``), so its
  vol is random but averages out to the implied vol: the run isolates the effect of vol being
  random, not of it being on average too high or too low.

Dollar P&L scales with spot (the whole problem is homogeneous: double S and K and every
price, hedge and P&L doubles), so at SPY's ~$770 the dollar numbers are ~7.7 times the
spot-100 ones; compare them as a fraction of the premium.

Command line: ``python -m optdesk.sim.plot`` (options: ``--paths``, ``--seed``, ``--out``,
and ``--ticker/--date/--root/--european`` to take the market from a fitted snapshot).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.sim.hedging import (
    delta_hedge,
    discrete_hedging_std,
    frequency_sweep,
    gamma_pnl,
)
from optdesk.sim.paths import gbm_paths, heston_paths

if TYPE_CHECKING:
    from optdesk.surface.market import MarketData

matplotlib.use("Agg")  # draw to files only, never open a window (works on servers and in tests)

__all__ = [
    "HedgeSetup",
    "Study",
    "costs_figure",
    "frequency_figure",
    "gamma_figure",
    "histogram_figure",
    "main",
    "run_study",
]

# --- The experiment ----------------------------------------------------------------------------
S0, K, T, R, Q = 100.0, 100.0, 0.25, 0.04, 0.013
HEDGE_VOL = 0.20  # the implied vol the option is bought at and hedged with
DAYS, MARKS_PER_DAY = 63, 7  # 3 months of trading days, hourly marks 9:30 ... 15:30
N_STEPS = DAYS * MARKS_PER_DAY
FREQUENCIES = {  # label -> grid steps between hedges
    "monthly": 147,
    "biweekly": 70,
    "weekly": 35,
    "every 2 days": 14,
    "daily": 7,
    "every 3 hours": 3,
    "hourly": 1,
}
HISTOGRAM_FREQUENCIES = ["monthly", "weekly", "daily", "hourly"]
REALIZED_OFFSETS = [-0.10, -0.05, 0.0, 0.05, 0.10]  # realized vol minus the hedge vol
SCATTER_OFFSET = 0.05  # realized vol for the path-by-path scatter: hedge vol + 5 points
HESTON = {"kappa": 2.0, "xi": 0.5, "rho": -0.7}  # v0 = theta = hedge vol squared
COSTS_BP = [0, 2, 5]  # half-spread in basis points


@dataclass(frozen=True)
class HedgeSetup:
    """The market the hedged option lives in. Defaults: SPY-like, spot normalized to 100.

    ``from_market`` fills it from a fitted snapshot instead (see "Live inputs" above).
    """

    S0: float = S0
    K: float = K
    r: float = R
    q: float = Q
    hedge_vol: float = HEDGE_VOL
    label: str = "SPY-like stock, spot 100"

    @property
    def realized_vols(self) -> list[float]:
        """Realized vols for the wrong-vol experiment (rounded so the fair one is exact)."""
        return [round(self.hedge_vol + d, 4) for d in REALIZED_OFFSETS]

    @property
    def scatter_vol(self) -> float:
        return round(self.hedge_vol + SCATTER_OFFSET, 4)

    @property
    def heston(self) -> dict[str, float]:
        return {**HESTON, "v0": self.hedge_vol**2, "theta": self.hedge_vol**2}

    @property
    def premium(self) -> float:
        return float(bs_price(self.S0, self.K, T, self.r, self.hedge_vol, self.q))

    @classmethod
    def from_market(cls, market: MarketData) -> HedgeSetup:
        """ATM-spot call at the fitted surface's 3-month vol, with the snapshot's spot, r and q."""
        S = market.spot
        k = float(np.log(S / forward_price(S, T, market.r, market.q)))
        vol = float(market.surface.implied_vol(k, T))
        return cls(S, S, market.r, market.q, vol, f"{market.label}, spot {S:.2f}")


# --- Style (project palette: blue = model/formula, orange = simulation/market) -----------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 9.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
}


@dataclass
class Study:
    """Everything the charts and tables need."""

    sweep: pd.DataFrame  # fair vol, no costs: one row per frequency (frequency_sweep output)
    wrong_vol: pd.DataFrame  # one row per realized vol, hourly hedging
    scatter: dict[str, tuple[np.ndarray, np.ndarray]]  # label -> (gamma formula, P&L) per path
    costs: pd.DataFrame  # one row per (cost, frequency)
    n_paths: int
    setup: HedgeSetup = HedgeSetup()


def run_study(n_paths: int = 5_000, seed: int = 2026, setup: HedgeSetup | None = None) -> Study:
    """Run every experiment of the M4a study on ``n_paths`` paths (reproducible from ``seed``)."""
    m = setup if setup is not None else HedgeSetup()
    S0, K, R, Q, vol = m.S0, m.K, m.r, m.q, m.hedge_vol
    rng = np.random.default_rng(seed)
    fair = gbm_paths(S0, R - Q, vol, T, N_STEPS, n_paths, rng)
    sweep = frequency_sweep(fair, K, R, vol, Q, frequencies=FREQUENCIES)
    sweep["derman_kamal"] = [
        discrete_hedging_std(S0, K, T, R, vol, Q, n) for n in sweep["n_rebalances"]
    ]

    rows, scatter = [], {}
    for realized in m.realized_vols:
        paths = gbm_paths(S0, R - Q, realized, T, N_STEPS, n_paths, rng)
        pnl = delta_hedge(paths, K, R, vol, Q).pnl
        formula = gamma_pnl(paths, K, R, vol, Q)
        rows.append(
            {
                "model": "GBM",
                "realized_vol": realized,
                "sim_mean": pnl.mean(),
                "sim_se": pnl.std(ddof=1) / np.sqrt(n_paths),
                "gamma_formula": formula.mean(),
                "bs_value_gap": float(bs_price(S0, K, T, R, realized, Q)) - m.premium,
                "sim_std": pnl.std(ddof=1),
                "path_corr": np.corrcoef(pnl, formula)[0, 1] if realized != vol else np.nan,
            }
        )
        if realized == m.scatter_vol:
            scatter[f"GBM, realized {realized:.0%}"] = (formula, pnl)

    paths = heston_paths(S0, mu=R - Q, T=T, n_steps=N_STEPS, n_paths=n_paths, rng=rng, **m.heston)
    pnl = delta_hedge(paths, K, R, vol, Q).pnl
    formula = gamma_pnl(paths, K, R, vol, Q)
    rows.append(
        {
            "model": "Heston",
            "realized_vol": np.sqrt(paths.v[:, :-1].mean()),  # average realized vol
            "sim_mean": pnl.mean(),
            "sim_se": pnl.std(ddof=1) / np.sqrt(n_paths),
            "gamma_formula": formula.mean(),
            "bs_value_gap": np.nan,  # no closed form: vol is random
            "sim_std": pnl.std(ddof=1),
            "path_corr": np.corrcoef(pnl, formula)[0, 1],
        }
    )
    scatter["Heston, stochastic vol"] = (formula, pnl)

    cost_tables = []
    for bp in COSTS_BP:
        table = frequency_sweep(fair, K, R, vol, Q, frequencies=FREQUENCIES, cost=bp / 1e4)
        cost_tables.append(table.drop(columns="pnl").assign(cost_bp=bp))
    return Study(
        sweep=sweep,
        wrong_vol=pd.DataFrame(rows),
        scatter=scatter,
        costs=pd.concat(cost_tables, ignore_index=True),
        n_paths=n_paths,
        setup=m,
    )


def _log_x(ax, sweep: pd.DataFrame) -> None:
    """Log x axis with ticks at each frequency's rebalance count."""
    ax.set_xscale("log")
    ax.set_xticks(sweep["n_rebalances"])
    ax.set_xticklabels(sweep["n_rebalances"])
    ax.minorticks_off()
    ax.set_xlabel("Hedges over the option's life (log scale)")


def _dollar_log_y(ax) -> None:
    """Log y axis labelled in plain dollars (0.1, 0.2, 0.5, 1, 2) instead of powers of ten."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def frequency_figure(study: Study) -> Figure:
    """P&L spread vs hedging frequency (with Derman-Kamal), and the mean staying at zero."""
    s = study.sweep
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        left.plot(s["n_rebalances"], s["derman_kamal"], color=BLUE, label="Derman-Kamal formula")
        left.plot(
            s["n_rebalances"],
            s["std"],
            "o",
            color=ORANGE,
            mec=SURFACE,
            mew=1.5,
            label="Simulation (std of P&L)",
        )
        _dollar_log_y(left)
        _log_x(left, s)
        left.set_ylabel("Std of hedged P&L ($, log scale)")
        left.set_title("Spread shrinks like 1/√N")
        for _, row in s.iloc[[0, -1]].iterrows():
            left.annotate(
                f"{row['frequency']}\n${row['std']:.2f}",
                (row["n_rebalances"], row["std"]),
                textcoords="offset points",
                xytext=(8, 4),
                color=INK_2,
            )
        left.legend(loc="lower left")

        right.axhline(0, color=INK_2, lw=1)
        right.errorbar(
            s["n_rebalances"],
            s["mean"],
            yerr=2 * s["se"],
            fmt="o",
            color=ORANGE,
            mec=SURFACE,
            mew=1.5,
            ecolor=ORANGE,
            elinewidth=1.5,
            capsize=0,
            label="Simulation mean ± 2 SE",
        )
        _log_x(right, s)
        lim = 1.15 * float((s["mean"].abs() + 2 * s["se"]).max())
        right.set_ylim(-max(lim, 0.05), max(lim, 0.05))
        right.set_ylabel("Mean hedged P&L ($)")
        right.set_title("Mean stays at zero")
        right.legend(loc="upper right")
        fig.suptitle(
            f"Delta-hedging a 3-month ATM call, realized vol = implied vol = "
            f"{study.setup.hedge_vol:.1%}, no costs ({study.n_paths:,} paths, "
            f"{study.setup.label})",
            x=0.01,
            ha="left",
            color=INK,
            fontsize=12,
        )
        fig.tight_layout()
    return fig


def histogram_figure(study: Study) -> Figure:
    """P&L distribution at four hedging frequencies, small multiples on a shared axis."""
    s = study.sweep.set_index("frequency")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(HISTOGRAM_FREQUENCIES), figsize=(12, 3.4), sharex=True)
        # x range covers 99% of the widest (monthly) distribution; the outer 1% isn't drawn
        lo, hi = np.percentile(s.loc["monthly", "pnl"], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for ax, label in zip(axes, HISTOGRAM_FREQUENCIES, strict=True):
            pnl = s.loc[label, "pnl"]
            ax.hist(pnl, bins=bins, color=BLUE, edgecolor=SURFACE, lw=0.5)
            ax.axvline(0, color=INK_2, lw=1)
            ax.set_title(f"{label} ({s.loc[label, 'n_rebalances']} hedges)")
            ax.text(
                0.97,
                0.95,
                f"std ${pnl.std():.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=INK_2,
            )
            ax.set_yticks([])
            ax.set_xlabel("Hedged P&L ($)")
        fig.suptitle(
            f"Same {study.n_paths:,} paths, hedged more often: the P&L distribution tightens "
            "around zero",
            x=0.01,
            ha="left",
            color=INK,
            fontsize=12,
        )
        fig.tight_layout()
    return fig


def gamma_figure(study: Study) -> Figure:
    """Average P&L vs realized vol against the gamma formula, plus path-by-path scatters."""
    w = study.wrong_vol[study.wrong_vol["model"] == "GBM"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 1 + len(study.scatter), figsize=(13, 4.3))
        ax = axes[0]
        ax.axhline(0, color=INK_2, lw=1)
        x = 100 * w["realized_vol"]
        ax.plot(
            x, w["bs_value_gap"], color=AQUA, lw=6, alpha=0.45, label="BS(realized) − BS(implied)"
        )
        ax.plot(x, w["gamma_formula"], color=BLUE, label="Gamma formula ½ΓS²(σ²ᵣ − σ²ᵢ)")
        ax.errorbar(
            x,
            w["sim_mean"],
            yerr=2 * w["sim_se"],
            fmt="o",
            color=ORANGE,
            mec=SURFACE,
            mew=1.5,
            capsize=0,
            label="Simulated mean P&L ± 2 SE",
        )
        ax.axvline(100 * study.setup.hedge_vol, color=GRID, lw=1, ls="--")
        ax.set_xlabel(f"Realized vol (%), implied = {study.setup.hedge_vol:.1%}")
        ax.set_ylabel("Mean hedged P&L ($), hourly hedging")
        ax.set_title("Long gamma: make realized − implied")
        ax.legend(loc="upper left")

        for ax, (label, (formula, pnl)) in zip(axes[1:], study.scatter.items(), strict=True):
            lo = min(formula.min(), pnl.min())
            hi = max(formula.max(), pnl.max())
            ax.plot([lo, hi], [lo, hi], color=BLUE, lw=1.5, label="y = x")
            ax.scatter(formula, pnl, s=6, color=ORANGE, alpha=0.35, lw=0, label="One path")
            corr = np.corrcoef(formula, pnl)[0, 1]
            ax.set_title(f"{label}: corr {corr:.2f}")
            ax.set_xlabel("Gamma formula on that path ($)")
            ax.set_ylabel("Simulated hedged P&L ($)")
            ax.legend(loc="upper left", markerscale=3)
        fig.suptitle(
            "When realized vol ≠ implied vol, the gamma formula explains the P&L "
            "on average and path by path",
            x=0.01,
            ha="left",
            color=INK,
            fontsize=12,
        )
        fig.tight_layout()
    return fig


def costs_figure(study: Study) -> Figure:
    """With costs: mean P&L by cost level, and the risk-vs-cost trade-off at the highest cost."""
    c = study.costs
    colors = dict(zip(COSTS_BP, [BLUE, ORANGE, AQUA], strict=True))
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        left.axhline(0, color=INK_2, lw=1)
        for bp, group in c.groupby("cost_bp"):
            left.plot(
                group["n_rebalances"],
                group["mean"],
                "-o",
                color=colors[bp],
                mec=SURFACE,
                mew=1.5,
                label="no costs" if bp == 0 else f"{bp} bp half-spread",
            )
        _log_x(left, group)
        left.set_ylabel("Mean hedged P&L ($)")
        left.set_title("Costs drag the mean down, more the more you trade")
        left.legend(loc="lower left")

        top = c[c["cost_bp"] == max(COSTS_BP)]
        right.plot(
            top["n_rebalances"],
            top["std"],
            "-o",
            color=BLUE,
            mec=SURFACE,
            mew=1.5,
            label="Hedging risk (std of P&L)",
        )
        right.plot(
            top["n_rebalances"],
            top["mean_cost"],
            "-o",
            color=ORANGE,
            mec=SURFACE,
            mew=1.5,
            label="Expected cost",
        )
        _log_x(right, top)
        _dollar_log_y(right)
        right.set_ylabel("$ per option (log scale)")
        right.set_title(f"At {max(COSTS_BP)} bp: risk falls like 1/√N, cost rises roughly like √N")
        right.legend(loc="center left")
        fig.suptitle(
            "Transaction costs: why desks don't hedge continuously",
            x=0.01,
            ha="left",
            color=INK,
            fontsize=12,
        )
        fig.tight_layout()
    return fig


def _print_tables(study: Study) -> None:
    fmt = {"float_format": lambda x: f"{x:.4f}", "index": False}
    m = study.setup
    print(f"Market: {m.label}")
    print(
        f"Option: 3-month ATM call, S=K={m.S0:.2f}, r={m.r:.2%}, q={m.q:.2%}, "
        f"vol {m.hedge_vol:.2%}, premium ${m.premium:.4f}"
    )
    print(f"Paths: {study.n_paths:,}, grid {DAYS} days x {MARKS_PER_DAY} hourly marks\n")
    print(f"1) Realized vol = implied vol = {m.hedge_vol:.2%}, no costs")
    cols = ["frequency", "n_rebalances", "mean", "se", "std", "derman_kamal"]
    t = study.sweep[cols].assign(mean_over_se=study.sweep["mean"] / study.sweep["se"])
    print(t.to_string(**fmt))
    slope = np.polyfit(np.log(t["n_rebalances"]), np.log(t["std"]), 1)[0]
    print(f"log-log slope of std vs N: {slope:.3f} (theory -0.5)\n")
    print(f"2) Realized vol != implied vol (hedged hourly at {m.hedge_vol:.2%})")
    print(study.wrong_vol.to_string(**fmt))
    print("\n3) Transaction costs (fair vol)")
    cols = ["cost_bp", "frequency", "n_rebalances", "mean", "std", "mean_cost"]
    print(study.costs[cols].to_string(**fmt))


def main(argv: list[str] | None = None) -> int:
    """Run the hedging study, print the tables and write the four PNGs."""
    parser = argparse.ArgumentParser(description="Delta-hedging study (M4a).")
    parser.add_argument("--paths", type=int, default=5_000, help="number of simulated paths")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--out", type=Path, default=Path("docs/images"), help="output folder")
    parser.add_argument("--ticker", help="take the market from a fitted snapshot (default: 100)")
    parser.add_argument("--date", help="snapshot date YYYY-MM-DD (with --ticker)")
    parser.add_argument("--european", action="store_true", help="options are European (^SPX)")
    parser.add_argument("--root", type=Path, default=None, help="snapshot folder (with --ticker)")
    args = parser.parse_args(argv)

    setup = None
    if args.ticker:
        from optdesk.data.store import DEFAULT_ROOT
        from optdesk.surface.market import snapshot_market

        if not args.date:
            parser.error("--ticker needs --date")
        market = snapshot_market(
            args.ticker, args.date, args.root or DEFAULT_ROOT, american=not args.european
        )
        setup = HedgeSetup.from_market(market)
    study = run_study(args.paths, args.seed, setup)
    _print_tables(study)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = {
        "hedge_frequency.png": frequency_figure(study),
        "hedge_pnl_histograms.png": histogram_figure(study),
        "gamma_pnl.png": gamma_figure(study),
        "hedge_costs.png": costs_figure(study),
    }
    for name, fig in figures.items():
        fig.savefig(args.out / name, dpi=150)
        plt.close(fig)
    print(f"\nwrote {', '.join(str(args.out / n) for n in figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
