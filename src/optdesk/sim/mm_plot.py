"""The M4b market-making study: skewed vs unskewed quotes, P&L attribution, and the charts.

The experiment
--------------
A desk makes markets in 12 SPY options for one month (21 trading days x 26 fifteen-minute
steps): 2-, 3- and 6-month expiries, strikes at 90%, 95%, 100% and 105% of the forward (puts
below the forward, calls at and above it). Theo comes from a swappable surface: the synthetic
SSVI surface by default, or an SVI fit of a saved snapshot (``--ticker SPY --date ...``).
The surface moves in parallel with vol of vol 10 points a year, correlated -0.7 with spot.

Two desks see exactly the same market paths and the same customer random numbers
(common random numbers, as in M4a), so every difference comes from the quoting rule:

- **No skew**: quotes centred on theo, the A-S spread.
- **A-S skew**: the same spread, centred on the reservation vol ``theo - gamma sigma_v^2 H V``.

Both run the same risk limits (vega and gamma). A sweep over risk aversion then traces the
trade-off between average P&L and its spread.

Four charts (``docs/images/``): ``mm_attribution.png`` (P&L by bucket), ``mm_inventory.png``
(book vega over time and the P&L distribution), ``mm_path.png`` (one path: spot, vol, the
book's delta, vega and gamma, and P&L), ``mm_frontier.png`` (how quotes skew, and the
risk-aversion frontier).

Command line: ``python -m optdesk.sim.mm_plot`` (options ``--paths``, ``--seed``, ``--out``,
and ``--ticker/--date/--root/--european`` to use a fitted snapshot surface).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from optdesk.pricing.black_scholes import forward_price
from optdesk.sim.market_maker import (
    Contract,
    MarketPaths,
    MMResult,
    QuoteParams,
    quote_vols,
    run_market_maker,
    simulate_market,
)
from optdesk.sim.plot import BLUE, GRID, INK, INK_2, ORANGE, STYLE, SURFACE
from optdesk.surface.source import VolSurface, synthetic_surface
from optdesk.surface.synthetic import DIV_YIELD, SPOT

matplotlib.use("Agg")

__all__ = [
    "MMStudy",
    "attribution_figure",
    "frontier_figure",
    "inventory_figure",
    "main",
    "path_figure",
    "run_study",
    "study_contracts",
    "summarize",
]

# --- The experiment ----------------------------------------------------------------------------
RATE = 0.04
DAYS, STEPS_PER_DAY = 21, 26  # one month of 15-minute steps (6.5-hour sessions)
HORIZON = 1 / 12
TENORS = [2 / 12, 3 / 12, 6 / 12]
MONEYNESS = [0.90, 0.95, 1.00, 1.05]  # strike / forward
VOL_OF_VOL, SPOT_VOL_CORR = 0.10, -0.7
LIMITS = {"vega_limit": 25_000.0, "gamma_limit": 1_000_000.0}  # $/vol point, $ delta per 1%
BASE = QuoteParams(risk_aversion=5e-5, vol_of_vol=VOL_OF_VOL, **LIMITS)
STRATEGIES = {"No skew": replace(BASE, skew=False), "A-S skew": BASE}
COLORS = {"No skew": BLUE, "A-S skew": ORANGE}
RISK_AVERSIONS = [0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
BUCKETS = {
    "spread": "Spread capture",
    "hedging": "Hedging",
    "inventory": "Inventory",
    "total": "Total",
}
PATH_SHOWN = 0  # the first simulated path, not a cherry-picked one


@dataclass
class MMStudy:
    """Everything the charts and tables need."""

    market: MarketPaths
    contracts: list[Contract]
    surface: VolSurface
    base_vol: float
    results: dict[str, MMResult]  # strategy label -> result
    summary: pd.DataFrame  # one row per strategy
    sweep: pd.DataFrame  # one row per risk aversion (skew on)
    n_paths: int
    surface_label: str


def study_contracts(
    S0: float, r: float, q: float, tenors=TENORS, moneyness=MONEYNESS
) -> list[Contract]:
    """Listed contracts: strikes at ``moneyness x forward`` rounded to $5; puts below F."""
    out = []
    for T in tenors:
        F = float(forward_price(S0, T, r, q))
        for m in moneyness:
            K = float(np.round(m * F / 5) * 5)
            out.append(Contract(K=K, T=T, call=K >= F))
    return out


def summarize(res: MMResult) -> dict[str, float]:
    """Mean, standard error and std of each P&L bucket, plus inventory and activity stats."""
    n = len(res.total)
    row: dict[str, float] = {}
    for key in BUCKETS:
        x = getattr(res, key)
        row[f"{key}_mean"] = x.mean()
        row[f"{key}_se"] = x.std(ddof=1) / np.sqrt(n)
        row[f"{key}_std"] = x.std(ddof=1)
    row["fills"] = res.fills.mean()
    row["hedge_costs"] = res.costs.mean()
    row["end_vega_std"] = res.vega[:, -1].std(ddof=1)
    row["avg_abs_vega"] = np.abs(res.vega).mean()
    row["avg_abs_gamma"] = np.abs(res.gamma).mean()
    row["at_limit"] = res.at_limit.mean()
    row["mean_over_std"] = row["total_mean"] / row["total_std"]
    return row


def run_study(
    n_paths: int = 1_000,
    seed: int = 2026,
    surface: VolSurface | None = None,
    S0: float = SPOT,
    r: float = RATE,
    q: float = DIV_YIELD,
    surface_label: str = "synthetic SSVI surface",
) -> MMStudy:
    """Run both strategies and the risk-aversion sweep on the same ``n_paths`` market paths."""
    surface = surface if surface is not None else synthetic_surface()
    contracts = study_contracts(S0, r, q)
    base_vol = float(surface.implied_vol(0.0, TENORS[0]))  # spot realizes the 2-month ATM vol
    rng = np.random.default_rng(seed)
    market = simulate_market(
        S0,
        r,
        q,
        base_vol,
        HORIZON,
        DAYS * STEPS_PER_DAY,
        n_paths,
        rng,
        vol_of_vol=VOL_OF_VOL,
        spot_vol_corr=SPOT_VOL_CORR,
    )
    fill_seed = int(rng.integers(2**32))  # every run draws customers from the same stream

    def run(params: QuoteParams) -> MMResult:
        return run_market_maker(market, contracts, surface, params, fill_seed)

    results = {label: run(params) for label, params in STRATEGIES.items()}
    summary = pd.DataFrame(
        [{"strategy": label, **summarize(res)} for label, res in results.items()]
    )
    sweep = pd.DataFrame(
        [
            {"risk_aversion": g, **summarize(run(replace(BASE, risk_aversion=g)))}
            for g in RISK_AVERSIONS
        ]
    )
    return MMStudy(
        market=market,
        contracts=contracts,
        surface=surface,
        base_vol=base_vol,
        results=results,
        summary=summary,
        sweep=sweep,
        n_paths=n_paths,
        surface_label=surface_label,
    )


# --- Charts ------------------------------------------------------------------------------------


def _dollars_k(y: float, decimals: int = 0) -> str:
    """$12k / −$9.9k: the sign goes before the dollar sign."""
    sign = "−" if y < 0 else ""
    return f"{sign}${abs(y) / 1e3:,.{decimals}f}k"


_thousands = FuncFormatter(lambda y, _: _dollars_k(y))


def _suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", color=INK, fontsize=12)


def attribution_figure(study: MMStudy) -> Figure:
    """Mean and std of each P&L bucket, grouped by strategy."""
    s = study.summary.set_index("strategy")
    x = np.arange(len(BUCKETS))
    width = 0.38
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
        for j, (label, row) in enumerate(s.iterrows()):
            offset = (j - 0.5) * width
            means = [row[f"{b}_mean"] for b in BUCKETS]
            ses = [2 * row[f"{b}_se"] for b in BUCKETS]
            stds = [row[f"{b}_std"] for b in BUCKETS]
            kw = {"width": width - 0.04, "color": COLORS[label], "label": label}
            left.bar(x + offset, means, yerr=ses, ecolor=INK_2, capsize=0, **kw)
            right.bar(x + offset, stds, **kw)
            for ax, values in ((left, means), (right, stds)):
                for xi, v in zip(x + offset, values, strict=True):
                    ax.annotate(
                        _dollars_k(v, 1 if abs(v) < 1e4 else 0),
                        (xi, max(v, 0)),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        va="bottom",
                        color=INK_2,
                        fontsize=8,
                    )
        for ax in (left, right):
            ax.set_xticks(x, list(BUCKETS.values()))
            ax.yaxis.set_major_formatter(_thousands)
            ax.grid(axis="x", visible=False)
        left.axhline(0, color=INK_2, lw=1)
        left.set_title("Average P&L per month (± 2 SE)")
        left.legend(loc="upper center")
        right.set_title("Std of P&L across paths: skewing cuts inventory risk")
        right.legend(loc="upper left")
        right.set_ylim(0, 1.15 * float(s[[f"{b}_std" for b in BUCKETS]].to_numpy().max()))
        _suptitle(
            fig,
            f"Where a month of market making P&L comes from ({study.n_paths:,} paths, "
            f"{study.surface_label})",
        )
        fig.tight_layout()
    return fig


def _days(study: MMStudy) -> np.ndarray:
    return study.market.times / HORIZON * DAYS


def inventory_figure(study: MMStudy) -> Figure:
    """Book vega over time (median and 5-95% band) and the distribution of total P&L."""
    days = _days(study)
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
        left.axhline(0, color=INK_2, lw=1)
        for label, res in study.results.items():
            lo, med, hi = np.percentile(res.vega, [5, 50, 95], axis=0)
            left.fill_between(days, lo, hi, color=COLORS[label], alpha=0.18, lw=0)
            left.plot(days, med, color=COLORS[label], label=f"{label}: median, 5–95% band")
        limit = BASE.vega_limit
        for y in (-limit, limit):
            left.axhline(y, color=INK_2, lw=1, ls="--")
        left.annotate("vega limit", (days[-1], limit), ha="right", va="bottom", color=INK_2)
        left.set_xlabel("Trading day")
        left.set_ylabel("Book vega ($ per vol point)")
        left.yaxis.set_major_formatter(_thousands)
        left.set_title("Skewing keeps vega inventory near zero")
        left.legend(loc="lower left")

        totals = [res.total for res in study.results.values()]
        bins = np.linspace(*np.percentile(np.concatenate(totals), [0.5, 99.5]), 50)
        for label, res in study.results.items():
            right.hist(
                res.total, bins=bins, histtype="step", lw=2, color=COLORS[label], label=label
            )
            right.axvline(res.total.mean(), color=COLORS[label], lw=1, ls="--")
        right.xaxis.set_major_formatter(_thousands)
        right.set_yticks([])
        right.set_xlabel("Total P&L over the month (dashed: mean)")
        right.set_title("Similar average, much narrower distribution")
        right.legend(loc="upper left")
        _suptitle(fig, f"Inventory and P&L across {study.n_paths:,} paths")
        fig.tight_layout()
    return fig


def path_figure(study: MMStudy, path: int = PATH_SHOWN) -> Figure:
    """One path: spot, ATM vol, delta before each rehedge, book vega and gamma, cumulative P&L."""
    days = _days(study)
    m = study.market
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
        (spot, vol, dlt), (veg, gam, pnl) = axes
        spot.plot(days, m.S[path], color=INK)
        spot.set_title("Spot")
        vol.plot(days, 100 * (study.base_vol + m.vol_shift[path]), color=INK)
        vol.set_title("2-month ATM implied vol (%)")
        for label, res in study.results.items():
            c = COLORS[label]
            dlt.plot(days[:-1], res.delta[path, :-1], color=c, lw=1, alpha=0.8, label=label)
            veg.plot(days, res.vega[path], color=c, label=label)
            gam.plot(days, res.gamma[path], color=c, label=label)
            cum = res.cum_spread + res.cum_hedging + res.cum_inventory
            pnl.plot(days, cum[path], color=c, label=label)
        dlt.set_title("Book delta before each rehedge (shares)")
        veg.set_title("Book vega ($ per vol point)")
        gam.set_title("Book gamma ($ delta per 1% move)")
        pnl.set_title("Cumulative P&L")
        for ax in (veg, gam, pnl):
            ax.yaxis.set_major_formatter(_thousands)
        for ax in (dlt, veg, gam):
            ax.axhline(0, color=INK_2, lw=1)
        for ax in axes[1]:
            ax.set_xlabel("Trading day")
        veg.legend(loc="best")
        _suptitle(fig, "One simulated month (the first path), both desks on the same market")
        fig.tight_layout()
    return fig


def frontier_figure(study: MMStudy) -> Figure:
    """How quotes move with inventory, and mean vs std of P&L as risk aversion grows."""
    p = BASE
    book_vega = np.linspace(-p.vega_limit, p.vega_limit, 101) * 100  # $ per 1.00 vol
    trade_vega = p.trade_size * p.multiplier * 0.95 * 100  # a 10-lot of 2m ATM (~$0.95/pt)
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
        left.axhline(0, color=INK_2, lw=1)
        left.axvline(0, color=GRID, lw=1)
        theo = 0.20
        for label, params in STRATEGIES.items():
            bid, ask = quote_vols(theo, trade_vega, book_vega, params)
            c = COLORS[label]
            left.plot(book_vega / 100, 100 * (ask - theo), color=c, label=f"{label}: ask")
            left.plot(book_vega / 100, 100 * (bid - theo), color=c, ls="--", label=f"{label}: bid")
        left.xaxis.set_major_formatter(_thousands)
        left.set_xlabel("Book vega ($ per vol point)")
        left.set_ylabel("Quote minus theo (vol points)")
        left.set_title("Long vega → lower both vols; short → raise them")
        left.legend(loc="upper right", fontsize=8.5)

        sw = study.sweep
        right.plot(sw["total_std"], sw["total_mean"], "-", color=ORANGE, lw=1.5)
        right.plot(sw["total_std"], sw["total_mean"], "o", color=ORANGE, mec=SURFACE, mew=1.5)
        for _, row in sw.iterrows():
            right.annotate(
                f"γ={row['risk_aversion']:g}",
                (row["total_std"], row["total_mean"]),
                textcoords="offset points",
                xytext=(6, 4),
                color=INK_2,
                fontsize=8.5,
            )
        ns = study.summary.set_index("strategy").loc["No skew"]
        right.plot(ns["total_std"], ns["total_mean"], "o", color=BLUE, mec=SURFACE, mew=1.5)
        right.annotate(
            "No skew",
            (ns["total_std"], ns["total_mean"]),
            textcoords="offset points",
            xytext=(6, -12),
            color=INK_2,
        )
        right.xaxis.set_major_formatter(_thousands)
        right.yaxis.set_major_formatter(_thousands)
        right.set_xlabel("Std of monthly P&L")
        right.set_ylabel("Mean monthly P&L")
        right.set_title("A little skew: same P&L, a third less risk; too much gives up spread")
        _suptitle(fig, "Avellaneda-Stoikov quoting: the skew rule and the risk-return trade-off")
        fig.tight_layout()
    return fig


# --- Command line ------------------------------------------------------------------------------


def _print_tables(study: MMStudy) -> None:
    print(f"Surface: {study.surface_label}; {len(study.contracts)} contracts:")
    print(", ".join(f"{'C' if c.call else 'P'}{c.K:g} {c.T * 12:.0f}m" for c in study.contracts))
    print(f"Paths: {study.n_paths:,}, {DAYS} days x {STEPS_PER_DAY} steps\n")
    cols = {
        "spread_mean": "spread",
        "hedging_mean": "hedging",
        "inventory_mean": "inventory",
        "total_mean": "total",
        "total_se": "total SE",
        "spread_std": "spread std",
        "hedging_std": "hedging std",
        "inventory_std": "inv std",
        "total_std": "total std",
        "mean_over_std": "mean/std",
    }
    s = study.summary.set_index("strategy")
    print("P&L by bucket ($ per month)")
    print(s[list(cols)].rename(columns=cols).to_string(float_format=lambda x: f"{x:,.0f}"))
    print("\nInventory and activity")
    cols2 = ["fills", "hedge_costs", "end_vega_std", "avg_abs_vega", "avg_abs_gamma", "at_limit"]
    print(s[cols2].to_string(float_format=lambda x: f"{x:,.3f}" if abs(x) < 1 else f"{x:,.0f}"))
    print("\nRisk-aversion sweep (A-S skew)")
    cols3 = ["risk_aversion", "spread_mean", "total_mean", "total_std", "end_vega_std", "fills"]
    print(study.sweep[cols3].to_string(index=False, float_format=lambda x: f"{x:,.6g}"))


def _snapshot_surface(args) -> tuple[VolSurface, float, float, float, str]:
    """A fitted SVI surface from a saved snapshot, with its spot, rate and implied dividend."""
    from optdesk.data.clean import clean_chain
    from optdesk.data.store import DEFAULT_ROOT, load_snapshot
    from optdesk.surface.source import fitted_surface
    from optdesk.surface.svi import fit_chain

    raw, curve = load_snapshot(args.ticker, args.date, args.root or DEFAULT_ROOT)
    fits = fit_chain(clean_chain(raw, curve, american=not args.european))
    S0 = float(raw["spot"].iloc[0])
    row = fits.iloc[(fits["T"] - TENORS[1]).abs().argmin()]  # forward nearest 3 months
    r = float(curve(row["T"]))
    q = r - float(np.log(row["forward"] / S0) / row["T"])  # dividend yield implied by F
    return fitted_surface(fits), S0, r, q, f"{args.ticker.upper()} {args.date} SVI fit"


def main(argv: list[str] | None = None) -> int:
    """Run the market-making study, print the tables and write the four PNGs."""
    parser = argparse.ArgumentParser(description="Options market-making study (M4b).")
    parser.add_argument("--paths", type=int, default=1_000, help="number of simulated paths")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--out", type=Path, default=Path("docs/images"), help="output folder")
    parser.add_argument("--ticker", help="use a fitted snapshot surface (default: synthetic)")
    parser.add_argument("--date", help="snapshot date YYYY-MM-DD (with --ticker)")
    parser.add_argument("--european", action="store_true", help="options are European (^SPX)")
    parser.add_argument("--root", type=Path, default=None, help="snapshot folder (with --ticker)")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.ticker:
        if not args.date:
            parser.error("--ticker needs --date")
        surface, S0, r, q, label = _snapshot_surface(args)
        kwargs = {"surface": surface, "S0": S0, "r": r, "q": q, "surface_label": label}
    study = run_study(args.paths, args.seed, **kwargs)
    _print_tables(study)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = {
        "mm_attribution.png": attribution_figure(study),
        "mm_inventory.png": inventory_figure(study),
        "mm_path.png": path_figure(study),
        "mm_frontier.png": frontier_figure(study),
    }
    for name, fig in figures.items():
        fig.savefig(args.out / name, dpi=150)
        plt.close(fig)
    print(f"\nwrote {', '.join(str(args.out / n) for n in figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
