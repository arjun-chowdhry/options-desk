"""Interactive experiments for the dashboard: the M4 studies with every knob exposed.

Why a separate module
---------------------
``sim/plot.py`` and ``sim/mm_plot.py`` run *fixed* experiments that produce the README
charts. The dashboard needs the same experiments with parameters the user picks: strike,
vols, Heston parameters, costs, risk aversion, limits. This module runs those experiments
through the same tested engines (``paths``, ``hedging``, ``market_maker``) and returns tidy
results. It also precomputes everything a chart needs (percentile bands, average P&L
paths, quote curves). The Streamlit app can then stay free of math: it draws what it's
given.

Two labs
--------
- ``run_hedge_lab``: buy or sell one European option at the *hedge vol* and delta-hedge it
  on GBM or Heston paths. One chosen frequency gives the P&L distribution; the same paths
  hedged at every frequency show the ``1/sqrt(N)`` law and the cost trade-off. The answer
  keys come alongside: Derman-Kamal's std, the gamma-P&L formula, and (GBM)
  ``BS(realized) - BS(hedge)``.
- ``run_mm_lab``: the M4b desk, skewed and unskewed, on the same market and customer flow,
  pricing on whatever ``VolSurface`` is passed in (synthetic or a fitted snapshot).

Time grid for hedging
---------------------
Trading time, as in the M4a study: ``T`` in years covers ``round(252 T)`` trading days of
7 hourly marks. Frequencies are labelled by how many marks pass between hedges, and only
those that fit inside the option's life are run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.pricing.greeks import vega
from optdesk.sim.hedging import delta_hedge, discrete_hedging_std, frequency_sweep, gamma_pnl
from optdesk.sim.market_maker import (
    Contract,
    MarketPaths,
    MMResult,
    QuoteParams,
    quote_vols,
    run_market_maker,
    simulate_market,
)
from optdesk.sim.mm_plot import MONEYNESS, STEPS_PER_DAY, TENORS, study_contracts, summarize
from optdesk.sim.paths import Paths, gbm_paths, heston_paths
from optdesk.surface.source import VolSurface

__all__ = [
    "FREQUENCIES",
    "HedgeConfig",
    "HedgeLab",
    "MMConfig",
    "MMLab",
    "hedge_grid",
    "heston_params",
    "run_hedge_lab",
    "run_mm_lab",
]

TRADING_DAYS = 252
MARKS_PER_DAY = 7  # hourly marks 9:30 ... 15:30
FREQUENCIES = {  # label -> hourly marks between hedges
    "monthly": 21 * MARKS_PER_DAY,
    "weekly": 5 * MARKS_PER_DAY,
    "daily": MARKS_PER_DAY,
    "every 3 hours": 3,
    "hourly": 1,
}
N_SAMPLE_PATHS = 30  # paths kept for the "what spot did" chart


# --- Hedging lab -------------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeConfig:
    """One hedging experiment. Vols and rates are annual decimals; ``cost`` is a fraction."""

    model: str = "gbm"  # "gbm" or "heston"
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25
    r: float = 0.04
    q: float = 0.013
    call: bool = True
    position: float = 1.0  # +1 long the option, -1 short
    hedge_vol: float = 0.20  # implied vol traded at and hedged with
    realized_vol: float = 0.20  # GBM only: the vol the paths actually have
    heston: dict = field(
        default_factory=lambda: {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    )
    cost: float = 0.0
    frequency: str = "daily"
    n_paths: int = 2_000
    seed: int = 2026


@dataclass
class HedgeLab:
    """Results of ``run_hedge_lab``. Money is per option, discounted to today."""

    config: HedgeConfig
    days: np.ndarray  # trading-day axis of the grid
    sample_S: np.ndarray  # (N_SAMPLE_PATHS, n_steps + 1)
    pnl: np.ndarray  # per-path P&L at the chosen frequency, after costs
    gamma_formula: np.ndarray  # per-path gamma-P&L formula (no costs)
    sweep: pd.DataFrame  # frequency_sweep over every frequency that fits, + derman_kamal
    summary: dict[str, float]


def heston_params(
    start_vol: float, long_run_vol: float, kappa: float, xi: float, rho: float
) -> dict[str, float]:
    """Heston parameters from vols (how people think) to variances (what the model takes)."""
    return {"v0": start_vol**2, "kappa": kappa, "theta": long_run_vol**2, "xi": xi, "rho": rho}


def hedge_grid(T: float) -> tuple[int, dict[str, int]]:
    """Grid steps for an option of life ``T`` and the frequencies that fit inside it."""
    n_days = max(int(round(TRADING_DAYS * T)), 1)
    n_steps = n_days * MARKS_PER_DAY
    return n_steps, {label: every for label, every in FREQUENCIES.items() if every <= n_steps}


def _hedge_paths(c: HedgeConfig, n_steps: int) -> Paths:
    mu = c.r - c.q  # risk-neutral drift, so average P&L lines up with the BS answer keys
    if c.model == "gbm":
        return gbm_paths(c.S0, mu, c.realized_vol, c.T, n_steps, c.n_paths, c.seed)
    if c.model == "heston":
        return heston_paths(
            c.S0, mu=mu, T=c.T, n_steps=n_steps, n_paths=c.n_paths, rng=c.seed, **c.heston
        )
    raise ValueError("model must be 'gbm' or 'heston'")


def run_hedge_lab(config: HedgeConfig) -> HedgeLab:
    """Simulate, hedge at the chosen frequency and at every frequency, and summarize."""
    c = config
    n_steps, freqs = hedge_grid(c.T)
    if c.frequency not in freqs:
        raise ValueError(f"frequency {c.frequency!r} doesn't fit an option of {c.T:.3f} years")
    paths = _hedge_paths(c, n_steps)
    common = (paths, c.K, c.r, c.hedge_vol, c.q, c.call)
    res = delta_hedge(*common, rebalance_every=freqs[c.frequency], cost=c.cost, position=c.position)
    formula = gamma_pnl(*common, position=c.position)
    sweep = frequency_sweep(*common, frequencies=freqs, cost=c.cost, position=c.position)
    sweep["derman_kamal"] = [
        discrete_hedging_std(c.S0, c.K, c.T, c.r, c.hedge_vol, c.q, n)
        for n in sweep["n_rebalances"]
    ]
    premium = float(bs_price(c.S0, c.K, c.T, c.r, c.hedge_vol, c.q, c.call))
    bs_gap = np.nan
    if c.model == "gbm":
        realized_value = float(bs_price(c.S0, c.K, c.T, c.r, c.realized_vol, c.q, c.call))
        bs_gap = c.position * (realized_value - premium)
    n = len(res.pnl)
    summary = {
        "premium": premium,
        "n_rebalances": float(res.n_rebalances),
        "mean": float(res.pnl.mean()),
        "se": float(res.pnl.std(ddof=1) / np.sqrt(n)),
        "std": float(res.pnl.std(ddof=1)),
        "p5": float(np.percentile(res.pnl, 5)),
        "p95": float(np.percentile(res.pnl, 95)),
        "mean_cost": float(res.costs.mean()),
        "gamma_formula_mean": float(formula.mean()),
        "bs_gap": bs_gap,
        "derman_kamal": discrete_hedging_std(
            c.S0, c.K, c.T, c.r, c.hedge_vol, c.q, res.n_rebalances
        ),
        "realized_vol": float(np.sqrt(paths.v[:, :-1].mean())),
    }
    return HedgeLab(
        config=c,
        days=paths.times / c.T * (n_steps / MARKS_PER_DAY),
        sample_S=np.array(paths.S[:N_SAMPLE_PATHS]),
        pnl=res.pnl,
        gamma_formula=formula,
        sweep=sweep,
        summary=summary,
    )


# --- Market-making lab -------------------------------------------------------------------------


@dataclass(frozen=True)
class MMConfig:
    """One market-making experiment (the M4b study with its knobs exposed)."""

    risk_aversion: float = 5e-5
    vol_of_vol: float = 0.10  # in the simulated world and in the desk's quoting
    spot_vol_corr: float = -0.7
    vega_limit: float = 25_000.0  # $ per vol point
    gamma_limit: float = 1_000_000.0  # $ of delta per 1% move
    trade_size: float = 10.0
    arrival_rate: float = 1000.0
    fill_decay: float = 200.0
    hedge_cost: float = 0.0001
    days: int = 21
    n_paths: int = 200
    seed: int = 2026

    def quote_params(self) -> QuoteParams:
        return QuoteParams(
            risk_aversion=self.risk_aversion,
            vol_of_vol=self.vol_of_vol,
            vega_limit=self.vega_limit,
            gamma_limit=self.gamma_limit,
            trade_size=self.trade_size,
            arrival_rate=self.arrival_rate,
            fill_decay=self.fill_decay,
            hedge_cost=self.hedge_cost,
        )


@dataclass
class MMLab:
    """Results of ``run_mm_lab``: both desks on the same market, plus chart-ready series."""

    config: MMConfig
    contracts: list[Contract]
    base_vol: float
    market: MarketPaths
    days: np.ndarray
    results: dict[str, MMResult]  # "No skew", "A-S skew"
    summary: pd.DataFrame  # one row per strategy (mm_plot.summarize)
    vega_bands: dict[str, np.ndarray]  # strategy -> (3, n_steps + 1): 5th, 50th, 95th pct
    mean_cum: dict[str, pd.DataFrame]  # strategy -> average cumulative P&L by bucket
    quote_curve: pd.DataFrame  # book vega vs bid/ask offset from theo, for both desks


def run_mm_lab(config: MMConfig, surface: VolSurface, S0: float, r: float, q: float) -> MMLab:
    """Run the skewed and unskewed desks on the same paths and customer flow."""
    c = config
    contracts = study_contracts(S0, r, q, TENORS, MONEYNESS)
    base_vol = float(surface.implied_vol(0.0, TENORS[0]))  # spot realizes the 2m ATM vol
    horizon = c.days / TRADING_DAYS
    rng = np.random.default_rng(c.seed)
    market = simulate_market(
        S0,
        r,
        q,
        base_vol,
        horizon,
        c.days * STEPS_PER_DAY,
        c.n_paths,
        rng,
        vol_of_vol=c.vol_of_vol,
        spot_vol_corr=c.spot_vol_corr,
    )
    fill_seed = int(rng.integers(2**32))
    skewed = c.quote_params()
    strategies = {"No skew": replace(skewed, skew=False), "A-S skew": skewed}
    results = {
        label: run_market_maker(market, contracts, surface, params, fill_seed)
        for label, params in strategies.items()
    }
    summary = pd.DataFrame(
        [{"strategy": label, **summarize(res)} for label, res in results.items()]
    )
    vega_bands = {
        label: np.percentile(res.vega, [5, 50, 95], axis=0) for label, res in results.items()
    }
    mean_cum = {
        label: pd.DataFrame(
            {
                "spread": res.cum_spread.mean(axis=0),
                "hedging": res.cum_hedging.mean(axis=0),
                "inventory": res.cum_inventory.mean(axis=0),
            }
        ).assign(total=lambda d: d.sum(axis=1))
        for label, res in results.items()
    }

    # How quotes move with inventory, for a 10-lot of the 2-month ATM option.
    T0 = TENORS[0]
    F0 = float(forward_price(S0, T0, r, q))
    trade_vega = c.trade_size * skewed.multiplier * float(vega(S0, F0, T0, r, base_vol, q))
    book_vega = np.linspace(-c.vega_limit, c.vega_limit, 101) * 100  # $ per 1.00 vol
    curve = {"book_vega": book_vega / 100}
    for label, params in strategies.items():
        bid, ask = quote_vols(base_vol, trade_vega, book_vega, params)
        curve[f"{label} bid"] = 100 * (bid - base_vol)
        curve[f"{label} ask"] = 100 * (ask - base_vol)

    return MMLab(
        config=c,
        contracts=contracts,
        base_vol=base_vol,
        market=market,
        days=market.times / horizon * c.days,
        results=results,
        summary=summary,
        vega_bands=vega_bands,
        mean_cum=mean_cum,
        quote_curve=pd.DataFrame(curve),
    )
