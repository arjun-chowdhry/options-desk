"""Delta-hedging simulator: what a trader actually makes or loses hedging an option.

The setup
---------
At time 0 the trader buys one option (``position=+1``; ``-1`` sells it) at its Black-Scholes
price at the *hedge vol* (think: the implied vol they traded at), and sells ``delta`` shares
against it. At each rebalance date they reset the share position to the new Black-Scholes
delta at the same hedge vol. Cash earns ``r``, shares held pay dividends at ``q``, and at
expiry the option pays off. The P&L of each path is reported in today's money (every cash
flow discounted at ``r``), so it is directly comparable to option prices.

Two answer keys
---------------
1. **Realized vol = hedge vol, continuous hedging:** P&L is exactly zero on every path
   (that *is* the Black-Scholes replication argument). Hedging only at discrete times leaves
   a hedging error with mean about zero and a spread that shrinks like ``1/sqrt(N)`` in the
   number of rebalances ``N``. Derman & Kamal (1999) give the size for an at-the-money option:
   ``std ~ sqrt(pi/4) * vega * sigma / sqrt(N)`` (``discrete_hedging_std``).
2. **Realized vol != hedge vol:** the hedge no longer replicates, and over each small step a
   long option earns ``1/2 * gamma * S^2 * (realized variance - hedge variance) * dt``. Summed
   over the life of the trade (``gamma_pnl``), that is the P&L of a delta-hedged option: long
   gamma makes money when the stock moves more than the vol you paid for. When the paths are
   risk-neutral (``mu = r - q``) its average is exactly ``BS(realized) - BS(hedge)``.

Design choices
--------------
- **Hedge with Black-Scholes delta at the hedge vol, not the realized vol.** The trader
  doesn't know future realized vol; the implied vol is the one number they have, and it is
  how desks mark and hedge. Hedging at realized vol instead gives a smooth P&L equal to
  ``BS(realized) - BS(implied)``, but needs foresight.
- **Proportional transaction costs** (``cost`` = half the bid-ask spread as a fraction of the
  price, so 0.0005 = 5 bp): every share traded costs ``cost * |shares| * S``. This charges
  both the opening hedge and each rebalance. The expiry unwind is free because a physically
  settled option delivers the very shares the hedge holds (for a cash-settled option add one
  more trade). Costs are what stop a trader hedging every second: they grow roughly like
  ``sqrt(N)`` while the hedging error shrinks like ``1/sqrt(N)``.
- **Paths are simulated once and hedged at many frequencies** (``rebalance_every`` subsamples
  the grid). Comparing frequencies on the same paths removes simulation noise from the
  comparison: differences come from hedging, not from luck.
- **Dividends accrue step by step** on the shares held (left-point rule). With ``q = 0`` the
  accounting is exact, which the put-call parity test checks to machine precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import delta, gamma, vega
from optdesk.sim.paths import Paths

__all__ = [
    "HedgeResult",
    "delta_hedge",
    "discrete_hedging_std",
    "frequency_sweep",
    "gamma_pnl",
]


class HedgeResult(NamedTuple):
    """Per-path outcome of ``delta_hedge``. Money amounts are discounted to time 0."""

    pnl: NDArray  # P&L after transaction costs, shape (n_paths,)
    costs: NDArray  # transaction costs paid (>= 0), shape (n_paths,)
    n_rebalances: int  # hedge trades per path, counting the opening hedge


def _payoff(S: NDArray, K: float, call: bool) -> NDArray:
    return np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)


def delta_hedge(
    paths: Paths,
    K: float,
    r: float,
    hedge_vol: float,
    q: float = 0.0,
    call: bool = True,
    *,
    rebalance_every: int = 1,
    cost: float = 0.0,
    position: float = 1.0,
) -> HedgeResult:
    """Buy (or sell) a European option and delta-hedge it along every path.

    Parameters
    ----------
    paths : simulated prices (``paths.times`` runs from 0 to the option's expiry).
    K : strike.
    r, q : interest rate and dividend yield (continuous, annual).
    hedge_vol : vol used to price the option at inception and to compute every hedge delta.
    call : True for a call, False for a put.
    rebalance_every : rehedge every this many grid steps (1 = every step). The last interval
        is shorter if it doesn't divide the grid.
    cost : proportional cost per share traded, as a fraction of price (half-spread).
    position : +1 long one option, -1 short one (any size works; P&L scales linearly).
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    if cost < 0:
        raise ValueError("cost must be non-negative")
    times, S = paths.times, paths.S
    n_steps = len(times) - 1
    T = times[-1]
    tau = T - times  # time to expiry at each grid point
    dt = np.diff(times)
    disc = np.exp(-r * times)
    rebalance_steps = range(0, n_steps, rebalance_every)

    premium = bs_price(S[:, 0], K, T, r, hedge_vol, q, call)
    pv = -position * premium  # pay for the option
    costs = np.zeros(S.shape[0])
    shares = np.zeros(S.shape[0])
    for i in range(n_steps):
        if i % rebalance_every == 0:
            target = -position * delta(S[:, i], K, tau[i], r, hedge_vol, q, call)
            trade = target - shares
            pv = pv - disc[i] * trade * S[:, i]
            costs = costs + disc[i] * cost * np.abs(trade) * S[:, i]
            shares = target
        pv = pv + disc[i + 1] * shares * S[:, i] * q * dt[i]  # dividends (paid out when short)
    pv = pv + disc[-1] * (position * _payoff(S[:, -1], K, call) + shares * S[:, -1])
    return HedgeResult(pnl=pv - costs, costs=costs, n_rebalances=len(rebalance_steps))


def gamma_pnl(
    paths: Paths,
    K: float,
    r: float,
    hedge_vol: float,
    q: float = 0.0,
    call: bool = True,
    *,
    position: float = 1.0,
) -> NDArray:
    """The gamma-P&L formula, evaluated along each path (discounted to time 0).

    ``sum over steps of e^{-r t} * 1/2 * gamma * S^2 * (v - hedge_vol^2) * dt``, with gamma
    at the hedge vol and ``v`` the variance the path actually had over that step
    (``paths.v``). This is the continuous-time P&L of a delta-hedged option. It explains why
    a trader who buys vol at 20% and sees 25% realized makes money, and why the profit is
    largest when the stock spends its time near the strike (where gamma is high) close to
    expiry.
    """
    times, S, v = paths.times, paths.S, paths.v
    tau = (times[-1] - times)[:-1]
    g = gamma(S[:, :-1], K, tau, r, hedge_vol, q, call)
    integrand = 0.5 * g * S[:, :-1] ** 2 * (v[:, :-1] - hedge_vol**2)
    weights = np.exp(-r * times[:-1]) * np.diff(times)
    return position * integrand @ weights


def discrete_hedging_std(
    S: float, K: float, T: float, r: float, sigma: float, q: float, n_rebalances: int
) -> float:
    """Derman-Kamal estimate of the P&L std from hedging ``n_rebalances`` times.

    ``sqrt(pi/4) * vega * sigma / sqrt(N)``. Derived for an at-the-money option with the
    realized vol equal to the hedge vol: each step leaves an unhedged
    ``1/2 gamma S^2 (dS^2/S^2 - sigma^2 dt)`` that averages to zero but doesn't cancel, and
    ``N`` independent errors of size ``~1/N`` add up to ``~1/sqrt(N)``.
    """
    v = float(vega(S, K, T, r, sigma, q))
    return float(np.sqrt(np.pi / 4.0) * v * sigma / np.sqrt(n_rebalances))


def frequency_sweep(
    paths: Paths,
    K: float,
    r: float,
    hedge_vol: float,
    q: float = 0.0,
    call: bool = True,
    *,
    frequencies: Mapping[str, int],
    cost: float = 0.0,
    position: float = 1.0,
) -> pd.DataFrame:
    """Hedge the same paths at several frequencies and summarize the P&L of each.

    ``frequencies`` maps a label to ``rebalance_every`` (grid steps between hedges), e.g.
    ``{"daily": 7, "hourly": 1}`` on an hourly grid. Returns one row per frequency with
    ``n_rebalances``, the P&L ``mean``, its standard error ``se``, ``std``, the mean cost
    ``mean_cost``, and the per-path ``pnl`` array (for histograms).
    """
    rows = []
    for label, every in frequencies.items():
        res = delta_hedge(
            paths, K, r, hedge_vol, q, call, rebalance_every=every, cost=cost, position=position
        )
        n = len(res.pnl)
        rows.append(
            {
                "frequency": label,
                "rebalance_every": every,
                "n_rebalances": res.n_rebalances,
                "mean": res.pnl.mean(),
                "se": res.pnl.std(ddof=1) / np.sqrt(n),
                "std": res.pnl.std(ddof=1),
                "mean_cost": res.costs.mean(),
                "pnl": res.pnl,
            }
        )
    return pd.DataFrame(rows)
