"""Options market-making simulator: quote around theo, take random customer flow, hedge delta,
skew quotes by inventory (Avellaneda-Stoikov), respect risk limits, and explain the P&L.

The desk
--------
The market maker quotes a bid and an ask on a small book of listed options (puts below the
forward, calls above, a few expiries). Every quote is centred on **theoretical value** (theo):
Black-Scholes at the vol the surface gives that strike and expiry, plus the day's vol move.
Customers arrive at random and trade against the quotes. After every step the desk sells or
buys shares so the book is delta-neutral, and it tracks the book's delta, gamma and vega.

The world (``simulate_market``)
-------------------------------
Spot follows a log-normal step whose vol is the surface's ATM level plus a random **vol
shift** ``x``. The same ``x`` moves the whole surface up or down in parallel, and it is an
arithmetic Brownian motion with "vol of vol" ``sigma_v`` and correlation ``rho`` with spot
(negative for equities: selloffs come with vol spikes). Moving implied and realized vol
together means the desk has no systematic edge or loss on gamma; its risk is the vol move.
The surface moves *sticky-moneyness*: the vol at ``k = ln(K/F)`` is fixed as spot moves, so the
smile travels with the forward. Where the surface comes from is swappable (any
``surface.source.VolSurface``): the synthetic SSVI surface today, live SVI fits later.

Why Avellaneda-Stoikov, and why in vol space
--------------------------------------------
Avellaneda & Stoikov (2008) solve the problem of a dealer with exponential utility (risk
aversion ``gamma``) who earns the spread but carries inventory ``q`` in an asset of vol
``sigma``. The answer has two parts:

- **Reservation price** ``s - q gamma sigma^2 H``: the dealer's own value of the asset, shaded
  *against* the inventory. Long inventory lowers both quotes, so customers buy from you more
  and sell to you less, and the inventory drifts back towards zero.
- **Optimal spread** ``gamma sigma^2 H + (2/gamma) ln(1 + gamma/k)`` around it, where fills
  arrive at rate ``A e^{-k delta}`` for a quote ``delta`` away from fair value. The log term is
  the monopolist's markup (wide enough to earn, tight enough to trade); the first term is extra
  compensation for the risk each fill adds.

An options desk hedges delta, so what's left is **vega**: every option loses or gains
``vega x (vol move)``. Treat the vol shift as the risky asset and A-S carries over: with
book vega ``V`` (dollars per 1.00 vol) and vol of vol ``sigma_v``, every option's reservation
*vol* is theo vol minus ``gamma sigma_v^2 H V``. That's the same shift for every strike,
because they all load on the one vol factor (the multi-asset A-S reservation price
``s_i - gamma H Cov(dV_i, dBook)``, divided by ``vega_i``). Quoting in vol is also how option
traders actually think: "I'm long vega, lower my vols." The spread in vol for a fill that
adds vega ``nu`` (dollars per 1.00 vol) is

    gamma nu sigma_v^2 H + (2 / (gamma nu)) ln(1 + gamma nu / k)

with ``k`` the fill-rate decay per 1.00 vol. Its limit as ``gamma -> 0`` is ``2 / k``: a
risk-neutral dealer just quotes the monopolist's spread and never skews.

Design choices
--------------
- **Constant risk horizon** ``H`` instead of A-S's time-to-session-end. A-S's quotes collapse
  to symmetric at the close because the dealer is assumed to be flat then; a desk carrying a
  book for weeks isn't. Guéant, Lehalle & Fernandez-Tapia (2013) show the optimal quotes
  settle to stationary values far from the terminal date, and a fixed ``H`` ("how long it
  takes to work out of a position", 1 day by default) is the usual practical stand-in.
- **Poisson fills with exponential intensity** ``A e^{-k delta}``, ``delta`` the quote's distance
  from theo in vol. Better prices fill more often; this is the A-S fill model, and with it the
  quotes above are optimal.
- **Delta hedged every step** in the underlying, paying a proportional cost. By default with
  Black-Scholes delta at each option's smile vol (the desk-standard "sticky-strike" delta).
  ``delta_method="sticky_moneyness"`` instead bumps spot and reprices on the surface, so the
  delta includes the smile moving with spot. That makes the *hedging* bucket quieter, but in
  this world it raises *total* risk: vols fall when spot rises (``rho < 0``), and the
  sticky-strike delta happens to lean against that, a crude minimum-variance hedge. Neither
  delta is "right"; the best one depends on how vols really move with spot.
- **Risk limits** on book vega (dollars per vol point) and gamma (dollars of delta per 1% move),
  enforced like a desk's pre-trade check: a bid (or offer) on a contract is shown only if one
  more fill there would keep the book inside both limits. Near a long-vega limit the desk
  stops bidding but keeps offering, which also pulls inventory back. The desk never trades
  *out* of a breach, so exposure can still drift past a limit when the market moves the
  Greeks (gamma of a near-dated option near its strike), or when several fills land in one step.

P&L attribution (exact: the three buckets sum to the mark-to-market P&L on every path)
--------------------------------------------------------------------------------------
- **Spread capture**: at each fill, the price traded minus theo, from the desk's side. This is
  what market making is paid for. Skewing gives some of it back: when you're long, the offer you
  shade lower may sit *below* theo.
- **Hedging P&L**: option value change from spot and time at unchanged vols (gamma, theta,
  smile roll) plus the share hedge's P&L, dividends, interest on cash, minus hedge costs. With
  realized vol tracking implied, this averages zero; it's the noise of discrete hedging.
- **Inventory P&L**: option value change from the vol shift, i.e. ``V x dsigma`` plus convexity.
  This is the risk of carrying inventory, and it's what skewing reduces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import delta, gamma, vega
from optdesk.sim.paths import _rng, time_grid
from optdesk.surface.source import VolSurface

__all__ = [
    "Contract",
    "MMResult",
    "MarketPaths",
    "QuoteParams",
    "fill_intensity",
    "half_spread",
    "quote_vols",
    "reservation_shift",
    "run_market_maker",
    "simulate_market",
]

MIN_VOL = 0.01  # floor on any vol used to price, quote or simulate
DELTA_BUMP = 1e-4  # relative spot bump for the surface-consistent (sticky-moneyness) delta


class Contract(NamedTuple):
    """A listed option: strike, expiry in years from the start of the simulation, call/put."""

    K: float
    T: float
    call: bool


class MarketPaths(NamedTuple):
    """Simulated market: spot and the parallel vol shift on a time grid, plus rates.

    ``S`` and ``vol_shift`` have shape ``(n_paths, n_steps + 1)``; ``vol_shift`` is added to every
    vol on the surface (0.01 = one vol point).
    """

    times: NDArray
    S: NDArray
    vol_shift: NDArray
    r: float
    q: float


@dataclass(frozen=True)
class QuoteParams:
    """How the desk quotes, trades and limits risk. Money is in dollars.

    risk_aversion : A-S ``gamma``, per dollar. 0 = never skew, quote the monopolist spread.
    vol_of_vol : the desk's estimate of ``sigma_v`` (absolute vol per sqrt(year)).
    horizon : risk horizon ``H`` in years (see module notes).
    fill_decay : ``k``, fill-rate decay per 1.00 of vol distance (200 = e-fold per 0.5 point).
    arrival_rate : ``A``, customer arrivals per year per side per contract at zero distance.
    trade_size : contracts per customer trade.
    multiplier : shares per contract.
    skew : shift quotes by inventory (True) or keep them centred on theo (False). The spread
        is the A-S spread either way, so the comparison isolates the skew.
    vega_limit : max |book vega| in dollars per vol point (``inf`` = no limit).
    gamma_limit : max |book gamma| in dollars of delta per 1% spot move.
    hedge_cost : proportional cost per share traded (half-spread, 0.0001 = 1 bp).
    delta_method : ``"sticky_strike"`` (Black-Scholes delta at the smile vol) or
        ``"sticky_moneyness"`` (spot bumped on the surface; see module notes).
    """

    risk_aversion: float = 5e-5
    vol_of_vol: float = 0.10
    horizon: float = 1 / 252
    fill_decay: float = 200.0
    arrival_rate: float = 1000.0
    trade_size: float = 10.0
    multiplier: float = 100.0
    skew: bool = True
    vega_limit: float = np.inf
    gamma_limit: float = np.inf
    hedge_cost: float = 0.0001
    delta_method: str = "sticky_strike"


class MMResult(NamedTuple):
    """Outcome of ``run_market_maker``. Per-path arrays have shape ``(n_paths,)``; time series
    have shape ``(n_paths, n_steps + 1)`` (the last column is the final state)."""

    spread: NDArray  # P&L buckets, summed over the run
    hedging: NDArray
    inventory: NDArray
    total: NDArray  # mark-to-market P&L: cash + options at theo + shares
    costs: NDArray  # hedge costs (already inside ``hedging``)
    fills: NDArray  # customer trades
    positions: NDArray  # final contracts held, shape (n_paths, n_contracts)
    vega: NDArray  # book vega, dollars per vol point
    gamma: NDArray  # book gamma, dollars of delta per 1% move
    delta: NDArray  # book delta in shares just before each rehedge (0 right after)
    shift: NDArray  # reservation shift of the quotes, vol (0.01 = one point)
    cum_spread: NDArray
    cum_hedging: NDArray
    cum_inventory: NDArray
    at_limit: NDArray  # fraction of steps with at least one quote pulled by a limit


# --- Quoting formulas ---------------------------------------------------------------------------


def reservation_shift(
    book_vega: ArrayLike, risk_aversion: float, vol_of_vol: float, horizon: float
) -> NDArray:
    """A-S reservation shift in vol: ``gamma sigma_v^2 H V`` (subtract it from theo vol).

    ``book_vega`` in dollars per 1.00 vol. Positive when long vega: quotes move down.
    """
    return risk_aversion * vol_of_vol**2 * horizon * np.asarray(book_vega, dtype=float)


def half_spread(
    trade_vega: ArrayLike,
    risk_aversion: float,
    vol_of_vol: float,
    horizon: float,
    fill_decay: float,
) -> NDArray:
    """A-S optimal half-spread in vol for a fill that adds ``trade_vega`` (dollars per 1.00 vol).

    ``1/2 gamma nu sigma_v^2 H + (1 / (gamma nu)) ln(1 + gamma nu / k)``; ``1 / k`` when
    ``gamma nu = 0``.
    """
    g_nu = risk_aversion * np.asarray(trade_vega, dtype=float)
    inventory_term = 0.5 * g_nu * vol_of_vol**2 * horizon
    with np.errstate(divide="ignore", invalid="ignore"):
        markup = np.where(g_nu > 0, np.log1p(g_nu / fill_decay) / g_nu, 1.0 / fill_decay)
    return inventory_term + markup


def quote_vols(
    theo_vol: ArrayLike, trade_vega: ArrayLike, book_vega: ArrayLike, params: QuoteParams
) -> tuple[NDArray, NDArray]:
    """Bid and ask vols: ``theo - shift -/+ half_spread`` (shift is 0 when ``params.skew`` is off).

    ``book_vega`` broadcasts against the contract axis (pass shape ``(n_paths, 1)``).
    """
    p = params
    half = half_spread(trade_vega, p.risk_aversion, p.vol_of_vol, p.horizon, p.fill_decay)
    shift = reservation_shift(book_vega, p.risk_aversion, p.vol_of_vol, p.horizon) * p.skew
    centre = np.asarray(theo_vol, dtype=float) - shift
    return np.maximum(centre - half, MIN_VOL), centre + half


def fill_intensity(distance: ArrayLike, arrival_rate: float, fill_decay: float) -> NDArray:
    """Customer arrivals per year at a quote ``distance`` vol away from theo: ``A e^{-k d}``."""
    return arrival_rate * np.exp(-fill_decay * np.asarray(distance, dtype=float))


# --- The world ----------------------------------------------------------------------------------


def simulate_market(
    S0: float,
    r: float,
    q: float,
    base_vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | int | None = None,
    *,
    vol_of_vol: float = 0.10,
    spot_vol_corr: float = -0.7,
) -> MarketPaths:
    """Spot and a parallel vol shift, correlated (see module notes).

    Over each step spot moves log-normally at vol ``max(base_vol + x, MIN_VOL)`` with
    risk-neutral drift ``r - q``, and ``x`` takes an arithmetic Brownian step of size
    ``vol_of_vol sqrt(dt)``, correlated ``spot_vol_corr`` with the spot shock.
    """
    if S0 <= 0 or base_vol <= 0:
        raise ValueError("S0 and base_vol must be positive")
    if vol_of_vol < 0:
        raise ValueError("vol_of_vol must be non-negative")
    if not -1.0 <= spot_vol_corr <= 1.0:
        raise ValueError("spot_vol_corr must be in [-1, 1]")
    times = time_grid(T, n_steps)
    dt = T / n_steps
    gen = _rng(rng)
    z_s = gen.standard_normal((n_paths, n_steps))
    z_v = spot_vol_corr * z_s + np.sqrt(1 - spot_vol_corr**2) * gen.standard_normal(z_s.shape)
    x = np.zeros((n_paths, n_steps + 1))
    x[:, 1:] = np.cumsum(vol_of_vol * np.sqrt(dt) * z_v, axis=1)
    vol = np.maximum(base_vol + x[:, :-1], MIN_VOL)
    log_steps = (r - q - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z_s
    log_S = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return MarketPaths(times=times, S=S0 * np.exp(log_S), vol_shift=x, r=float(r), q=float(q))


# --- The desk -----------------------------------------------------------------------------------


def run_market_maker(
    market: MarketPaths,
    contracts: list[Contract],
    surface: VolSurface,
    params: QuoteParams = QuoteParams(),  # noqa: B008 (frozen dataclass: safe default)
    rng: np.random.Generator | int | None = None,
) -> MMResult:
    """Make markets in ``contracts`` along every simulated path. See the module notes.

    Each step: price every contract at theo, read the book's Greeks, pull a side if a limit is
    hit, quote (skewed by book vega if ``params.skew``), draw Poisson customer fills, rehedge
    delta, then let spot and vols move and attribute the change in value.
    """
    p = params
    times, S, x, r, q = market
    n_paths, n_steps = S.shape[0], len(times) - 1
    K = np.array([c.K for c in contracts], dtype=float)
    T0 = np.array([c.T for c in contracts], dtype=float)
    call = np.array([c.call for c in contracts], dtype=bool)
    if np.any(T0 <= times[-1]):
        raise ValueError("every contract must expire after the simulation ends")
    if p.delta_method not in ("sticky_strike", "sticky_moneyness"):
        raise ValueError("delta_method must be 'sticky_strike' or 'sticky_moneyness'")
    gen = _rng(rng)
    mult = p.multiplier

    def theo(spot: NDArray, t: float, shift: NDArray) -> tuple[NDArray, NDArray]:
        """Theo vol and price of every contract, shape (n_paths, n_contracts)."""
        tau = T0 - t
        F = spot[:, None] * np.exp((r - q) * tau)
        vol = np.maximum(surface.implied_vol(np.log(K / F), tau) + shift[:, None], MIN_VOL)
        return vol, bs_price(spot[:, None], K, tau, r, vol, q, call)

    pos = np.zeros((n_paths, len(contracts)))
    shares, cash = np.zeros(n_paths), np.zeros(n_paths)
    costs, fills, limit_steps = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)
    series = {name: np.zeros((n_paths, n_steps + 1)) for name in ("vega", "gamma", "delta")}
    series["shift"] = np.zeros((n_paths, n_steps + 1))
    cum = {name: np.zeros((n_paths, n_steps + 1)) for name in ("spread", "hedging", "inventory")}
    spread, hedging, inventory = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)

    def greeks(spot, t, vol, shift):
        """Per-contract delta, gamma and vega at theo vol, shape (n_paths, n_contracts)."""
        args = (spot[:, None], K, T0 - t, r, vol, q, call)
        d = delta(*args)
        if p.delta_method == "sticky_moneyness":  # include the smile moving with spot
            up = theo(spot * (1 + DELTA_BUMP), t, shift)[1]
            down = theo(spot * (1 - DELTA_BUMP), t, shift)[1]
            d = (up - down) / (2 * DELTA_BUMP * spot[:, None])
        return d, gamma(*args), vega(*args)

    def book(spot, d, g, v):
        """Book delta (shares), vega (per 1.00 vol) and gamma ($ of delta per 1% move)."""
        return (
            mult * (pos * d).sum(axis=1),
            mult * (pos * v).sum(axis=1),
            mult * (pos * g).sum(axis=1) * spot**2 / 100,
        )

    vol, px = theo(S[:, 0], times[0], x[:, 0])
    for i in range(n_steps):
        s, tau, dt = S[:, i], T0 - times[i], times[i + 1] - times[i]
        d, g, v = greeks(s, times[i], vol, x[:, i])
        _, book_vega, gamma_1pct = book(s, d, g, v)

        # Pre-trade risk check: quote a side only if one more fill there stays inside limits.
        fill_vega = p.trade_size * mult * v / 100  # $ per vol point added by buying one fill
        fill_gamma = p.trade_size * mult * g * s[:, None] ** 2 / 100
        vega_pt = book_vega[:, None] / 100
        can_buy = (vega_pt + fill_vega <= p.vega_limit) & (
            gamma_1pct[:, None] + fill_gamma <= p.gamma_limit
        )
        can_sell = (vega_pt - fill_vega >= -p.vega_limit) & (
            gamma_1pct[:, None] - fill_gamma >= -p.gamma_limit
        )
        limit_steps += ~(can_buy & can_sell).all(axis=1)

        bid_vol, ask_vol = quote_vols(vol, p.trade_size * mult * v, book_vega[:, None], p)
        if p.skew:
            series["shift"][:, i] = reservation_shift(
                book_vega, p.risk_aversion, p.vol_of_vol, p.horizon
            )
        bid_px = bs_price(s[:, None], K, tau, r, bid_vol, q, call)
        ask_px = bs_price(s[:, None], K, tau, r, ask_vol, q, call)
        lam_bid = fill_intensity(vol - bid_vol, p.arrival_rate, p.fill_decay) * can_buy
        lam_ask = fill_intensity(ask_vol - vol, p.arrival_rate, p.fill_decay) * can_sell
        bought = p.trade_size * gen.poisson(lam_bid * dt)  # customers sell to us at our bid
        sold = p.trade_size * gen.poisson(lam_ask * dt)  # customers buy from us at our ask
        fills += ((bought + sold) / p.trade_size).sum(axis=1)
        spread += mult * (bought * (px - bid_px) + sold * (ask_px - px)).sum(axis=1)
        cash += mult * (sold * ask_px - bought * bid_px).sum(axis=1)
        pos += bought - sold

        # Record the book after trading, then rehedge delta to zero in the underlying.
        option_delta, book_vega, gamma_1pct = book(s, d, g, v)
        series["delta"][:, i] = option_delta + shares
        series["vega"][:, i], series["gamma"][:, i] = book_vega / 100, gamma_1pct
        trade = -option_delta - shares
        cost = p.hedge_cost * np.abs(trade) * s
        cash -= trade * s + cost
        shares = -option_delta
        costs += cost
        hedging -= cost

        # Markets move: attribute the change in option value to spot/time, then to vols.
        s1 = S[:, i + 1]
        _, px_spot_moved = theo(s1, times[i + 1], x[:, i])
        vol1, px1 = theo(s1, times[i + 1], x[:, i + 1])
        carry = shares * s * q * dt + cash * r * dt  # dividends on shares, interest on cash
        hedging += mult * (pos * (px_spot_moved - px)).sum(axis=1) + shares * (s1 - s) + carry
        inventory += mult * (pos * (px1 - px_spot_moved)).sum(axis=1)
        cash += carry
        vol, px = vol1, px1
        cum["spread"][:, i + 1], cum["hedging"][:, i + 1] = spread, hedging
        cum["inventory"][:, i + 1] = inventory

    option_delta, book_vega, gamma_1pct = book(
        S[:, -1], *greeks(S[:, -1], times[-1], vol, x[:, -1])
    )
    series["vega"][:, -1], series["gamma"][:, -1] = book_vega / 100, gamma_1pct
    series["delta"][:, -1] = option_delta + shares
    total = cash + mult * (pos * px).sum(axis=1) + shares * S[:, -1]
    return MMResult(
        spread=spread,
        hedging=hedging,
        inventory=inventory,
        total=total,
        costs=costs,
        fills=fills,
        positions=pos,
        vega=series["vega"],
        gamma=series["gamma"],
        delta=series["delta"],
        shift=series["shift"],
        cum_spread=cum["spread"],
        cum_hedging=cum["hedging"],
        cum_inventory=cum["inventory"],
        at_limit=limit_steps / n_steps,
    )
