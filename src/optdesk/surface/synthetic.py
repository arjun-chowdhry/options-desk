"""A synthetic SPY-like option chain built from a known, arbitrage-free SVI surface.

Why synthetic
-------------
A real chain has no answer key: we could check that an SVI fit *runs*, not that it finds the
*right* smile. Here every quote is generated from parameters we choose, so the tests can ask
"did the fit recover them?" It also lets us develop the surface before we have a live
snapshot (the first one is due Monday).

How it's built
--------------
1. **The true surface is SSVI** (Gatheral & Jacquier 2014): every expiry's smile is an SVI
   slice tied to its ATM total variance ``theta(T)`` by a skew function
   ``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))``. SSVI comes with simple,
   proven no-arbitrage conditions. With ``eta (1 + |rho|) <= 2`` there is no butterfly
   arbitrage, and with ``theta`` increasing in ``T`` and ``0 < gamma <= 1/2`` there is no
   calendar arbitrage. Our values (``rho = -0.7``, ``eta = 1.1``, ``gamma = 0.45``) satisfy
   both, so the true surface is a clean negative control for ``arbitrage.py``. Each slice
   converts exactly to raw SVI (``svi.ssvi_slice``), which gives the answer key.
2. **Realistic SPY levels:** spot 580, 1.3% dividend yield, a Treasury curve near 4%, ATM vol
   rising from ~14% (1 week) to ~17% (14 months), and a skew of about -1 vol point per 1%
   of moneyness at 1 month, all typical of SPY in a calm market.
3. **American prices.** Each quote is priced on a CRR tree with early exercise (as SPY options
   are), at the vol the true surface gives its strike. So the true surface is the
   *European-equivalent* vol, exactly what the cleaner's de-Americanizing should recover.
4. **Market dirt:** a bid-ask spread (one cent each side, or 0.25% of the price if larger),
   random noise in the mid (``noise`` x half-spread), and quotes rounded out to the cent. In
   the far wings that rounding alone moves the implied vol by whole points, which is why the
   fit weights quotes by vega. Options worth under a cent get a zero bid, as on a real screen.

Strikes are $1 apart for expiries under ~2 months and $5 apart beyond (like SPY's listing),
from about 4 ATM standard deviations below the forward to 2.5 above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from optdesk.data.clean import year_fraction
from optdesk.data.fetch import RAW_COLUMNS
from optdesk.data.rates import RateCurve, treasury_curve
from optdesk.pricing.binomial import binomial_price
from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.surface.svi import PARAM_NAMES, SVIParams, ssvi_slice

__all__ = ["SyntheticMarket", "atm_vol", "ssvi_phi", "synthetic_chain", "true_slices"]

TICKER = "SPY"
ASOF = pd.Timestamp("2026-10-16 19:45", tz="UTC")  # Friday 3:45pm New York time
SPOT = 580.0
DIV_YIELD = 0.013
TREASURY_QUOTES_PCT = {"^IRX": 4.00, "^FVX": 3.90, "^TNX": 4.10}
EXPIRIES = [
    "2026-10-23",
    "2026-10-30",
    "2026-11-20",
    "2026-12-18",
    "2027-01-15",
    "2027-03-19",
    "2027-06-17",
    "2027-12-17",
]
SSVI_RHO = -0.7  # spot-vol correlation: strongly negative for equity indices
SSVI_ETA = 1.1  # skew level; eta (1 + |rho|) = 1.87 <= 2 rules out butterfly arbitrage
SSVI_GAMMA = 0.45  # how fast skew flattens with maturity; <= 1/2 rules out calendar arbitrage
TREE_STEPS = 400  # American prices: average of 400 and 401 steps damps the odd/even wobble


def atm_vol(T: ArrayLike) -> NDArray:
    """ATM implied vol term structure: ~14% at 1 week rising to 17% long-dated (upward, calm)."""
    return 0.17 - 0.03 * np.exp(-np.asarray(T, dtype=float) / 0.5)


def ssvi_phi(theta: ArrayLike) -> NDArray:
    """SSVI power-law skew function ``eta / (theta^gamma (1 + theta)^(1 - gamma))``."""
    theta = np.asarray(theta, dtype=float)
    return SSVI_ETA / (theta**SSVI_GAMMA * (1 + theta) ** (1 - SSVI_GAMMA))


def true_slices(T: ArrayLike) -> list[SVIParams]:
    """Raw SVI parameters of the true surface at each expiry ``T``."""
    out = []
    for t in np.atleast_1d(np.asarray(T, dtype=float)):
        theta = float(atm_vol(t)) ** 2 * t
        out.append(ssvi_slice(theta, SSVI_RHO, float(ssvi_phi(theta))))
    return out


@dataclass(frozen=True)
class SyntheticMarket:
    """A raw chain (as ``fetch_chain`` returns it), its rate curve, and the answer key."""

    raw: pd.DataFrame  # RAW_COLUMNS, one row per contract
    curve: RateCurve
    truth: pd.DataFrame  # one row per expiry: expiry, T, r, forward, a, b, rho, m, sigma


def _contract(expiry: pd.Timestamp, option_type: str, strike: float) -> str:
    return f"{TICKER}{expiry:%y%m%d}{option_type[0].upper()}{round(strike * 1000):08d}"


def _strikes(F: float, T: float) -> NDArray:
    step = 1.0 if T < 0.15 else 5.0
    sd = float(atm_vol(T)) * np.sqrt(T)
    lo = np.floor(F * np.exp(-4.0 * sd) / step) * step
    hi = np.ceil(F * np.exp(2.5 * sd) / step) * step
    return np.arange(lo, hi + step / 2, step)


def synthetic_chain(
    *,
    american: bool = True,
    noise: float = 0.25,
    tick: float | None = 0.01,
    seed: int = 0,
    expiries: list[str] | None = None,
) -> SyntheticMarket:
    """Generate a SPY-like raw chain from the true SSVI surface.

    Parameters
    ----------
    american : price with early exercise on a CRR tree (True, like SPY) or Black-Scholes.
    noise : standard deviation of the error added to each mid, as a fraction of the
        half-spread. 0 gives mids equal to the model price (before rounding).
    tick : round bids down and asks up to this price increment; ``None`` = no rounding.
        ``american=False, noise=0, tick=None`` gives exact European quotes.
    seed : seed for the ``np.random.Generator`` (noise, volumes, trade times).
    expiries : expiry dates (``YYYY-MM-DD``); default ``EXPIRIES``.
    """
    rng = np.random.default_rng(seed)
    curve = treasury_curve(TREASURY_QUOTES_PCT)
    frames, truth = [], []
    for expiry in pd.to_datetime(expiries or EXPIRIES):
        T = float(year_fraction(ASOF, expiry))
        r = float(curve(T))
        F = float(forward_price(SPOT, T, r, DIV_YIELD))
        params = true_slices(T)[0]
        truth.append({"expiry": expiry, "T": T, "r": r, "forward": F, **vars(params)})

        K = _strikes(F, T)
        vol = params.implied_vol(np.log(K / F), T)
        for option_type in ("call", "put"):
            is_call = option_type == "call"
            if american:
                price = (
                    sum(
                        binomial_price(SPOT, K, T, r, vol, DIV_YIELD, is_call, steps=n)
                        for n in (TREE_STEPS, TREE_STEPS + 1)
                    )
                    / 2
                )
            else:
                price = bs_price(SPOT, K, T, r, vol, DIV_YIELD, is_call)
            half = np.maximum(0.01, 0.0025 * price)
            mid = price + rng.normal(0.0, noise * half) if noise else price
            bid, ask = mid - half, mid + half
            if tick:
                bid = np.floor(np.round(bid / tick, 6)) * tick
                ask = np.ceil(np.round(ask / tick, 6)) * tick
            bid = np.where(bid < 0.005, 0.0, bid)  # nobody bids for an option under a cent
            n = len(K)
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": TICKER,
                        "asof": ASOF,
                        "spot": SPOT,
                        "expiry": expiry,
                        "option_type": option_type,
                        "contract": [_contract(expiry, option_type, k) for k in K],
                        "strike": K,
                        "bid": bid,
                        "ask": ask,
                        "last": np.round((bid + ask) / 2, 2),
                        "last_trade": ASOF - pd.to_timedelta(rng.uniform(0, 3, n), unit="h"),
                        "volume": rng.integers(0, 5000, n),
                        "open_interest": rng.integers(0, 50000, n),
                    }
                )
            )
    raw = pd.concat(frames, ignore_index=True)[RAW_COLUMNS]
    truth_df = pd.DataFrame(truth)[["expiry", "T", "r", "forward", *PARAM_NAMES]]
    return SyntheticMarket(raw=raw, curve=curve, truth=truth_df)
