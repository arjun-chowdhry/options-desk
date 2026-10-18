"""Black-Scholes-Merton prices for European options on an asset with a continuous dividend yield.

Why this model
--------------
Black-Scholes is the market's quoting language: even desks that price with richer models
quote and risk-manage in Black-Scholes implied vol. So it is the base everything else in this
package is built on (Greeks, implied vol, the vol surface, the hedging simulator).

Why a continuous dividend yield ``q`` (Merton's 1973 extension)
---------------------------------------------------------------
Index and ETF options (SPX, SPY, QQQ) are the most liquid listed options, and an index pays a
stream of small dividends that is well approximated by a constant yield. With ``q`` the same
formula also covers FX options (``q`` = foreign interest rate) and, with ``q = r``, options on
futures (Black-76). One formula, many underlyings.

Why the forward form
--------------------
We write the price as ``DF * (F * N(d1) - K * N(d2))`` where ``F = S * exp((r - q) T)`` is the
forward and ``DF = exp(-r T)`` is the discount factor. The forward is what the option is really
written on economically (carry is folded into ``F``), and the vol-surface work later in M3
uses log-moneyness ``k = ln(K / F)``, so working in forward terms from the start keeps the
conventions consistent.

Conventions: ``T`` in years (ACT/365), ``r`` and ``q`` continuously compounded annual rates,
``sigma`` annualized. All inputs broadcast like NumPy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr  # standard normal CDF; faster than scipy.stats.norm.cdf

__all__ = ["bs_price", "d1_d2", "forward_price"]


def _validate(S: NDArray, K: NDArray, T: NDArray, sigma: NDArray) -> None:
    """Reject inputs where Black-Scholes is undefined, rather than silently returning junk."""
    if np.any(S <= 0):
        raise ValueError("spot S must be positive")
    if np.any(K <= 0):
        raise ValueError("strike K must be positive")
    if np.any(T < 0):
        raise ValueError("time to expiry T must be non-negative")
    if np.any(sigma < 0):
        raise ValueError("volatility sigma must be non-negative")


def forward_price(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> NDArray:
    """Forward price ``F = S * exp((r - q) * T)``: the no-arbitrage price today for delivery at T.

    Holding the asset earns the dividend yield ``q`` but ties up cash that could earn ``r``;
    the forward adjusts spot by that net cost of carry.
    """
    S, T, r, q = (np.asarray(x, dtype=float) for x in (S, T, r, q))
    return S * np.exp((r - q) * T)


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[NDArray, NDArray]:
    """The two standardized distances ``d1`` and ``d2`` that appear in every Black-Scholes formula.

    ``d2`` is how many standard deviations the log-forward sits above the log-strike, so
    ``N(d2)`` is the risk-neutral probability the option finishes in the money. ``d1 = d2 +
    sigma * sqrt(T)``; ``N(d1)`` is the same probability measured with the stock as numeraire,
    and for a call (with ``q = 0``) it equals the delta.

    Requires ``sigma * sqrt(T) > 0``; the pricer handles the zero case separately.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """Black-Scholes-Merton price of a European call or put.

    Parameters
    ----------
    S : spot price of the underlying (> 0).
    K : strike (> 0).
    T : time to expiry in years (>= 0).
    r : continuously compounded risk-free rate.
    sigma : annualized volatility (>= 0).
    q : continuous dividend yield (default 0).
    call : True for a call, False for a put. May be an array to price a mixed book at once.

    Returns
    -------
    Price(s), broadcast to the common shape of the inputs (a 0-d value for scalar inputs).

    Edge cases
    ----------
    When ``sigma * sqrt(T) == 0`` there is no uncertainty left: the asset will be worth the
    forward ``F`` at expiry for sure, so the option is worth its discounted intrinsic value on
    the forward, ``DF * max(F - K, 0)`` for a call. At ``T = 0`` that is plain intrinsic value
    ``max(S - K, 0)``. We return that limit instead of dividing by zero.

    The put is computed from its own formula, not from put-call parity, so the parity test in
    the test suite is a genuine check of the implementation.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    call = np.asarray(call, dtype=bool)
    _validate(S, K, T, sigma)

    F = forward_price(S, T, r, q)
    DF = np.exp(-r * T)
    vol_sqrt_t = sigma * np.sqrt(T)
    degenerate = vol_sqrt_t == 0

    # d1 in forward form: ln(F/K) / (sigma sqrt T) + (sigma sqrt T) / 2, same as d1_d2() gives.
    # Divide by 1 where sigma*sqrt(T) is zero; those entries are overwritten with the
    # no-uncertainty limit below, so the placeholder value never escapes.
    safe_vst = np.where(degenerate, 1.0, vol_sqrt_t)
    d1 = np.log(F / K) / safe_vst + 0.5 * vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    call_px = DF * (F * ndtr(d1) - K * ndtr(d2))
    put_px = DF * (K * ndtr(-d2) - F * ndtr(-d1))

    call_px = np.where(degenerate, DF * np.maximum(F - K, 0.0), call_px)
    put_px = np.where(degenerate, DF * np.maximum(K - F, 0.0), put_px)

    return np.where(call, call_px, put_px)[()]
