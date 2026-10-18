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

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr  # standard normal CDF; faster than scipy.stats.norm.cdf

__all__ = ["bs_price", "d1_d2", "forward_price"]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


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


class _BSTerms(NamedTuple):
    """Intermediate quantities shared by ``bs_price`` and the Greeks (internal)."""

    S: NDArray
    K: NDArray
    T: NDArray
    r: NDArray
    sigma: NDArray
    q: NDArray
    F: NDArray  # forward
    disc_r: NDArray  # e^{-rT}, discount factor
    disc_q: NDArray  # e^{-qT}, dividend discount factor
    sqrt_t: NDArray
    vol_sqrt_t: NDArray
    d1: NDArray
    d2: NDArray
    cdf_d1: NDArray  # N(d1)
    cdf_d2: NDArray  # N(d2)
    cdf_neg_d1: NDArray  # N(-d1), kept separately: 1 - N(d1) loses precision deep in the money
    cdf_neg_d2: NDArray  # N(-d2)
    pdf_d1: NDArray  # n(d1), standard normal density
    degenerate: NDArray  # sigma * sqrt(T) == 0: no uncertainty left
    at_kink: NDArray  # degenerate and F == K: payoff kink, where Greeks don't exist


def _bs_terms(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> _BSTerms:
    """Validate inputs and compute everything the price and Greek formulas need.

    When ``sigma * sqrt(T) == 0`` the terminal asset price is the forward for sure, so
    ``d1`` and ``d2`` are effectively +infinity (if ``F > K``) or -infinity (if ``F < K``).
    We plug in the limits directly: ``N(d) = 1`` or ``0`` and ``n(d1) = 0``. With those, the
    ordinary formulas give the correct no-uncertainty price and Greeks with no special-casing
    downstream. ``d1``/``d2`` themselves hold finite placeholders there; every formula that
    uses them multiplies by ``n(d1) = 0``.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    _validate(S, K, T, sigma)

    F = forward_price(S, T, r, q)
    sqrt_t = np.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    degenerate = vol_sqrt_t == 0

    # d1 in forward form: ln(F/K) / (sigma sqrt T) + (sigma sqrt T) / 2, same as d1_d2() gives.
    # Divide by 1 where sigma*sqrt(T) is zero; those entries are replaced by limits below.
    safe_vst = np.where(degenerate, 1.0, vol_sqrt_t)
    d1 = np.log(F / K) / safe_vst + 0.5 * vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # Limit of N(d) as sigma*sqrt(T) -> 0: 1 in the money, 0 out of it, 1/2 exactly at the money.
    itm = (F > K) + 0.5 * (F == K)
    return _BSTerms(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        F=F,
        disc_r=np.exp(-r * T),
        disc_q=np.exp(-q * T),
        sqrt_t=sqrt_t,
        vol_sqrt_t=vol_sqrt_t,
        d1=d1,
        d2=d2,
        cdf_d1=np.where(degenerate, itm, ndtr(d1)),
        cdf_d2=np.where(degenerate, itm, ndtr(d2)),
        cdf_neg_d1=np.where(degenerate, 1.0 - itm, ndtr(-d1)),
        cdf_neg_d2=np.where(degenerate, 1.0 - itm, ndtr(-d2)),
        pdf_d1=np.where(degenerate, 0.0, _INV_SQRT_2PI * np.exp(-0.5 * d1**2)),
        degenerate=degenerate,
        at_kink=degenerate & (F == K),
    )


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
    t = _bs_terms(S, K, T, r, sigma, q)
    call = np.asarray(call, dtype=bool)
    # In the zero-vol limit N(d1) = N(d2) = 1 or 0, which turns these into DF * max(F - K, 0)
    # and DF * max(K - F, 0); see _bs_terms.
    call_px = t.disc_r * (t.F * t.cdf_d1 - t.K * t.cdf_d2)
    put_px = t.disc_r * (t.K * t.cdf_neg_d2 - t.F * t.cdf_neg_d1)
    return np.where(call, call_px, put_px)[()]
