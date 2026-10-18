"""Cox-Ross-Rubinstein (CRR) binomial tree, for American as well as European options.

Why a tree at all
-----------------
Black-Scholes only prices European options, which can be exercised at expiry only. Listed
single-stock options in the US are American: the holder can exercise any day. Early exercise
can be worth something (a deep in-the-money put, or a call just before a large dividend),
and there is no closed form for it. A tree handles it naturally. Walking backward from expiry,
at every node the option is worth the larger of holding on and exercising now.

How it works
------------
Split ``[0, T]`` into ``N`` steps of length ``dt``. In each step the asset moves up by a
factor ``u`` or down by ``d``. The up-probability ``p`` is chosen so the asset earns the
risk-free rate net of dividends on average (``p u + (1 - p) d = e^{(r-q) dt}``), which is the
same risk-neutral pricing as Black-Scholes. Then, from the last step backward, each node's
value is the discounted average of its two children.

Why CRR rather than Jarrow-Rudd or other trees
----------------------------------------------
CRR picks ``u = e^{sigma sqrt(dt)}`` and ``d = 1/u``. Because ``u * d = 1``, an up move
followed by a down move returns to the same price, so the tree *recombines*: after ``i`` steps
there are only ``i + 1`` distinct prices, not ``2^i``. The grid is centred on today's spot and
is the same for every rate and dividend, and only ``p`` carries the drift. Jarrow-Rudd instead
fixes ``p = 1/2`` and puts the drift into ``u`` and ``d``. That's equally valid and has the
same convergence rate, but CRR is the textbook standard (Hull), easier to explain, and its
symmetric grid makes the early-exercise boundary easy to read off. As ``N`` grows, the CRR
price converges to Black-Scholes for European options, with error shrinking like ``1/N``.
The convergence oscillates between odd and even ``N``; averaging ``N`` and ``N + 1`` steps
damps that, which the tests use.

Requires ``0 <= p <= 1``. That fails only when ``dt`` is so large that the drift over one
step beats the vol (``|r - q| sqrt(dt) > sigma`` roughly); more steps fixes it, so we raise an
error and say so rather than return a price from negative probabilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optdesk.pricing.black_scholes import _validate

__all__ = ["binomial_price"]


def binomial_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
    *,
    american: bool = True,
    steps: int = 500,
) -> NDArray:
    """Price European or American vanilla options on a CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q, call : as in ``bs_price``. Requires ``sigma > 0`` when ``T > 0``
        (with no vol, the tree has nowhere to branch).
    american : allow exercise at every node (True, default) or only at expiry (False).
    steps : number of time steps ``N``. Error is O(1/N); 500 gives about a cent on a $100
        stock.

    Returns
    -------
    Price(s), broadcast to the common shape of the inputs. At ``T = 0`` the price is the
    intrinsic value.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    shape = arrays[0].shape
    # Column vectors: one row per option, so the tree for every option is built at once.
    S, K, T, r, sigma, q = (a.reshape(-1, 1) for a in arrays)
    call = np.broadcast_to(np.asarray(call, dtype=bool), shape).reshape(-1, 1)
    _validate(S, K, T, sigma)
    if np.any((sigma == 0) & (T > 0)):
        raise ValueError("binomial tree needs sigma > 0 (use bs_price for the zero-vol limit)")

    dt = T / steps
    log_u = sigma * np.sqrt(dt)
    u, d = np.exp(log_u), np.exp(-log_u)
    growth = np.exp((r - q) * dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (growth - d) / (u - d)  # 0/0 where T = 0; those options are handled at the end
    expiring = T == 0
    p = np.where(expiring, 0.5, p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(
            "risk-neutral probability outside [0, 1]: drift beats vol over one step; increase steps"
        )
    disc = np.exp(-r * dt)

    def payoff(spot: NDArray) -> NDArray:
        return np.where(call, np.maximum(spot - K, 0.0), np.maximum(K - spot, 0.0))

    # Node j at step i has had j up moves and i - j down moves: price S * u^(2j - i).
    j = np.arange(steps + 1)
    value = payoff(S * np.exp(log_u * (2 * j - steps)))
    for i in range(steps - 1, -1, -1):
        # Children of node j at step i are j (down) and j + 1 (up) at step i + 1.
        value = disc * (p * value[:, 1 : i + 2] + (1 - p) * value[:, : i + 1])
        if american:
            value = np.maximum(value, payoff(S * np.exp(log_u * (2 * j[: i + 1] - i))))

    price = np.where(expiring, payoff(S), value)
    return price.reshape(shape)[()]
