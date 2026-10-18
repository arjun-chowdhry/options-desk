"""Black-Scholes implied volatility: the vol that makes the model price equal a market price.

Why invert at all
-----------------
Traders quote options in implied vol, not dollars, because vol is comparable across strikes,
expiries and underlyings. Every later piece of this package (chain cleaning, the SVI surface,
the dashboard) starts by turning market prices into implied vols with this function.

Why Newton's method with a Brent fallback
-----------------------------------------
Newton uses the slope of price in vol, which is vega, and we have vega in closed form. Near
the answer it converges quadratically (the number of correct digits roughly doubles each step),
so a handful of iterations gives machine precision on a whole chain at once, fully vectorized.

Newton can fail where vega is tiny (far out of the money, very short expiry): a small slope
means a huge step that can overshoot to a negative or absurd vol. Brent's method is a
bracketing root-finder: given a vol interval where the price error changes sign, it always
converges. It is slower and runs one option at a time, so we use it only for the options where
Newton didn't converge. This gives Newton's speed on typical options and Brent's reliability
on the hard ones.

Why start Newton at sqrt(2 |ln(F/K)| / T) (Manaster-Koehler)
------------------------------------------------------------
The Black-Scholes price as a function of vol is convex below this vol and concave above it
(it is the inflection point). Starting Newton at the inflection point makes the iterates move
monotonically toward the root instead of overshooting, which is what makes the plain method
reliable for most inputs. At the money the inflection point is 0, where vega vanishes, so the
guess is floored at a small positive vol.

Why invert the out-of-the-money option
--------------------------------------
An in-the-money call is mostly intrinsic value plus a little time value, and only the time
value carries information about vol. Inverting it directly loses digits. Put-call parity
``C - P = S e^{-qT} - K e^{-rT}`` turns it into the matching out-of-the-money put with the
same implied vol, whose price is pure time value. So we always solve on the OTM side (the
call when ``F <= K``, the put when ``F > K``).

When the answer is NaN
----------------------
A European option's price must lie strictly between its no-arbitrage bounds,
``max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}`` for a call and
``max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}`` for a put. Outside them, no vol
reproduces the price: the quote is stale or an arbitrage. On the lower bound itself the price
has no time value, and on the upper bound the vol is infinite; in both cases the vol isn't
identifiable, so we return NaN rather than invent a number. ``T = 0`` gives NaN for the same
reason.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from optdesk.pricing.black_scholes import _validate, bs_price, forward_price
from optdesk.pricing.greeks import vega

__all__ = ["implied_vol"]

MIN_GUESS = 0.05  # floor on Newton's starting vol (the inflection point is 0 at the money)
VOL_LO = 1e-8  # Brent's search interval for vol: 1e-6 % up to 10,000 %
VOL_HI = 100.0


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> NDArray:
    """Implied volatility of European option prices, NaN where none exists.

    Parameters
    ----------
    price : market price(s) of the option.
    S, K, T, r, q, call : as in ``bs_price``.
    tol : Newton stops once the vol update is smaller than this.
    max_iter : Newton iteration cap; options not converged by then go to Brent. (Setting it
        to 0 sends everything to Brent, which the tests use to check the fallback.)

    Returns
    -------
    Annualized implied vol(s), broadcast to the common shape of the inputs.
    """
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)))
    shape = arrays[0].shape
    price, S, K, T, r, q = (a.ravel() for a in arrays)
    call = np.broadcast_to(np.asarray(call, dtype=bool), shape).ravel()
    _validate(S, K, T, np.zeros(1))

    pv_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    F = forward_price(S, T, r, q)

    # Solve on the out-of-the-money side; put-call parity converts the price (same vol).
    solve_call = F <= K
    target = price.copy()
    target[call & ~solve_call] -= pv_s[call & ~solve_call] - pv_k[call & ~solve_call]
    target[~call & solve_call] += pv_s[~call & solve_call] - pv_k[~call & solve_call]

    # OTM bounds are 0 < price < S e^{-qT} (call) or K e^{-rT} (put); equivalent to the
    # bounds on the original option. NaN inputs fail these comparisons and stay NaN.
    upper = np.where(solve_call, pv_s, pv_k)
    valid = (T > 0) & (target > 0) & (target < upper)

    sigma = np.full(price.shape, np.nan)
    with np.errstate(divide="ignore"):
        guess = np.sqrt(2.0 * np.abs(np.log(F / K)) / np.where(T > 0, T, 1.0))
    sigma[valid] = np.maximum(guess[valid], MIN_GUESS)

    converged = _newton(sigma, target, S, K, T, r, q, solve_call, valid, tol, max_iter)

    for i in np.flatnonzero(valid & ~converged):
        sigma[i] = _brent(target[i], S[i], K[i], T[i], r[i], q[i], solve_call[i])

    sigma[~valid] = np.nan
    return sigma.reshape(shape)[()]


def _newton(sigma, target, S, K, T, r, q, is_call, valid, tol, max_iter):
    """Vectorized Newton iterations, updating ``sigma`` in place on the ``valid`` entries.

    Returns a mask of entries that converged. An entry drops out (unconverged, for Brent) as
    soon as a step produces a non-finite or out-of-range vol.
    """
    active = valid.copy()
    converged = np.zeros_like(valid)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        args = (S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx], is_call[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            step = (bs_price(*args) - target[idx]) / vega(*args)
        new = sigma[idx] - step

        failed = ~np.isfinite(new) | (new <= 0) | (new > VOL_HI)
        done = ~failed & (np.abs(step) < tol)
        ok = ~failed
        sigma[idx[ok]] = new[ok]
        converged[idx[done]] = True
        active[idx[failed | done]] = False
    return converged


def _brent(target, S, K, T, r, q, is_call) -> float:
    """Bracketing fallback for one option: find a sign change, then run Brent's method."""

    def error(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, q, is_call)) - target

    if error(VOL_LO) > 0:
        return np.nan  # price is below what even a near-zero vol gives: no time value to fit
    hi = 1.0
    while error(hi) < 0:
        if hi >= VOL_HI:
            return np.nan
        hi = min(2.0 * hi, VOL_HI)
    return brentq(error, VOL_LO, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
