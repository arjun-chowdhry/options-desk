"""Closed-form Black-Scholes-Merton Greeks: the sensitivities a desk hedges and risk-manages.

Why closed forms rather than bumping the pricer
-----------------------------------------------
Black-Scholes Greeks have exact formulas, which are faster and more accurate than finite
differences (no bump size to tune, no rounding noise). Finite differences are still used, but
in the tests, as an independent check that every formula here is the true derivative of the
price.

Which Greeks, and why
---------------------
- delta, gamma: exposure to the underlying, and how fast that exposure changes. Delta is the
  hedge ratio; gamma is what makes delta-hedging cost or earn money (M4a).
- vega, volga, vanna: exposure to implied vol, its convexity, and the cross term with spot.
  Volga and vanna are what a desk uses to think about smile risk (M3).
- theta, charm: what time passing does to the price and to the delta hedge (charm is why a
  hedge drifts overnight even if spot doesn't move).
- rho: exposure to the interest rate.

Units (project convention)
--------------------------
- vega, volga: per 1.00 change in vol (vega 12.1 means +1 vol point, 0.20 -> 0.21, adds about
  0.121). Use ``vega_per_vol_point`` / ``volga_per_vol_point`` for per-point quotes.
- theta, charm: per year of *time passing*, i.e. minus the derivative in T. A long option
  usually has negative theta. Use ``theta_per_day`` / ``charm_per_day`` (ACT/365).
- rho: per 1.00 change in the rate.

Zero vol / at expiry
--------------------
When ``sigma * sqrt(T) == 0`` the option's value is a kinked function of spot. Away from the
kink the Greeks have clean limits: a call deep in the money has delta ``e^{-qT}`` and zero
gamma and vega. Exactly at the kink (``F == K``) delta jumps and gamma is infinite, so every
Greek returns NaN there rather than a misleading number.

Every function takes the same arguments as ``bs_price`` (including ``call``, which is ignored
by the Greeks that are identical for calls and puts) and broadcasts like NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optdesk.pricing.black_scholes import _bs_terms, _BSTerms

__all__ = [
    "charm",
    "charm_per_day",
    "delta",
    "gamma",
    "rho",
    "theta",
    "theta_per_day",
    "vanna",
    "vega",
    "vega_per_vol_point",
    "volga",
    "volga_per_vol_point",
]

DAYS_PER_YEAR = 365.0  # ACT/365, matching how T is measured


def _safe(t: _BSTerms, x: NDArray) -> NDArray:
    """Replace ``x`` by 1 where there is no uncertainty, so dividing by it is harmless.

    Only used in denominators of terms multiplied by ``n(d1)``, which is exactly 0 there.
    """
    return np.where(t.degenerate, 1.0, x)


def _finish(t: _BSTerms, value: NDArray) -> NDArray:
    """NaN at the zero-vol payoff kink (Greek undefined); unwrap 0-d arrays to scalars."""
    return np.where(t.at_kink, np.nan, value)[()]


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """dV/dS: shares of the underlying to hold to hedge one option.

    Call ``e^{-qT} N(d1)``, put ``-e^{-qT} N(-d1)``. The ``e^{-qT}`` appears because the hedge
    holds the asset and collects its dividends, so fewer shares are needed.
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    call = np.asarray(call, dtype=bool)
    value = np.where(call, t.disc_q * t.cdf_d1, -t.disc_q * t.cdf_neg_d1)
    return _finish(t, value)


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """d2V/dS2 = e^{-qT} n(d1) / (S sigma sqrt T): how fast delta changes. Same for puts."""
    t = _bs_terms(S, K, T, r, sigma, q)
    return _finish(t, t.disc_q * t.pdf_d1 / (t.S * _safe(t, t.vol_sqrt_t)))


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """dV/dsigma = S e^{-qT} n(d1) sqrt T, per 1.00 of vol. Same for puts."""
    t = _bs_terms(S, K, T, r, sigma, q)
    return _finish(t, t.S * t.disc_q * t.pdf_d1 * t.sqrt_t)


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """-dV/dT: change in value per year of time passing, everything else held fixed.

    Three pieces: the loss of optionality as uncertainty runs off (always negative for a long
    option), interest on the strike, and dividends on the asset. Call:
    ``-S e^{-qT} n(d1) sigma / (2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)``.
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    call = np.asarray(call, dtype=bool)
    decay = -t.S * t.disc_q * t.pdf_d1 * t.sigma / (2.0 * _safe(t, t.sqrt_t))
    call_theta = decay - t.r * t.K * t.disc_r * t.cdf_d2 + t.q * t.S * t.disc_q * t.cdf_d1
    put_theta = decay + t.r * t.K * t.disc_r * t.cdf_neg_d2 - t.q * t.S * t.disc_q * t.cdf_neg_d1
    return _finish(t, np.where(call, call_theta, put_theta))


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """dV/dr per 1.00 of rate. Call ``K T e^{-rT} N(d2)``, put ``-K T e^{-rT} N(-d2)``.

    Higher rates make the strike you pay later cheaper in today's money (good for calls) and
    the strike you receive later worth less (bad for puts).
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    call = np.asarray(call, dtype=bool)
    value = np.where(call, t.K * t.T * t.disc_r * t.cdf_d2, -t.K * t.T * t.disc_r * t.cdf_neg_d2)
    return _finish(t, value)


def vanna(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """d2V/dS dsigma = -e^{-qT} n(d1) d2 / sigma. Same for puts.

    Read it two ways: how delta moves when vol moves, or how vega moves when spot moves. Desks
    care because spot and vol tend to move together (equity skew).
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    return _finish(t, -t.disc_q * t.pdf_d1 * t.d2 / _safe(t, t.sigma))


def volga(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """d2V/dsigma2 = vega * d1 * d2 / sigma, per 1.00 of vol squared. Same for puts.

    Vega convexity ("vomma"). Near zero at the money and positive in both wings, so a long
    wing option gains from vol moving either way; that is one reason wings trade at higher
    implied vol than a flat-vol model would suggest.
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    vega_ = t.S * t.disc_q * t.pdf_d1 * t.sqrt_t
    return _finish(t, vega_ * t.d1 * t.d2 / _safe(t, t.sigma))


def charm(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> NDArray:
    """-d(delta)/dT: drift in delta per year of time passing ("delta decay").

    Same sign convention as theta. With ``dd1/dT = (2 (r-q) T - d2 sigma sqrt T) /
    (2 T sigma sqrt T)``, call charm is ``q e^{-qT} N(d1) - e^{-qT} n(d1) dd1/dT`` and put
    charm is ``-q e^{-qT} N(-d1) - e^{-qT} n(d1) dd1/dT``.
    """
    t = _bs_terms(S, K, T, r, sigma, q)
    call = np.asarray(call, dtype=bool)
    dd1_dT = (2.0 * (t.r - t.q) * t.T - t.d2 * t.vol_sqrt_t) / (
        2.0 * _safe(t, t.T) * _safe(t, t.vol_sqrt_t)
    )
    common = -t.disc_q * t.pdf_d1 * dd1_dT
    call_charm = t.q * t.disc_q * t.cdf_d1 + common
    put_charm = -t.q * t.disc_q * t.cdf_neg_d1 + common
    return _finish(t, np.where(call, call_charm, put_charm))


# --- Desk-quote units ------------------------------------------------------------------------


def vega_per_vol_point(*args, **kwargs) -> NDArray:
    """Vega for a 1 vol point move (0.01): ``vega / 100``. Same arguments as ``vega``."""
    return vega(*args, **kwargs) / 100.0


def volga_per_vol_point(*args, **kwargs) -> NDArray:
    """Change in per-point vega for a 1 vol point move: ``volga / 100**2``."""
    return volga(*args, **kwargs) / 100.0**2


def theta_per_day(*args, **kwargs) -> NDArray:
    """Value lost (or gained) per calendar day: ``theta / 365``. Same arguments as ``theta``."""
    return theta(*args, **kwargs) / DAYS_PER_YEAR


def charm_per_day(*args, **kwargs) -> NDArray:
    """Delta drift per calendar day: ``charm / 365``. Same arguments as ``charm``."""
    return charm(*args, **kwargs) / DAYS_PER_YEAR
