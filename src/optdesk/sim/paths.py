"""Simulated price paths: geometric Brownian motion (exact) and Heston (full-truncation Euler).

Why two models
--------------
- **GBM** is the world Black-Scholes assumes: constant vol, log-normal prices. In it we know
  every answer in closed form (the option's value, its Greeks, the P&L of hedging it), so it
  is the lab where the hedging engine is validated.
- **Heston** lets the variance itself wander: it mean-reverts towards a long-run level and is
  negatively correlated with the price (equity selloffs come with vol spikes). It is the
  standard first step beyond Black-Scholes, and it shows what happens to a Black-Scholes
  delta hedge when the real world doesn't have constant vol.

Why exact simulation for GBM
----------------------------
GBM has an exact solution, ``S_t+dt = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)``, so
we step with that instead of an Euler approximation. There is no discretization error at
all: one step to expiry gives the same distribution as a thousand. The time grid then only
matters for how often the hedger looks at the price, which is exactly the question M4a
studies.

Why full-truncation Euler for Heston
------------------------------------
Heston's variance ``v`` follows ``dv = kappa (theta - v) dt + xi sqrt(v) dW``. The true
process never goes negative, but any simple discretization can step below zero, where
``sqrt(v)`` is undefined. The common fixes are:

- *absorption* (set ``v = 0``) or *reflection* (use ``|v|``): simple, but both push the
  variance up and bias prices noticeably;
- *full truncation* (Lord, Koekkoek & van Dijk 2010): let the simulated ``v`` go negative but
  use ``v+ = max(v, 0)`` everywhere it enters the drift and the diffusion. Of the simple
  Euler fixes it has the smallest bias, which their paper measures across schemes;
- *exact* (Broadie-Kaya) or *quadratic-exponential* (Andersen) schemes: more accurate per
  step, but much more code and slower, which buys little when we already take hundreds of
  small steps for hourly hedging.

So: full truncation for ``v``, and a log-Euler step for the price
(``S_t+dt = S_t exp((mu - v+/2) dt + sqrt(v+ dt) Z_S)``), which keeps ``S`` positive and is
exact whenever ``v`` happens to be constant.

Conventions
-----------
``mu`` is the drift of the *price* (net of dividends): use ``mu = r - q`` for the risk-neutral
world, or a real-world drift to check that hedged P&L doesn't care. Time is in years, vol and
variance are annual. All randomness comes from a seeded ``np.random.Generator``, so a seed
reproduces every path.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["Paths", "gbm_paths", "heston_paths", "time_grid"]


class Paths(NamedTuple):
    """Simulated paths on a time grid.

    ``times`` has shape ``(n_steps + 1,)`` and starts at 0. ``S`` and ``v`` have shape
    ``(n_paths, n_steps + 1)``. ``v`` is the instantaneous variance each path actually
    experienced over the step starting at each time: ``sigma^2`` everywhere for GBM, the
    truncated ``max(v, 0)`` for Heston. The hedging module uses it for the gamma-P&L formula.
    """

    times: NDArray
    S: NDArray
    v: NDArray


def time_grid(T: float, n_steps: int) -> NDArray:
    """Equally spaced times ``0, T/n, ..., T``."""
    if T <= 0:
        raise ValueError("horizon T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    return np.linspace(0.0, T, n_steps + 1)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Accept a Generator or a seed (an int) and return a Generator."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | int | None = None,
) -> Paths:
    """Exact geometric Brownian motion paths.

    Parameters
    ----------
    S0 : starting price (> 0).
    mu : annual drift of the price (``r - q`` for risk-neutral paths).
    sigma : annual volatility (>= 0), the vol the paths will actually realize.
    T : horizon in years.
    n_steps : number of equal time steps.
    n_paths : number of independent paths.
    rng : ``np.random.Generator`` or an integer seed.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    times = time_grid(T, n_steps)
    dt = T / n_steps
    z = _rng(rng).standard_normal((n_paths, n_steps))
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_S = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    S = S0 * np.exp(log_S)
    # constant variance: a read-only broadcast view, no memory used
    v = np.broadcast_to(np.float64(sigma**2), S.shape)
    return Paths(times=times, S=S, v=v)


def heston_paths(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | int | None = None,
) -> Paths:
    """Heston stochastic-volatility paths by full-truncation Euler (see module notes).

    ``dS/S = mu dt + sqrt(v) dW_S`` and ``dv = kappa (theta - v) dt + xi sqrt(v) dW_v``,
    with ``corr(dW_S, dW_v) = rho``.

    Parameters
    ----------
    S0, mu, T, n_steps, n_paths, rng : as in ``gbm_paths``.
    v0 : starting variance (vol squared; 0.04 = 20% vol).
    kappa : speed of mean reversion (per year; the half-life is ``ln 2 / kappa``).
    theta : long-run variance that ``v`` reverts to.
    xi : vol of variance ("vol of vol").
    rho : correlation between price and variance shocks, in [-1, 1]; negative for equities.

    The Feller condition ``2 kappa theta >= xi^2`` (which keeps the true ``v`` off zero) is
    not required. Fitted equity parameters often break it, and full truncation copes.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if v0 < 0 or theta < 0 or kappa < 0 or xi < 0:
        raise ValueError("v0, theta, kappa and xi must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    times = time_grid(T, n_steps)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    gen = _rng(rng)

    log_S = np.empty((n_paths, n_steps + 1))
    v_used = np.empty((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    v = np.full(n_paths, float(v0))  # the raw Euler variance, which may dip below zero
    for i in range(n_steps):
        z_s = gen.standard_normal(n_paths)
        z_v = rho * z_s + np.sqrt(1.0 - rho**2) * gen.standard_normal(n_paths)
        v_plus = np.maximum(v, 0.0)
        v_used[:, i] = v_plus
        log_S[:, i + 1] = log_S[:, i] + (mu - 0.5 * v_plus) * dt + np.sqrt(v_plus) * sqrt_dt * z_s
        v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus) * sqrt_dt * z_v
    v_used[:, -1] = np.maximum(v, 0.0)
    return Paths(times=times, S=np.exp(log_S), v=v_used)
