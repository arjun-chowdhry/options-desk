"""Raw SVI: a five-parameter smile for one expiry, and a robust way to fit it to market vols.

What SVI is
-----------
SVI ("stochastic volatility inspired", Gatheral 2004) describes one expiry's smile as total
implied variance ``w = sigma_impl^2 * T`` against log-moneyness ``k = ln(K / F)``:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Each parameter has a picture: ``a`` sets the overall variance level, ``b`` the steepness of
the wings, ``rho`` (between -1 and 1) tilts the smile (negative for equities: puts are richer),
``m`` slides it left or right, and ``sigma`` rounds off the bottom (small sigma = sharp V).

Why SVI and not a polynomial or a spline
----------------------------------------
- **Right shape far from the money.** For large ``|k|`` SVI's total variance grows
  *linearly*, with slopes ``b(1 - rho)`` (left) and ``b(1 + rho)`` (right). Roger Lee's moment
  formula proves no arbitrage-free smile can grow faster than linearly (slope <= 2). A
  quadratic in ``k`` grows faster and is guaranteed to be wrong in the wings; a spline says
  nothing at all beyond the last strike.
- **Five numbers per expiry,** each with a meaning. That is compact enough to store, compare
  day to day, and talk about ("skew steepened": ``rho`` more negative).
- **It fits.** It was designed at Merrill Lynch from years of SPX data, and it is the limit of
  the Heston model for long expiries, so it matches how equity smiles actually look.
- **Its arbitrage conditions are checkable** with closed-form derivatives (``arbitrage.py``).

Why we fit in implied vol, weighted by vega
-------------------------------------------
Residuals are in vol units (``sigma_model - sigma_market``), so the fit error reads directly
as "vol points", the unit traders quote. Each residual is weighted by the option's vega
(normalized to average 1). Vega is how much the price moves per unit of vol, so it measures
how much vol information a quote really carries: a deep-OTM option worth a few cents has tiny
vega, and half a cent of quote noise there is several vol points of noise. Near the money,
vega is large and the vol is pinned down precisely. Weighting by vega (rather than vega
squared, which is what fitting raw prices amounts to) trusts the money-ness region most but
still leaves the wings enough weight to anchor the tails.

How the fit avoids bad local minima
-----------------------------------
A plain 5-parameter optimizer started from one guess often gets stuck. SVI has a structure we
can exploit (the "quasi-explicit" idea of Zeliade Systems, 2009): with ``m`` and ``sigma``
held fixed, write ``y = (k - m) / sigma`` and the formula becomes

    w = a + (b rho sigma) * y + (b sigma) * sqrt(y^2 + 1)

which is *linear* in three unknowns, so ordinary least squares solves it exactly. We scan a
grid of ``(m, sigma)`` pairs, solve the linear problem at each (cheap), and start the full
nonlinear fit (``scipy.optimize.least_squares``) from the best few. The final answer is the
best of those polished fits.

Constraints. Box bounds keep every parameter sensible (``b >= 0``, ``|rho| < 1``,
``sigma > 0``, ``m`` near the quoted strikes). Two conditions that aren't simple boxes are
added as penalty terms: total variance must stay non-negative everywhere (its minimum is
``a + b sigma sqrt(1 - rho^2)``), and the steeper wing must respect Lee's bound
``b (1 + |rho|) <= 2``. Full butterfly and calendar checks live in ``arbitrage.py``.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from optdesk.pricing.greeks import vega

__all__ = [
    "PARAM_NAMES",
    "SVIFit",
    "SVIParams",
    "fit_chain",
    "fit_svi",
    "ssvi_slice",
    "svi_derivatives",
    "svi_total_variance",
]

PARAM_NAMES = ["a", "b", "rho", "m", "sigma"]
LEE_SLOPE = 2.0  # max wing slope of total variance (Roger Lee's moment formula)
RHO_MAX = 0.999
PENALTY = 1e3
N_STARTS = 3  # polish this many of the best grid starting points


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters for one expiry (total variance in ``k = ln(K/F)``)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: ArrayLike) -> NDArray:
        return svi_total_variance(k, *astuple(self))

    def implied_vol(self, k: ArrayLike, T: float) -> NDArray:
        """``sqrt(w / T)``; NaN where the total variance is negative (an invalid slice)."""
        w = self.total_variance(k)
        with np.errstate(invalid="ignore"):
            return np.sqrt(w / T)

    @property
    def min_variance(self) -> float:
        """Lowest total variance over all ``k`` (at ``k = m - rho sigma / sqrt(1 - rho^2)``)."""
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)

    @property
    def wing_slopes(self) -> tuple[float, float]:
        """Asymptotic slopes of total variance: (left wing, right wing)."""
        return self.b * (1 - self.rho), self.b * (1 + self.rho)

    @classmethod
    def from_mapping(cls, row) -> SVIParams:
        """Build from anything indexable by the parameter names (dict, Series, DataFrame row)."""
        return cls(*(float(row[name]) for name in PARAM_NAMES))


def svi_total_variance(
    k: ArrayLike, a: float, b: float, rho: float, m: float, sigma: float
) -> NDArray:
    """Raw SVI total variance ``a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``."""
    x = np.asarray(k, dtype=float) - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def svi_derivatives(k: ArrayLike, params: SVIParams) -> tuple[NDArray, NDArray, NDArray]:
    """Total variance and its first two derivatives in ``k``: ``(w, w', w'')``.

    ``w' = b (rho + x / R)`` and ``w'' = b sigma^2 / R^3`` with ``x = k - m`` and
    ``R = sqrt(x^2 + sigma^2)``. ``w''`` is always >= 0: every SVI slice is convex.
    """
    x = np.asarray(k, dtype=float) - params.m
    R = np.sqrt(x * x + params.sigma**2)
    w = params.a + params.b * (params.rho * x + R)
    dw = params.b * (params.rho + x / R)
    d2w = params.b * params.sigma**2 / R**3
    return w, dw, d2w


def ssvi_slice(theta: float, rho: float, phi: float) -> SVIParams:
    """Raw SVI parameters of one slice of an SSVI surface (Gatheral & Jacquier 2014).

    SSVI writes a slice as ``w(k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``
    with ``theta`` the ATM total variance. It is exactly raw SVI with
    ``a = theta (1 - rho^2) / 2``, ``b = theta phi / 2``, ``m = -rho / phi`` and
    ``sigma = sqrt(1 - rho^2) / phi``. Used to build the synthetic market.
    """
    return SVIParams(
        a=theta * (1 - rho**2) / 2,
        b=theta * phi / 2,
        rho=rho,
        m=-rho / phi,
        sigma=np.sqrt(1 - rho**2) / phi,
    )


@dataclass(frozen=True)
class SVIFit:
    """Result of fitting one expiry."""

    params: SVIParams
    T: float
    rmse: float  # weighted root-mean-square vol error (1.00 = 100 vol points)
    max_error: float  # largest absolute vol error at any quote
    n: int  # number of quotes fitted
    success: bool  # optimizer converged and the penalty constraints hold
    message: str


def _bounds(k: NDArray, w: NDArray) -> tuple[NDArray, NDArray]:
    span = max(k.max() - k.min(), 0.01)
    lower = np.array([-w.max(), 0.0, -RHO_MAX, k.min() - span, 1e-4])
    upper = np.array([w.max(), LEE_SLOPE, RHO_MAX, k.max() + span, 2 * span])
    return lower, upper


def _grid_starts(k: NDArray, w: NDArray, row_scale: NDArray, lower, upper) -> list[NDArray]:
    """Best starting points from the linear sub-problem on a grid of ``(m, sigma)``."""
    span = max(k.max() - k.min(), 0.01)
    candidates = []
    for m in np.linspace(k.min(), k.max(), 9):
        for sigma in np.geomspace(0.02 * span, span, 9):
            y = (k - m) / sigma
            X = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1)])
            (a, d, c), *_ = np.linalg.lstsq(X * row_scale[:, None], w * row_scale, rcond=None)
            b = max(c / sigma, 1e-6)
            rho = np.clip(d / c, -RHO_MAX, RHO_MAX) if c > 0 else 0.0
            x0 = np.clip([a, b, rho, m, sigma], lower, upper)
            err = (svi_total_variance(k, *x0) - w) * row_scale
            candidates.append((float(err @ err), x0))
    candidates.sort(key=lambda item: item[0])
    return [x0 for _, x0 in candidates[:N_STARTS]]


def fit_svi(
    k: ArrayLike,
    iv: ArrayLike,
    T: float,
    weights: ArrayLike | None = None,
) -> SVIFit:
    """Fit raw SVI to one expiry's implied vols.

    Parameters
    ----------
    k : log-moneyness ``ln(K / F)`` of each quote.
    iv : market implied vol of each quote (annualized).
    T : time to expiry in years.
    weights : relative importance of each quote (e.g. vega); default equal. Normalized to
        average 1 internally, so only their ratios matter.

    Returns
    -------
    ``SVIFit`` with the parameters, the weighted RMS and max vol errors, and a success flag.
    Needs at least 5 quotes (one per parameter).
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if k.size < len(PARAM_NAMES):
        raise ValueError(f"need at least {len(PARAM_NAMES)} quotes to fit SVI, got {k.size}")
    wt = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wt = wt / wt.mean()
    sqrt_wt = np.sqrt(wt)
    w_mkt = iv * iv * T
    lower, upper = _bounds(k, w_mkt)
    w_scale = w_mkt.mean()

    def residuals(x: NDArray) -> NDArray:
        p = SVIParams(*x)
        model_vol = np.sqrt(np.maximum(p.total_variance(k), 1e-16) / T)
        negative_variance = max(0.0, -p.min_variance) / w_scale
        lee_excess = max(0.0, p.b * (1 + abs(p.rho)) - LEE_SLOPE)
        return np.concatenate(
            [sqrt_wt * (model_vol - iv), PENALTY * np.array([negative_variance, lee_excess])]
        )

    # Vol error ~ variance error / (2 sigma T): scale rows so the linear stage mimics vol space.
    row_scale = sqrt_wt / (2 * iv * T)
    best = None
    for x0 in _grid_starts(k, w_mkt, row_scale, lower, upper):
        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            max_nfev=5000,
        )
        if best is None or result.cost < best.cost:
            best = result

    params = SVIParams(*best.x)
    errors = params.implied_vol(k, T) - iv
    constraints_ok = params.min_variance >= 0 and params.b * (1 + abs(params.rho)) <= LEE_SLOPE
    return SVIFit(
        params=params,
        T=float(T),
        rmse=float(np.sqrt(np.mean(wt * errors**2))),
        max_error=float(np.max(np.abs(errors))),
        n=int(k.size),
        success=bool(best.success and constraints_ok),
        message=best.message if constraints_ok else "fit violates variance or Lee constraint",
    )


def fit_chain(clean: pd.DataFrame, *, weighting: str = "vega") -> pd.DataFrame:
    """Fit SVI to every expiry of a cleaned chain (the output of ``data.clean.clean_chain``).

    ``weighting`` is ``"vega"`` (default, see the module notes) or ``"equal"``. Expiries with
    fewer than 5 quotes are skipped.

    Returns one row per expiry: ``expiry``, ``T``, ``forward``, the five ``PARAM_NAMES``,
    ``rmse``, ``max_error``, ``n`` and ``success``, sorted by ``T``.
    """
    if weighting not in ("vega", "equal"):
        raise ValueError("weighting must be 'vega' or 'equal'")
    rows = []
    for expiry, group in clean.groupby("expiry"):
        if len(group) < len(PARAM_NAMES):
            continue
        T, F = group["T"].iloc[0], group["forward"].iloc[0]
        weights = None
        if weighting == "vega":
            # Vega on the forward (S = F, q = r), the same convention the cleaner used for IV.
            r = group["r"].to_numpy()
            weights = vega(F, group["strike"].to_numpy(), T, r, group["iv"].to_numpy(), r)
        fit = fit_svi(group["k"].to_numpy(), group["iv"].to_numpy(), T, weights)
        rows.append(
            {
                "expiry": expiry,
                "T": fit.T,
                "forward": F,
                **dict(zip(PARAM_NAMES, astuple(fit.params), strict=True)),
                "rmse": fit.rmse,
                "max_error": fit.max_error,
                "n": fit.n,
                "success": fit.success,
            }
        )
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
