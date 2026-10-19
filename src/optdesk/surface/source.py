"""Vol surface sources: one small interface so the simulators don't care where vols come from.

Why an interface
----------------
The market maker (``sim/market_maker.py``) needs one thing from a surface: the implied vol at a
log-moneyness ``k = ln(K/F)`` and an expiry ``T``. Today the only clean surface we have is the
synthetic one; from Monday there will be SVI fits of live SPY snapshots. Coding against the
``VolSurface`` protocol (anything with an ``implied_vol(k, T)`` method) means swapping one for
the other is a one-line change, and tests can use a flat surface where every answer is known.

Three sources
-------------
- ``SSVISurface`` / ``synthetic_surface()``: the synthetic market's *true* surface
  (``surface/synthetic.py``). It's analytic in ``T``, so it gives a vol at any expiry, not
  just the listed ones, and it is arbitrage-free by construction.
- ``SVISurface.from_fit(table)``: a surface built from per-expiry raw SVI parameters, the
  output of ``svi.fit_chain`` (live data) or the synthetic ``truth`` table. Between fitted
  expiries it interpolates **total variance linearly in T at fixed k**. That's the standard
  choice because it keeps ``w`` non-decreasing in ``T`` whenever the slices are
  (no calendar arbitrage is introduced). Outside the fitted range it holds each strike's
  vol constant (``w`` proportional to ``T``), the most neutral extrapolation.
- ``FlatSurface``: one vol everywhere. Black-Scholes's world; useful for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from optdesk.surface.svi import SVIParams
from optdesk.surface.synthetic import SSVI_RHO, atm_vol, ssvi_phi

__all__ = [
    "FlatSurface",
    "SSVISurface",
    "SVISurface",
    "VolSurface",
    "fitted_surface",
    "synthetic_surface",
]


@runtime_checkable
class VolSurface(Protocol):
    """Anything that returns an implied vol for log-moneyness ``k`` and expiry ``T``."""

    def implied_vol(self, k: ArrayLike, T: ArrayLike) -> NDArray: ...


@dataclass(frozen=True)
class FlatSurface:
    """The same implied vol at every strike and expiry."""

    vol: float

    def implied_vol(self, k: ArrayLike, T: ArrayLike) -> NDArray:
        return np.full(np.broadcast(np.asarray(k), np.asarray(T)).shape, float(self.vol))[()]


@dataclass(frozen=True)
class SSVISurface:
    """The synthetic market's true SSVI surface (see ``surface/synthetic.py``).

    ``w(k, T) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`` with
    ``theta(T) = atm_vol(T)^2 T`` and the power-law ``phi(theta)``.
    """

    rho: float = SSVI_RHO

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> NDArray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        theta = atm_vol(T) ** 2 * T
        phi = ssvi_phi(theta)
        rho = self.rho
        return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))

    def implied_vol(self, k: ArrayLike, T: ArrayLike) -> NDArray:
        T = np.asarray(T, dtype=float)
        return np.sqrt(self.total_variance(k, T) / T)[()]


@dataclass(frozen=True)
class SVISurface:
    """A surface from raw SVI slices, interpolated linearly in total variance between them."""

    T: NDArray  # slice expiries, increasing
    slices: tuple[SVIParams, ...]

    def __post_init__(self) -> None:
        if len(self.T) == 0 or len(self.T) != len(self.slices):
            raise ValueError("need one SVI slice per expiry, and at least one")
        if np.any(np.diff(self.T) <= 0):
            raise ValueError("slice expiries must be strictly increasing")

    @classmethod
    def from_fit(cls, table: pd.DataFrame) -> SVISurface:
        """Build from a table with ``T`` and the SVI parameter columns (``fit_chain`` output)."""
        table = table.sort_values("T")
        slices = tuple(SVIParams.from_mapping(row) for _, row in table.iterrows())
        return cls(T=table["T"].to_numpy(dtype=float), slices=slices)

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> NDArray:
        k, T = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        # w of every slice at every k: shape (n_slices, *k.shape)
        w_slices = np.stack([p.total_variance(k) for p in self.slices])
        n = len(self.T)
        j = np.clip(np.searchsorted(self.T, T, side="right") - 1, 0, max(n - 2, 0))
        if n == 1:
            return w_slices[0] * T / self.T[0]
        T_lo, T_hi = self.T[j], self.T[j + 1]
        w_lo = np.take_along_axis(w_slices, j[None], axis=0)[0]
        w_hi = np.take_along_axis(w_slices, (j + 1)[None], axis=0)[0]
        inside = w_lo + (T - T_lo) / (T_hi - T_lo) * (w_hi - w_lo)
        # Outside the fitted range keep each strike's vol fixed: w scales with T.
        short = w_slices[0] * T / self.T[0]
        long = w_slices[-1] * T / self.T[-1]
        return np.where(T < self.T[0], short, np.where(T > self.T[-1], long, inside))

    def implied_vol(self, k: ArrayLike, T: ArrayLike) -> NDArray:
        T = np.asarray(T, dtype=float)
        w = self.total_variance(k, T)
        with np.errstate(invalid="ignore"):
            return np.sqrt(w / T)[()]


def synthetic_surface() -> SSVISurface:
    """The synthetic SPY surface: the default until live fits exist."""
    return SSVISurface()


def fitted_surface(table: pd.DataFrame) -> SVISurface:
    """A surface from an SVI fit table, e.g. ``fit_chain(clean_chain(snapshot))``."""
    return SVISurface.from_fit(table)
