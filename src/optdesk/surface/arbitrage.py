"""Static-arbitrage checks for SVI slices: butterfly (within an expiry) and calendar (across).

A fitted surface is only usable for pricing if it is free of *static arbitrage*: no
combination of options that costs nothing (or less than nothing) today and can never lose.
There are two kinds to check.

Butterfly arbitrage (one expiry)
--------------------------------
A butterfly (long the K-h and K+h calls, short two K calls) pays something non-negative in
every state, so it must cost something non-negative. In the limit that's
``d^2 C / dK^2 >= 0``, and Breeden-Litzenberger says that second derivative, times
``e^{rT}``, *is* the market's risk-neutral density of the stock at expiry. So "no butterfly
arbitrage" means "the implied density is never negative".

Gatheral (2011) writes the density in terms of total variance ``w(k)``:

    density(k) = g(k) / sqrt(2 pi w) * exp(-d2^2 / 2),   d2 = -k / sqrt(w) - sqrt(w) / 2
    g(k) = (1 - k w' / (2w))^2 - (w')^2 / 4 * (1/w + 1/4) + w'' / 2

Everything except ``g`` is positive, so the test is simply ``g(k) >= 0`` for all ``k``.
SVI gives ``w'`` and ``w''`` in closed form, so this is exact, not a finite difference. We
evaluate ``g`` on a fine grid of ``k`` that reaches well beyond the quoted strikes, because a
fit can be fine where there are quotes and still imply negative probabilities in the wings.

We also check Roger Lee's bound on the wings, ``b (1 + |rho|) <= 2``. Lee showed the wing
slope is set by how many moments of the stock price are finite, and slope 2 is the extreme
where barely the mean exists, so no arbitrage-free model can have steeper wings (and ``g``
eventually goes negative far out if it does).

Calendar arbitrage (across expiries)
------------------------------------
At a fixed log-moneyness ``k = ln(K/F)``, total variance must not decrease with expiry:
``w(k, T2) >= w(k, T1)`` for ``T2 > T1``. The reason: with forward-moneyness held fixed, a
call's (forward-normalized) price is increasing in total variance, and a longer-dated option
must be worth at least a shorter one (the holder could replicate the short one's payoff). If
two fitted slices *cross*, a calendar spread at those strikes is an arbitrage. Comparing at
the same ``k`` rather than the same dollar strike is what makes dividends and rates drop out.

What we return
--------------
Reports rather than True/False, because a desk wants to know *where* and *how badly* a surface
fails: the worst value, where it occurs, and the ``k`` range affected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from optdesk.surface.svi import LEE_SLOPE, SVIParams, svi_derivatives

__all__ = [
    "DEFAULT_K",
    "ButterflyReport",
    "CalendarReport",
    "butterfly_check",
    "calendar_check",
    "check_surface",
    "g_function",
]

DEFAULT_K = np.linspace(-1.5, 1.5, 3001)  # far beyond any quoted strike; step 0.001
TOL = 1e-10  # floating-point slack: tiny negative values are rounding, not arbitrage


def g_function(k: ArrayLike, params: SVIParams) -> NDArray:
    """Gatheral's density factor ``g(k)``; the slice is butterfly-free where it is >= 0.

    NaN where total variance is <= 0 (the density isn't defined there).
    """
    w, dw, d2w = svi_derivatives(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - np.asarray(k) * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
    return np.where(w > 0, g, np.nan)


def _bad_ranges(k: NDArray, bad: NDArray) -> list[tuple[float, float]]:
    """Contiguous ``k`` intervals where ``bad`` is True, as ``(start, end)`` pairs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
    return [(float(k[s]), float(k[e - 1])) for s, e in zip(edges[::2], edges[1::2], strict=True)]


@dataclass(frozen=True)
class ButterflyReport:
    ok: bool  # no negative density and wings within Lee's bound
    min_g: float  # most negative g (NaN counts as failure: negative variance)
    k_at_min: float
    bad_ranges: list[tuple[float, float]]  # k intervals where g < 0 or w <= 0
    lee_ok: bool  # b (1 + |rho|) <= 2


def butterfly_check(params: SVIParams, k: ArrayLike = DEFAULT_K) -> ButterflyReport:
    """Check one slice for butterfly arbitrage on the grid ``k`` (see module notes)."""
    k = np.asarray(k, dtype=float)
    g = g_function(k, params)
    bad = ~(g >= -TOL)  # NaN (negative variance) is bad too
    filled = np.where(np.isnan(g), -np.inf, g)
    i = int(np.argmin(filled))
    lee_ok = params.b * (1 + abs(params.rho)) <= LEE_SLOPE + TOL
    return ButterflyReport(
        ok=bool(not bad.any() and lee_ok),
        min_g=float(filled[i]),
        k_at_min=float(k[i]),
        bad_ranges=_bad_ranges(k, bad),
        lee_ok=bool(lee_ok),
    )


@dataclass(frozen=True)
class CalendarReport:
    ok: bool  # no crossing between any pair of consecutive expiries
    pairs: pd.DataFrame  # one row per consecutive pair: T1, T2, ok, worst gap, where, ranges


def calendar_check(
    slices: list[tuple[float, SVIParams]], k: ArrayLike = DEFAULT_K
) -> CalendarReport:
    """Check that total variance never decreases from one expiry to the next at fixed ``k``.

    ``slices`` is a list of ``(T, params)``; it is sorted by ``T`` here. Each consecutive
    pair gets a row with ``min_gap`` = the smallest ``w(k, T2) - w(k, T1)`` (negative means
    the slices cross), the ``k`` where it happens and the ranges of ``k`` affected.
    """
    k = np.asarray(k, dtype=float)
    ordered = sorted(slices, key=lambda item: item[0])
    rows = []
    for (T1, p1), (T2, p2) in zip(ordered[:-1], ordered[1:], strict=True):
        gap = p2.total_variance(k) - p1.total_variance(k)
        bad = gap < -TOL
        i = int(np.argmin(gap))
        rows.append(
            {
                "T1": T1,
                "T2": T2,
                "ok": bool(not bad.any()),
                "min_gap": float(gap[i]),
                "k_at_min": float(k[i]),
                "bad_ranges": _bad_ranges(k, bad),
            }
        )
    pairs = pd.DataFrame(rows, columns=["T1", "T2", "ok", "min_gap", "k_at_min", "bad_ranges"])
    return CalendarReport(ok=bool(pairs["ok"].all()), pairs=pairs)


def check_surface(fits: pd.DataFrame, k: ArrayLike = DEFAULT_K) -> pd.DataFrame:
    """Both checks for a table of fitted slices (the output of ``svi.fit_chain``).

    Returns ``fits`` with added columns: ``butterfly_ok``, ``min_g``, ``lee_ok``, and
    ``calendar_ok`` (this slice versus the previous expiry; True for the first).
    """
    fits = fits.sort_values("T").reset_index(drop=True)
    params = [SVIParams.from_mapping(row) for _, row in fits.iterrows()]
    reports = [butterfly_check(p, k) for p in params]
    calendar = calendar_check(list(zip(fits["T"], params, strict=True)), k)
    out = fits.copy()
    out["butterfly_ok"] = [rep.ok for rep in reports]
    out["min_g"] = [rep.min_g for rep in reports]
    out["lee_ok"] = [rep.lee_ok for rep in reports]
    out["calendar_ok"] = [True, *calendar.pairs["ok"]]
    return out
