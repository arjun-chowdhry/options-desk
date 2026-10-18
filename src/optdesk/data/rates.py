"""Risk-free rates: turn Treasury quotes into continuously compounded zero rates by maturity.

Why we need a rate at all
-------------------------
Pricing and implied vol discount the option payoff at ``r``, and the chain cleaner uses ``r`` to
turn the call-minus-put price at each strike into a forward (see ``clean.py``). The rate is
known to within a few basis points, while vol is uncertain by whole vol points, so a rate error
of that size moves an option price far less than vol uncertainty does. A simple Treasury curve
is enough; we don't need a full OIS/SOFR bootstrap.

Why Treasuries, and why these three
-----------------------------------
US Treasuries are the standard proxy for the US dollar risk-free rate, and Yahoo Finance
publishes three CBOE yield indices we can fetch with the same library as the option chain:
``^IRX`` (13-week bill), ``^FVX`` (5-year note) and ``^TNX`` (10-year note). Listed equity
options mostly expire within two years, which the 3-month and 5-year points bracket well.

Why the quotes need converting
------------------------------
The indices are quoted in percent, and each uses a different market convention, while this
package uses continuously compounded rates everywhere (``DF = e^{-rT}``):

- A T-bill is quoted as a *bank discount yield* ``d``: price per 1 of face is
  ``1 - d * days / 360``. We solve ``e^{-r * days/365} = 1 - d * days/360`` for ``r``.
- Notes are quoted as a *bond-equivalent yield* ``y``, compounded twice a year:
  ``(1 + y/2)^{2T} = e^{rT}`` gives ``r = 2 ln(1 + y/2)``. (This treats the par yield of a
  coupon bond as a zero rate. The difference is a few basis points, which is negligible here.)

Why linear interpolation with flat ends
---------------------------------------
Between the quoted maturities we interpolate the rate linearly in ``T``: it is simple, never
overshoots, and the curve is smooth enough at this scale. Before the first point and after the
last we hold the rate flat, because extending a slope beyond the data invents information.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "TREASURY_TENORS",
    "RateCurve",
    "bond_yield_to_continuous",
    "discount_yield_to_continuous",
    "treasury_curve",
]

TBILL_DAYS = 91  # a 13-week bill
# Yahoo symbol -> (maturity in years, quote convention)
TREASURY_TENORS: dict[str, tuple[float, str]] = {
    "^IRX": (TBILL_DAYS / 365, "discount"),
    "^FVX": (5.0, "bond"),
    "^TNX": (10.0, "bond"),
}


def discount_yield_to_continuous(d: ArrayLike, days: ArrayLike = TBILL_DAYS) -> NDArray:
    """Continuous rate from a T-bill bank discount yield ``d`` (as a decimal, 0.05 = 5%).

    The bill's price is ``1 - d * days / 360`` (money-market ACT/360 convention), and the
    continuous rate is the one that discounts 1 to that price over ``days / 365`` years.
    """
    d, days = np.asarray(d, dtype=float), np.asarray(days, dtype=float)
    price = 1.0 - d * days / 360.0
    return -np.log(price) / (days / 365.0)


def bond_yield_to_continuous(y: ArrayLike, freq: int = 2) -> NDArray:
    """Continuous rate from a yield compounded ``freq`` times a year (2 = US Treasury notes)."""
    y = np.asarray(y, dtype=float)
    return freq * np.log1p(y / freq)


@dataclass(frozen=True)
class RateCurve:
    """Continuously compounded zero rates at a few maturities, interpolated in between.

    Calling the curve with ``T`` returns the rate(s) for those maturities; ``discount_factor``
    returns ``e^{-r(T) T}``. Both broadcast over NumPy arrays.
    """

    tenors: tuple[float, ...]  # years, strictly increasing
    rates: tuple[float, ...]  # continuous annual rates, one per tenor

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if tenors.ndim != 1 or tenors.size == 0 or tenors.shape != rates.shape:
            raise ValueError("tenors and rates must be non-empty 1-D sequences of equal length")
        if np.any(tenors <= 0) or np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be positive and strictly increasing")
        if not np.all(np.isfinite(rates)):
            raise ValueError("rates must be finite")
        # Store plain floats so the curve is hashable and round-trips through JSON exactly.
        object.__setattr__(self, "tenors", tuple(float(t) for t in tenors))
        object.__setattr__(self, "rates", tuple(float(r) for r in rates))

    @classmethod
    def flat(cls, rate: float) -> RateCurve:
        """A curve with the same rate at every maturity (handy for tests and quick pricing)."""
        return cls(tenors=(1.0,), rates=(rate,))

    def __call__(self, T: ArrayLike) -> NDArray:
        # np.interp is linear between knots and flat beyond the ends, exactly what we want.
        return np.interp(np.asarray(T, dtype=float), self.tenors, self.rates)[()]

    def discount_factor(self, T: ArrayLike) -> NDArray:
        """Price today of 1 dollar paid at ``T``: ``e^{-r(T) T}``."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self(T) * T)[()]

    def to_dict(self) -> dict[str, list[float]]:
        """Plain-Python form, used to store the curve inside a Parquet snapshot."""
        return {"tenors": list(self.tenors), "rates": list(self.rates)}

    @classmethod
    def from_dict(cls, d: Mapping[str, list[float]]) -> RateCurve:
        return cls(tenors=tuple(d["tenors"]), rates=tuple(d["rates"]))


def treasury_curve(quotes_pct: Mapping[str, float]) -> RateCurve:
    """Build a ``RateCurve`` from Yahoo Treasury index quotes in percent, e.g. ``{"^IRX": 4.2}``.

    Symbols not in ``TREASURY_TENORS`` are rejected; missing or NaN quotes are skipped (a failed
    download shouldn't sink the whole curve) as long as at least one usable quote remains.
    """
    unknown = set(quotes_pct) - set(TREASURY_TENORS)
    if unknown:
        raise ValueError(f"unknown Treasury symbols: {sorted(unknown)}")

    points = []
    for symbol, (tenor, convention) in TREASURY_TENORS.items():
        quote = quotes_pct.get(symbol)
        if quote is None or not np.isfinite(quote):
            continue
        decimal = quote / 100.0
        if convention == "discount":
            rate = discount_yield_to_continuous(decimal)
        else:
            rate = bond_yield_to_continuous(decimal)
        points.append((tenor, float(rate)))

    if not points:
        raise ValueError("no usable Treasury quotes")
    points.sort()
    tenors, rates = zip(*points, strict=True)
    return RateCurve(tenors=tenors, rates=rates)
