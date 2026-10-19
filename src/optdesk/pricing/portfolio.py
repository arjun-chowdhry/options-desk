"""Portfolio Greeks: risk of a book of options (and a share hedge), in desk units.

What a risk report shows
------------------------
A trader doesn't look at one option's delta; they look at the *book's* exposures, summed
over every position and expressed in money: "I'm long $40k of vega per vol point, short
$12k of gamma per 1% move, and paying $3k a day in theta." This module turns a list of
positions into that table, one row per position plus the totals, and a scenario grid of
the book's P&L under joint spot and vol shocks.

Where each option's vol comes from
----------------------------------
Each position is valued at the vol the surface gives its strike and expiry
(``k = ln(K/F)``), unless the position carries its own vol (e.g. the vol it was traded at).
So the same book shows different risk on a steep-skew surface than on a flat one, as it
should: an OTM put is priced, and hedged, at its own (higher) vol.

Units (desk convention; the raw per-1.00 Greeks live in ``greeks.py``)
---------------------------------------------------------------------
- delta: shares of the underlying (positive = long). ``dollar_delta`` = shares x spot.
- gamma: shares of delta per $1 move; ``gamma_1pct`` = $ of delta gained per 1% move
  (``gamma x S^2 / 100``), the number that's comparable across underlyings.
- vega: $ per vol point (0.20 -> 0.21). vanna: shares of delta per vol point.
  volga: $ of vega per vol point, per vol point.
- theta: $ per calendar day (ACT/365). rho: $ per 1% (100 bp) rate move.
Everything is per *contract* times ``multiplier`` (100 shares for listed US options).

Why full revaluation for scenarios
----------------------------------
The Greeks are a Taylor expansion: they are accurate for small moves and get worse for
large ones (a short straddle's loss on a 10% move is far more than gamma predicts, because
gamma itself changes). The scenario grid reprices every option at each shocked spot and vol
instead, so large moves are exact within the model. Vols are shocked in **parallel** and
held **sticky-strike** (each option keeps its own vol plus the shock as spot moves). That's
the usual convention for desk risk reports, and it is the world the Black-Scholes Greeks
above describe, so a small shock on the grid matches delta, gamma and vega.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from optdesk.pricing.black_scholes import bs_price, forward_price
from optdesk.pricing.greeks import (
    DAYS_PER_YEAR,
    delta,
    gamma,
    rho,
    theta,
    vanna,
    vega,
    volga,
)

if TYPE_CHECKING:  # pricing doesn't depend on surface at runtime: any implied_vol(k, T) works
    from optdesk.surface.source import VolSurface

__all__ = [
    "GREEK_COLUMNS",
    "PRESETS",
    "Position",
    "as_frame",
    "book_risk",
    "delta_hedge_shares",
    "position_vols",
    "positions_from_frame",
    "preset_book",
    "scenario_pnl",
    "taylor_pnl",
    "totals",
]

MULTIPLIER = 100.0
GREEK_COLUMNS = [
    "value",
    "delta",
    "dollar_delta",
    "gamma",
    "gamma_1pct",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
]


@dataclass(frozen=True)
class Position:
    """``quantity`` contracts (negative = short) of one option. ``T`` in years."""

    K: float
    T: float
    call: bool
    quantity: float
    vol: float | None = None  # None = use the surface's vol at this strike and expiry

    def __post_init__(self) -> None:
        if self.K <= 0 or self.T <= 0:
            raise ValueError("strike and expiry must be positive")
        if self.vol is not None and self.vol <= 0:
            raise ValueError("vol override must be positive")

    @property
    def label(self) -> str:
        side = "C" if self.call else "P"
        return f"{self.quantity:+g} x {self.K:g}{side} {self.T * DAYS_PER_YEAR:.0f}d"


def _arrays(positions: Sequence[Position]):
    K = np.array([p.K for p in positions], dtype=float)
    T = np.array([p.T for p in positions], dtype=float)
    call = np.array([p.call for p in positions], dtype=bool)
    qty = np.array([p.quantity for p in positions], dtype=float)
    return K, T, call, qty


def position_vols(
    positions: Sequence[Position], S: float, r: float, q: float, surface: VolSurface
) -> np.ndarray:
    """Each position's vol: its override, else the surface at ``k = ln(K/F)`` and ``T``."""
    if not positions:
        return np.array([])
    K, T, _, _ = _arrays(positions)
    F = forward_price(S, T, r, q)
    vol = np.asarray(surface.implied_vol(np.log(K / F), T), dtype=float).reshape(-1)
    override = np.array([np.nan if p.vol is None else p.vol for p in positions])
    return np.where(np.isnan(override), vol, override)


def book_risk(
    positions: Sequence[Position],
    S: float,
    r: float,
    q: float,
    surface: VolSurface,
    *,
    shares: float = 0.0,
    multiplier: float = MULTIPLIER,
) -> pd.DataFrame:
    """One row per position (plus a ``Stock`` row if ``shares``) with value and Greeks.

    Columns: ``position``, ``quantity``, ``vol``, ``price`` (per share, Black-Scholes at that
    vol), then ``GREEK_COLUMNS`` in the desk units of the module notes. Sum with ``totals``.
    """
    rows = []
    if positions:
        K, T, call, qty = _arrays(positions)
        vol = position_vols(positions, S, r, q, surface)
        args = (S, K, T, r, vol, q, call)
        size = qty * multiplier
        d = delta(*args)
        g = gamma(*args)
        price = bs_price(*args)
        table = pd.DataFrame(
            {
                "position": [p.label for p in positions],
                "quantity": qty,
                "vol": vol,
                "price": price,
                "value": size * price,
                "delta": size * d,
                "dollar_delta": size * d * S,
                "gamma": size * g,
                "gamma_1pct": size * g * S**2 / 100,
                "vega": size * vega(*args) / 100,
                "theta": size * theta(*args) / DAYS_PER_YEAR,
                "rho": size * rho(*args) / 100,
                "vanna": size * vanna(*args) / 100,
                "volga": size * volga(*args) / 100**2,
            }
        )
        rows.append(table)
    if shares:
        stock = {col: 0.0 for col in GREEK_COLUMNS}
        stock.update(value=shares * S, delta=shares, dollar_delta=shares * S)
        rows.append(
            pd.DataFrame(
                [
                    {
                        "position": f"{shares:+g} shares",
                        "quantity": shares,
                        "vol": np.nan,
                        "price": S,
                        **stock,
                    }
                ]
            )
        )
    if not rows:
        return pd.DataFrame(columns=["position", "quantity", "vol", "price", *GREEK_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def totals(risk: pd.DataFrame) -> pd.Series:
    """Book totals of every Greek column of a ``book_risk`` table."""
    return risk[GREEK_COLUMNS].astype(float).sum()


def scenario_pnl(
    positions: Sequence[Position],
    S: float,
    r: float,
    q: float,
    surface: VolSurface,
    spot_shocks: ArrayLike,
    vol_shocks: ArrayLike,
    *,
    shares: float = 0.0,
    multiplier: float = MULTIPLIER,
) -> pd.DataFrame:
    """Book P&L under every (vol shock, spot shock) pair, by full revaluation.

    ``spot_shocks`` are relative (0.05 = spot up 5%); ``vol_shocks`` are absolute and
    parallel (0.02 = every vol up 2 points), sticky-strike (see module notes). Returns a
    DataFrame indexed by vol shock with one column per spot shock.
    """
    spot_shocks = np.asarray(spot_shocks, dtype=float)
    vol_shocks = np.asarray(vol_shocks, dtype=float)
    pnl = shares * S * spot_shocks[None, :] * np.ones((vol_shocks.size, 1))
    if positions:
        K, T, call, qty = _arrays(positions)
        vol0 = position_vols(positions, S, r, q, surface)
        base = bs_price(S, K, T, r, vol0, q, call)
        S1 = S * (1 + spot_shocks)[None, :, None]  # (1, n_spot, 1)
        vol1 = np.maximum(vol0[None, None, :] + vol_shocks[:, None, None], 1e-4)
        shocked = bs_price(S1, K, T, r, vol1, q, call)  # (n_vol, n_spot, n_positions)
        pnl = pnl + multiplier * ((shocked - base) * qty).sum(axis=-1)
    return pd.DataFrame(pnl, index=pd.Index(vol_shocks, name="vol_shock"), columns=spot_shocks)


def taylor_pnl(
    total: pd.Series, S: float, spot_shocks: ArrayLike, vol_shock: float = 0.0
) -> pd.Series:
    """What the Greeks alone predict for each spot shock (and one parallel vol shock).

    ``delta dS + 1/2 gamma dS^2 + vega dv + vanna dS dv + 1/2 volga dv^2`` with ``dS`` in
    dollars and ``dv`` in vol points, using the book ``totals``. Compare with ``scenario_pnl``
    to see where the second-order picture breaks down.
    """
    spot_shocks = np.asarray(spot_shocks, dtype=float)
    dS = S * spot_shocks
    dv = 100 * vol_shock
    pnl = (
        total["delta"] * dS
        + 0.5 * total["gamma"] * dS**2
        + total["vega"] * dv
        + total["vanna"] * dS * dv
        + 0.5 * total["volga"] * dv**2
    )
    return pd.Series(pnl, index=spot_shocks, name="taylor")


def delta_hedge_shares(risk: pd.DataFrame) -> float:
    """Whole shares that flatten the *options'* delta in a ``book_risk`` table."""
    options = risk[~risk["position"].str.endswith("shares")]
    return -float(np.round(options["delta"].sum()))


def positions_from_frame(frame: pd.DataFrame) -> list[Position]:
    """Positions from an editable table: ``type`` (call/put), ``strike``, ``days``,
    ``quantity`` and an optional ``vol`` in percent (blank = surface vol).

    Rows with a zero or blank quantity are skipped; a blank strike or expiry is an error.
    """
    out = []
    for _, row in frame.iterrows():
        qty = row.get("quantity")
        if qty is None or pd.isna(qty) or qty == 0:
            continue
        kind = str(row["type"]).strip().lower()
        if kind not in ("call", "put"):
            raise ValueError(f"type must be 'call' or 'put', got {row['type']!r}")
        if pd.isna(row["strike"]) or pd.isna(row["days"]):
            raise ValueError("every position needs a strike and days to expiry")
        vol = row.get("vol")
        out.append(
            Position(
                K=float(row["strike"]),
                T=float(row["days"]) / DAYS_PER_YEAR,
                call=kind == "call",
                quantity=float(qty),
                vol=None if vol is None or pd.isna(vol) else float(vol) / 100,
            )
        )
    return out


# --- Example books ------------------------------------------------------------------------------

# name -> legs of (call?, strike as a fraction of the forward, days, contracts)
PRESETS: dict[str, list[tuple[bool, float, int, float]]] = {
    "Long 1m ATM straddle": [(True, 1.00, 30, 10), (False, 1.00, 30, 10)],
    "3m risk reversal (long 105% call, short 95% put)": [
        (True, 1.05, 91, 10),
        (False, 0.95, 91, -10),
    ],
    "ATM calendar (short 1m, long 3m)": [(True, 1.00, 30, -10), (True, 1.00, 91, 10)],
    "2m call butterfly (95/100/105)": [
        (True, 0.95, 61, 10),
        (True, 1.00, 61, -20),
        (True, 1.05, 61, 10),
    ],
}


def preset_book(
    name: str, S: float, r: float, q: float, strike_step: float = 1.0
) -> list[Position]:
    """One of ``PRESETS``, with strikes at ``fraction x forward`` rounded to ``strike_step``."""
    out = []
    for call, frac, days, qty in PRESETS[name]:
        T = days / DAYS_PER_YEAR
        K = float(np.round(frac * float(forward_price(S, T, r, q)) / strike_step) * strike_step)
        out.append(Position(K=K, T=T, call=call, quantity=qty))
    return out


def as_frame(positions: Iterable[Position]) -> pd.DataFrame:
    """The editable-table form of ``positions`` (the inverse of ``positions_from_frame``)."""
    return pd.DataFrame(
        [
            {
                "type": "call" if p.call else "put",
                "strike": p.K,
                "days": round(p.T * DAYS_PER_YEAR),
                "quantity": p.quantity,
                "vol": None if p.vol is None else 100 * p.vol,
            }
            for p in positions
        ],
        columns=["type", "strike", "days", "quantity", "vol"],
    )
