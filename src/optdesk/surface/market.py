"""Market data sources: one bundle (spot, rates, surface, fits) from the synthetic market or a
saved snapshot, so the dashboard and the studies can switch data with a single choice.

Why a bundle
------------
Every dashboard tab needs the same few things: today's spot, a rate and dividend yield, a
``VolSurface`` to price at, and (for the surface tab) the SVI fit table and the cleaned
quotes behind it. ``MarketData`` holds all of them. Code downstream only ever sees a
``MarketData``, never where it came from, so going from synthetic to live data is a choice in
a dropdown, not a code change.

Two sources
-----------
- **Synthetic** (``synthetic_market``): the SPY-like chain of ``surface/synthetic.py``, cleaned
  and fitted exactly like live data. Its pricing surface is the analytic SSVI *truth*, not
  the fit, so the simulators run on the known answer (the same choice ``sim/mm_plot.py``
  makes). The fit is still shown, which is a check that the pipeline recovers that truth.
- **Snapshot** (``snapshot_market``): a saved chain from ``data/snapshots/``, cleaned
  (de-Americanized for SPY), fitted per expiry with SVI and interpolated in total variance
  (``SVISurface``). The fit is the only surface we have, so it is also the pricing surface.

Rates from the chain, not the screen
------------------------------------
For a snapshot, ``r`` is the Treasury curve at the expiry nearest 3 months and ``q`` is the
dividend yield *implied by the parity forward* there: ``q = r - ln(F/S) / T``. Using the
forward the options themselves imply keeps theo consistent with the market's own carry,
which a quoted dividend yield (backward-looking and lumpy for SPY) would not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optdesk.data.clean import clean_chain
from optdesk.data.rates import RateCurve
from optdesk.data.store import DEFAULT_ROOT, list_snapshots, load_snapshot
from optdesk.pricing.black_scholes import forward_price
from optdesk.surface.arbitrage import check_surface
from optdesk.surface.source import VolSurface, fitted_surface, synthetic_surface
from optdesk.surface.svi import fit_chain

__all__ = [
    "SYNTHETIC",
    "MarketData",
    "MarketSource",
    "available_sources",
    "carry_from_fits",
    "load_market",
    "snapshot_market",
    "synthetic_market",
]

CARRY_TENOR = 0.25  # take r and q at the fitted expiry nearest 3 months


@dataclass(frozen=True)
class MarketData:
    """Everything the dashboard needs about one market, wherever it came from."""

    label: str  # shown to the user, e.g. "Synthetic SPY" or "SPY 2026-10-19"
    spot: float
    r: float  # continuous rate used by the simulators and the portfolio
    q: float  # continuous dividend yield
    surface: VolSurface  # what everything prices at
    fits: pd.DataFrame  # SVI fit per expiry with arbitrage checks (check_surface output)
    clean: pd.DataFrame  # the cleaned quotes behind the fits
    curve: RateCurve
    synthetic: bool
    american: bool = True  # whether the quotes were de-Americanized before fitting

    @property
    def key(self) -> str:
        """Identifies the market for caching: the same label can be fitted two ways."""
        return f"{self.label}|{'american' if self.american else 'european'}"

    def atm_vol(self, T: float) -> float:
        """At-the-forward implied vol at expiry ``T`` on the pricing surface."""
        return float(self.surface.implied_vol(0.0, T))

    def vol_at(self, K: float, T: float) -> float:
        """Surface vol for strike ``K`` and expiry ``T``: the smile at ``k = ln(K/F)``."""
        F = float(forward_price(self.spot, T, self.r, self.q))
        return float(self.surface.implied_vol(np.log(K / F), T))


@dataclass(frozen=True)
class MarketSource:
    """A choice in the data-source dropdown: the synthetic market or one saved snapshot."""

    kind: str  # "synthetic" or "snapshot"
    ticker: str | None = None
    date: str | None = None  # YYYY-MM-DD

    @property
    def label(self) -> str:
        return "Synthetic SPY (SSVI)" if self.kind == "synthetic" else f"{self.ticker} {self.date}"


SYNTHETIC = MarketSource("synthetic")


def carry_from_fits(fits: pd.DataFrame, spot: float, curve: RateCurve) -> tuple[float, float]:
    """``(r, q)`` at the fitted expiry nearest ``CARRY_TENOR``: curve rate and parity-implied q."""
    row = fits.iloc[(fits["T"] - CARRY_TENOR).abs().argmin()]
    T = float(row["T"])
    r = float(curve(T))
    q = r - float(np.log(row["forward"] / spot)) / T
    return r, q


def _fit(raw: pd.DataFrame, curve: RateCurve, american: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = clean_chain(raw, curve, american=american)
    return clean, check_surface(fit_chain(clean))


def synthetic_market(seed: int = 0) -> MarketData:
    """The synthetic SPY market: true SSVI surface for pricing, fitted SVI for display."""
    from optdesk.surface.synthetic import DIV_YIELD, SPOT, synthetic_chain

    market = synthetic_chain(seed=seed)
    clean, fits = _fit(market.raw, market.curve, american=True)
    r = float(market.curve(CARRY_TENOR))
    return MarketData(
        label="Synthetic SPY",
        spot=SPOT,
        r=r,
        q=DIV_YIELD,  # known exactly here, no need to back it out of the forward
        surface=synthetic_surface(),
        fits=fits,
        clean=clean,
        curve=market.curve,
        synthetic=True,
    )


def snapshot_market(
    ticker: str, date: str, root: Path | str = DEFAULT_ROOT, *, american: bool = True
) -> MarketData:
    """A saved snapshot, cleaned and fitted. ``american=False`` for European options (^SPX)."""
    raw, curve = load_snapshot(ticker, date, root)
    clean, fits = _fit(raw, curve, american=american)
    if fits.empty:
        raise ValueError(f"no expiry of {ticker} {date} had enough clean quotes to fit")
    spot = float(raw["spot"].iloc[0])
    r, q = carry_from_fits(fits, spot, curve)
    return MarketData(
        label=f"{ticker.upper()} {pd.Timestamp(date):%Y-%m-%d}",
        spot=spot,
        r=r,
        q=q,
        surface=fitted_surface(fits),
        fits=fits,
        clean=clean,
        curve=curve,
        synthetic=False,
        american=american,
    )


def available_sources(root: Path | str = DEFAULT_ROOT) -> list[MarketSource]:
    """The synthetic market first, then every saved snapshot under ``root`` (newest first)."""
    snapshots = []
    root = Path(root)
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            for day in list_snapshots(folder.name, root):
                snapshots.append(MarketSource("snapshot", folder.name.upper(), f"{day:%Y-%m-%d}"))
    snapshots.sort(key=lambda s: (s.date, s.ticker), reverse=True)
    return [SYNTHETIC, *snapshots]


def load_market(
    source: MarketSource, root: Path | str = DEFAULT_ROOT, *, american: bool = True
) -> MarketData:
    """Load whichever source was chosen."""
    if source.kind == "synthetic":
        return synthetic_market()
    if source.kind == "snapshot":
        return snapshot_market(source.ticker, source.date, root, american=american)
    raise ValueError(f"unknown source kind {source.kind!r}")
