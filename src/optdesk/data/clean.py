"""Turn a raw option chain into clean, out-of-the-money quotes with forwards and implied vols.

The steps, in order
-------------------
1. **Drop bad quotes** (``quote_flags``): zero bids, crossed markets, spreads that are too
   wide, and stale contracts. Each is a quote whose mid doesn't tell us the option's value.
2. **Mid price and time to expiry.** We value each option at the mid of bid and ask, and
   measure ``T`` in years (ACT/365) to 4pm New York time on the expiry date.
3. **Forward per expiry from put-call parity** (``parity_forward``).
4. **Keep only out-of-the-money options**: puts struck below the forward, calls at or above it.
5. **Implied vol and log-moneyness** ``k = ln(K / F)``, the coordinates the M3 surface uses.

Why these filters
-----------------
- *Zero bid:* nobody will pay anything for the option, so its "mid" is half the ask, an
  artifact of the quote rather than a price. These are the far wings, or every quote after
  hours.
- *Crossed (ask < bid):* impossible in a live market. It means the bid and ask come from
  different moments, or the data is bad.
- *Wide (spread / mid > ``max_rel_spread``):* when the spread is as big as the price itself,
  the mid could be anywhere in a huge range and the implied vol is noise. We measure the
  spread relative to the mid because a 5-cent spread is tight on a $10 option and enormous on
  a 6-cent one.
- *Stale (no trade in ``max_stale_days``):* a contract nobody has traded in days is usually
  illiquid, and its quotes are more likely to be left over and out of date. Contracts with no
  trade date at all count as stale.

Why infer the forward instead of computing ``S e^{(r-q)T}``
-----------------------------------------------------------
Computing it needs the dividend yield ``q``, which for an ETF or index is an estimate, and
the stock's borrow cost, which we can't observe at all. Put-call parity gives the forward
straight from the market: a call minus a put at the same strike pays ``S_T - K`` at expiry, the
same as a forward contract, so ``C - P = DF * (F - K)`` and ``F = K + (C - P) / DF``. Every
strike gives an estimate. We take those from the ``n_strikes`` strikes where ``|C - P|`` is
smallest (closest to the money, where both options are liquid and their spreads tightest) and
use the median, so one bad quote can't drag the answer. Only the discount factor
``DF = e^{-rT}`` comes from outside, from the Treasury curve.

Caveat: SPY and single-stock options are American, and parity is exact only for European
options. Near the money the early-exercise premium is tiny, so the error is small. That is
another reason to use only near-the-money strikes.

Why keep only out-of-the-money options
--------------------------------------
For a European option, a call and a put at the same strike carry the same implied vol (that's
parity). But the in-the-money one is mostly intrinsic value with wider spreads, so its vol is
noisier. Using the OTM option at every strike gives one clean vol per strike: puts below the
forward, calls above. This is the standard convention for building a smile (the VIX
calculation does the same).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from optdesk.data.rates import RateCurve
from optdesk.pricing.implied_vol import implied_vol

__all__ = ["CLEAN_COLUMNS", "clean_chain", "parity_forward", "quote_flags", "year_fraction"]

EXPIRY_HOUR = 16  # equity options stop trading at the 4pm close in New York
EXPIRY_TZ = "America/New_York"
SECONDS_PER_YEAR = 365 * 24 * 3600  # ACT/365

CLEAN_COLUMNS = [
    "ticker",
    "asof",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "rel_spread",  # (ask - bid) / mid
    "T",  # years to expiry, ACT/365
    "r",  # continuous risk-free rate for T
    "forward",  # implied from put-call parity, one per expiry
    "k",  # log-moneyness ln(K / forward)
    "iv",  # Black-Scholes implied vol of the mid
    "volume",
    "open_interest",
    "contract",
]


def year_fraction(asof: pd.Timestamp | pd.Series, expiry: pd.Timestamp | pd.Series) -> ArrayLike:
    """Years (ACT/365) from ``asof`` to 4pm New York time on the ``expiry`` date.

    Using the actual close time (instead of counting whole days) matters for short expiries:
    on expiry morning a weekly option has hours left, not zero or one full day. The time zone
    handles daylight saving, so 4pm New York is 20:00 UTC in summer and 21:00 UTC in winter.
    """
    expiry = pd.to_datetime(expiry)
    close_hour = pd.Timedelta(hours=EXPIRY_HOUR)
    if isinstance(expiry, pd.Series):
        close = (expiry.dt.normalize() + close_hour).dt.tz_localize(EXPIRY_TZ)
    else:
        close = (expiry.normalize() + close_hour).tz_localize(EXPIRY_TZ)
    seconds = (close - pd.to_datetime(asof)) / pd.Timedelta(seconds=1)
    return seconds / SECONDS_PER_YEAR


def quote_flags(
    raw: pd.DataFrame,
    *,
    max_rel_spread: float = 0.5,
    max_stale_days: float | None = 5.0,
) -> pd.DataFrame:
    """One boolean column per reason a quote is unusable; a row is bad if any is True.

    Columns: ``zero_bid`` (bid missing or <= 0), ``crossed`` (ask < bid), ``wide``
    ((ask - bid) / mid > ``max_rel_spread``), ``stale`` (last trade more than
    ``max_stale_days`` calendar days before ``asof``, or never; ``None`` turns this off).
    """
    bid, ask = raw["bid"], raw["ask"]
    mid = (bid + ask) / 2
    zero_bid = ~(bid > 0)  # also catches NaN
    crossed = ask < bid
    wide = ~zero_bid & ~crossed & ((ask - bid) / mid > max_rel_spread)
    if max_stale_days is None:
        stale = pd.Series(False, index=raw.index)
    else:
        age = raw["asof"] - raw["last_trade"]
        stale = ~(age <= pd.Timedelta(days=max_stale_days))  # NaT (never traded) -> stale
    return pd.DataFrame({"zero_bid": zero_bid, "crossed": crossed, "wide": wide, "stale": stale})


def parity_forward(
    strikes: ArrayLike,
    call_mids: ArrayLike,
    put_mids: ArrayLike,
    discount: float,
    n_strikes: int = 5,
) -> float:
    """Forward price implied by put-call parity across the strikes of one expiry.

    Each strike gives ``F = K + (C - P) / DF``; we return the median over the ``n_strikes``
    strikes with the smallest ``|C - P|`` (the ones nearest the forward). Inputs are matched
    arrays: one call mid and one put mid per strike. Returns NaN if there are no strikes.
    """
    strikes, call_mids, put_mids = (
        np.asarray(x, dtype=float) for x in (strikes, call_mids, put_mids)
    )
    if strikes.size == 0:
        return np.nan
    diff = call_mids - put_mids
    nearest = np.argsort(np.abs(diff), kind="stable")[:n_strikes]
    return float(np.median(strikes[nearest] + diff[nearest] / discount))


def clean_chain(
    raw: pd.DataFrame,
    curve: RateCurve,
    *,
    max_rel_spread: float = 0.5,
    max_stale_days: float | None = 5.0,
    n_parity_strikes: int = 5,
) -> pd.DataFrame:
    """Clean a raw chain from ``fetch_chain`` into OTM quotes with forward, ``k`` and implied vol.

    Parameters
    ----------
    raw : the long table from ``fetch.fetch_chain`` (or a loaded snapshot).
    curve : risk-free rate curve for discounting (e.g. ``fetch.fetch_rate_curve()``).
    max_rel_spread, max_stale_days : quote filters, see ``quote_flags``.
    n_parity_strikes : how many near-the-money strikes to use for each forward.

    Returns
    -------
    DataFrame with ``CLEAN_COLUMNS``, sorted by expiry then strike: one OTM option per strike.
    Also dropped: expiries already past; expiries with no strike quoted on both the call and
    the put side after filtering (no forward is possible, so this warns); and quotes with no
    implied vol (price outside the no-arbitrage bounds).
    """
    flags = quote_flags(raw, max_rel_spread=max_rel_spread, max_stale_days=max_stale_days)
    quotes = raw.loc[~flags.any(axis=1)].copy()
    if quotes.empty:
        warnings.warn(
            "every quote failed the filters (zero bids everywhere usually means the chain was "
            "fetched while the market was closed)",
            stacklevel=2,
        )
    # Adjusted contracts (after splits or special dividends) can repeat a strike; keep the first.
    quotes = quotes.drop_duplicates(subset=["expiry", "option_type", "strike"])
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    quotes["rel_spread"] = (quotes["ask"] - quotes["bid"]) / quotes["mid"]
    quotes["T"] = year_fraction(quotes["asof"], quotes["expiry"])
    quotes = quotes[quotes["T"] > 0]
    quotes["r"] = curve(quotes["T"].to_numpy())

    forwards = {}
    for expiry, group in quotes.groupby("expiry"):
        # One row per strike, call and put mids side by side; keep strikes that have both.
        mids = group.pivot(index="strike", columns="option_type", values="mid")
        pairs = mids.reindex(columns=["call", "put"]).dropna()
        if pairs.empty:
            warnings.warn(f"no call/put pair for {expiry:%Y-%m-%d}; dropping expiry", stacklevel=2)
            continue
        T, r = group["T"].iloc[0], group["r"].iloc[0]
        forwards[expiry] = parity_forward(
            pairs.index, pairs["call"], pairs["put"], np.exp(-r * T), n_parity_strikes
        )

    quotes["forward"] = quotes["expiry"].map(forwards)
    quotes = quotes[quotes["forward"].notna()]
    is_call = (quotes["option_type"] == "call").to_numpy()
    otm = np.where(
        is_call, quotes["strike"] >= quotes["forward"], quotes["strike"] < quotes["forward"]
    )
    quotes = quotes[otm]

    quotes["k"] = np.log(quotes["strike"] / quotes["forward"])
    # Price on the forward: S = F with q = r makes Black-Scholes' forward S e^{(r-q)T} equal F.
    quotes["iv"] = implied_vol(
        quotes["mid"].to_numpy(),
        S=quotes["forward"].to_numpy(),
        K=quotes["strike"].to_numpy(),
        T=quotes["T"].to_numpy(),
        r=quotes["r"].to_numpy(),
        q=quotes["r"].to_numpy(),
        call=(quotes["option_type"] == "call").to_numpy(),
    )
    quotes = quotes[quotes["iv"].notna()]
    return quotes.sort_values(["expiry", "strike"]).reset_index(drop=True)[CLEAN_COLUMNS]
