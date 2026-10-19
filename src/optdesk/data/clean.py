"""Turn a raw option chain into clean, out-of-the-money quotes with forwards and implied vols.

The steps, in order
-------------------
1. **Drop bad quotes** (``quote_flags``): zero bids, crossed markets, spreads that are too
   wide, and stale contracts. Each is a quote whose mid doesn't tell us the option's value.
2. **Mid price and time to expiry.** We value each option at the mid of bid and ask, and
   measure ``T`` in years (ACT/365) to 4pm New York time on the expiry date. Expiries less
   than ``min_days`` away (by default: same-day, "0DTE" options) are dropped.
3. **Forward per expiry from put-call parity** (``parity_forward``).
4. **Keep only out-of-the-money options**: puts struck below the forward, calls at or above it.
5. **Implied vol and log-moneyness** ``k = ln(K / F)``, the coordinates the M3 surface uses.

For American options (``american=True``), steps 3 and 5 first remove the early-exercise
premium; see "American options" below.

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
- *Under a day to expiry (``min_days``):* SPY lists options expiring every weekday, so a
  snapshot taken in the afternoon includes options with minutes left. With ``T`` that small,
  a one-cent tick moves the implied vol by many points (on 19 Oct 2026 at 3:40pm the 0DTE
  at-the-money put implied 25% while next-day options implied 12%). That vol says nothing
  about the surface, and as the first point of every term structure it would trip the
  calendar-arbitrage check. One day is the natural cut: it removes today's expiry whatever
  time the snapshot is taken, and keeps tomorrow's.

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

American options (SPY, single stocks): de-Americanizing
-------------------------------------------------------
Black-Scholes and put-call parity are for European options. SPY options are American, so
each price includes an *early-exercise premium* (EEP). Inverting an American price with
Black-Scholes as if it were European overstates the vol, and the premium on the puts also
drags the parity forward down. Measured on a SPY-like setup (r = 4%, q = 1.3%), the vol error
on a near-the-money OTM put is ~0.04 vol points at 1 month, ~0.14 at 3 months and ~0.6 at 1
year, and the forward comes out ~6bp low at 3 months and ~38bp low at 1 year. Short-dated and
OTM call quotes are barely affected.

With ``american=True`` we remove the premium before anything else uses the prices:

1. Price each quote's premium on a CRR tree (American minus European, same tree), at the
   vol the quote implies, and subtract it: ``European-equivalent = mid - EEP``. The premium
   depends on the vol, which depends on the premium, so we repeat twice (it settles fast
   because the premium is small and barely moves with vol).
2. The tree needs spot and a dividend yield, and the forward needs de-Americanized prices,
   so the forward is found by iterating too: forward from parity -> implied dividend yield
   ``q = r - ln(F/S)/T`` -> de-Americanize the call/put pairs -> forward from parity again.
3. The OTM quotes are de-Americanized with the final forward, then inverted with
   Black-Scholes as before.

Why this method and not a tree-implied vol: inverting the tree directly needs a root search
of full American trees for every quote. Subtracting the premium needs a few trees in total
(batched across every quote at once) and reuses the fast Black-Scholes solver. The result is
a *European-equivalent* vol, which is the right input for SVI and for the arbitrage checks
(both are European theory). Remaining approximation:
SPY pays discrete quarterly dividends while the tree uses a continuous yield. That matters
for calls just before an ex-dividend date, but those are ITM calls, which we don't use.

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
from numpy.typing import ArrayLike, NDArray

from optdesk.data.rates import RateCurve
from optdesk.pricing.binomial import early_exercise_premium
from optdesk.pricing.implied_vol import implied_vol

__all__ = [
    "CLEAN_COLUMNS",
    "clean_chain",
    "european_equivalent",
    "parity_forward",
    "quote_flags",
    "year_fraction",
]

EXPIRY_HOUR = 16  # equity options stop trading at the 4pm close in New York
EXPIRY_TZ = "America/New_York"
SECONDS_PER_YEAR = 365 * 24 * 3600  # ACT/365
EEP_ITERATIONS = 2  # premium <-> vol fixed point (it settles fast; the premium is small)
FORWARD_ITERATIONS = 3  # forward <-> dividend yield fixed point, American chains only
TREE_STEPS = 200  # premium good to ~1 cent on SPY (see pricing.binomial.early_exercise_premium)

CLEAN_COLUMNS = [
    "ticker",
    "asof",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "eep",  # early-exercise premium removed from mid (0 for European options)
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


def european_equivalent(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    call: ArrayLike,
    *,
    steps: int = TREE_STEPS,
) -> tuple[NDArray, NDArray]:
    """Strip the early-exercise premium from American option prices.

    Returns ``(european_price, premium)``. The premium is priced on a CRR tree at the vol
    the quote implies, which itself depends on the premium, so we alternate the two
    ``EEP_ITERATIONS`` times. Quotes with no implied vol keep their price (premium 0); the
    cleaner drops them later anyway.
    """
    price = np.asarray(price, dtype=float)
    premium = np.zeros_like(price)
    for _ in range(EEP_ITERATIONS):
        vol = implied_vol(price - premium, S, K, T, r, q, call)
        ok = np.isfinite(vol) & (vol > 0)
        premium = np.where(
            ok, early_exercise_premium(S, K, T, r, np.where(ok, vol, 0.2), q, call, steps=steps), 0
        )
    return price - premium, premium


def clean_chain(
    raw: pd.DataFrame,
    curve: RateCurve,
    *,
    american: bool = False,
    max_rel_spread: float = 0.5,
    max_stale_days: float | None = 5.0,
    min_days: float = 1.0,
    n_parity_strikes: int = 5,
) -> pd.DataFrame:
    """Clean a raw chain from ``fetch_chain`` into OTM quotes with forward, ``k`` and implied vol.

    Parameters
    ----------
    raw : the long table from ``fetch.fetch_chain`` (or a loaded snapshot).
    curve : risk-free rate curve for discounting (e.g. ``fetch.fetch_rate_curve()``).
    american : the options are American (SPY, single stocks). Strips the early-exercise
        premium before computing forwards and vols (see the module notes). Needs the
        ``spot`` column. Leave False for European options (SPX) and the European fixture.
    max_rel_spread, max_stale_days : quote filters, see ``quote_flags``.
    min_days : drop expiries fewer than this many calendar days (ACT/365) away; the default
        removes same-day options (see "Why these filters"). ``0`` keeps everything unexpired.
    n_parity_strikes : how many near-the-money strikes to use for each forward.

    Returns
    -------
    DataFrame with ``CLEAN_COLUMNS``, sorted by expiry then strike: one OTM option per strike.
    Also dropped: expiries already past or under ``min_days`` away; expiries with no strike
    quoted on both the call and the put side after filtering (no forward is possible, so this
    warns); and quotes with no implied vol (price outside the no-arbitrage bounds).
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
    quotes = quotes[(quotes["T"] > 0) & (quotes["T"] >= min_days / 365)]
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
        K, calls, puts = pairs.index.to_numpy(), pairs["call"], pairs["put"]
        F = parity_forward(K, calls, puts, np.exp(-r * T), n_parity_strikes)
        if american:
            # Forward and premium depend on each other: alternate until they settle.
            S = group["spot"].iloc[0]
            # Parity only uses strikes near the money; don't price trees for the rest.
            near = np.argsort(np.abs(calls - puts).to_numpy(), kind="stable")
            near = near[: 2 * n_parity_strikes]
            K, calls, puts = K[near], calls.iloc[near], puts.iloc[near]
            for _ in range(FORWARD_ITERATIONS):
                q = r - np.log(F / S) / T
                eu_calls, _ = european_equivalent(calls, S, K, T, r, q, True)
                eu_puts, _ = european_equivalent(puts, S, K, T, r, q, False)
                F = parity_forward(K, eu_calls, eu_puts, np.exp(-r * T), n_parity_strikes)
        forwards[expiry] = F

    quotes["forward"] = quotes["expiry"].map(forwards)
    quotes = quotes[quotes["forward"].notna()]
    is_call = (quotes["option_type"] == "call").to_numpy()
    otm = np.where(
        is_call, quotes["strike"] >= quotes["forward"], quotes["strike"] < quotes["forward"]
    )
    quotes = quotes[otm]

    quotes["eep"] = 0.0
    if american:
        # One batch for every expiry: each option's tree has its own T, r and q.
        S, F, T, r = (quotes[c].to_numpy() for c in ("spot", "forward", "T", "r"))
        _, quotes["eep"] = european_equivalent(
            quotes["mid"], S, quotes["strike"], T, r, r - np.log(F / S) / T, is_call[otm]
        )

    quotes["k"] = np.log(quotes["strike"] / quotes["forward"])
    # Price on the forward: S = F with q = r makes Black-Scholes' forward S e^{(r-q)T} equal F.
    quotes["iv"] = implied_vol(
        (quotes["mid"] - quotes["eep"]).to_numpy(),
        S=quotes["forward"].to_numpy(),
        K=quotes["strike"].to_numpy(),
        T=quotes["T"].to_numpy(),
        r=quotes["r"].to_numpy(),
        q=quotes["r"].to_numpy(),
        call=(quotes["option_type"] == "call").to_numpy(),
    )
    quotes = quotes[quotes["iv"].notna()]
    return quotes.sort_values(["expiry", "strike"]).reset_index(drop=True)[CLEAN_COLUMNS]
