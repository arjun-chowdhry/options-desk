"""Download option chains and Treasury yields from Yahoo Finance (via yfinance).

This is the only module in the package that touches the network. Everything downstream
(cleaning, the surface, the app) works on the DataFrame it returns, so tests can swap
``yf.Ticker`` for a fake that serves a saved chain and run offline.

Why yfinance
------------
It is free, needs no API key, and returns full chains (every strike and expiry, with bid,
ask, last trade, volume and open interest) for US-listed options. The trade-offs: quotes can be
delayed about 15 minutes, and outside market hours many bids and asks come back as zero.
That's why the cleaner drops zero bids and why snapshots should be taken while the market is
open.

Why one long table
------------------
yfinance gives a separate calls table and puts table per expiry. We stack them all into one
"long" table with ``expiry`` and ``option_type`` columns, and rename the columns to
snake_case. One flat table is easy to filter, group by expiry, and save as a single Parquet
file. We keep the raw quotes untouched (no filtering here) so that cleaning choices can be
changed and re-run on old snapshots.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from optdesk.data.rates import TREASURY_TENORS, RateCurve, treasury_curve

__all__ = ["RAW_COLUMNS", "fetch_chain", "fetch_rate_curve", "fetch_treasury_quotes"]

# yfinance column -> our column
_RENAME = {
    "contractSymbol": "contract",
    "strike": "strike",
    "bid": "bid",
    "ask": "ask",
    "lastPrice": "last",
    "lastTradeDate": "last_trade",
    "volume": "volume",
    "openInterest": "open_interest",
}

RAW_COLUMNS = [
    "ticker",
    "asof",  # when the chain was downloaded (UTC)
    "spot",  # underlying price reported with this expiry's chain
    "expiry",  # expiration date (the option stops trading at the close that day)
    "option_type",  # "call" or "put"
    *_RENAME.values(),
]


def fetch_chain(
    ticker: str,
    max_expiries: int | None = None,
    *,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Download the full option chain for ``ticker`` as one long table of raw quotes.

    Parameters
    ----------
    ticker : underlying symbol, e.g. ``"SPY"``.
    max_expiries : only fetch the nearest this-many expiries (``None`` = all). Each expiry is
        one web request, so this keeps quick experiments fast.
    asof : timestamp to record as the download time; defaults to now (UTC).

    Returns
    -------
    DataFrame with columns ``RAW_COLUMNS``, one row per contract.
    """
    asof = pd.Timestamp.now(tz="UTC") if asof is None else pd.Timestamp(asof)
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)[:max_expiries]
    if not expiries:
        raise ValueError(f"no listed options found for {ticker!r}")

    frames = []
    for expiry in expiries:
        chain = tk.option_chain(expiry)
        spot = chain.underlying.get("regularMarketPrice", np.nan)
        for option_type, quotes in (("call", chain.calls), ("put", chain.puts)):
            if quotes.empty:
                continue
            frame = quotes[list(_RENAME)].rename(columns=_RENAME)
            frame.insert(0, "option_type", option_type)
            frame.insert(0, "expiry", pd.Timestamp(expiry))
            frame.insert(0, "spot", float(spot))
            frames.append(frame)
    if not frames:
        raise ValueError(f"every expiry of {ticker!r} came back empty")

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "asof", asof)
    out.insert(0, "ticker", ticker.upper())
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out["open_interest"] = out["open_interest"].fillna(0).astype("int64")
    return out[RAW_COLUMNS]


def fetch_treasury_quotes() -> dict[str, float]:
    """Latest close of each Treasury yield index in ``TREASURY_TENORS``, in percent.

    A symbol that fails to download comes back as NaN with a warning, and ``treasury_curve``
    skips it, so one bad request doesn't stop the whole fetch.
    """
    quotes = {}
    for symbol in TREASURY_TENORS:
        try:
            closes = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
            quotes[symbol] = float(closes.iloc[-1]) if len(closes) else np.nan
        except Exception as exc:  # yfinance raises many error types for network/parse failures
            warnings.warn(f"could not fetch {symbol}: {exc}", stacklevel=2)
            quotes[symbol] = np.nan
    return quotes


def fetch_rate_curve() -> RateCurve:
    """Download Treasury yields and convert them to a continuous zero-rate curve."""
    return treasury_curve(fetch_treasury_quotes())
