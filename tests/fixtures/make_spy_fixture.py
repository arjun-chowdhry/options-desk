"""Generate the synthetic SPY option chain used by the offline data tests.

Run from the repo root to regenerate (output is deterministic):

    python tests/fixtures/make_spy_fixture.py

Why synthetic instead of a real download
----------------------------------------
A real chain has no answer key: we can't check that the cleaner found the *right* forward or
implied vol, only that it produced something. Here every quote comes from Black-Scholes with
inputs we choose (spot, dividend yield, a Treasury curve and a volatility smile), so the true
forward and true vol of every contract are known exactly. Bid and ask sit symmetrically
around the model price, so the mid equals it to floating-point precision.

The file mimics yfinance's format (same column names, one row per contract) with two extra
columns, ``expiry`` and ``option_type``, so the fake ``yf.Ticker`` in the tests can split it
back into per-expiry calls and puts tables. yfinance's ``impliedVolatility`` column holds the
true vol. It also includes the dirt the cleaner must remove: zero bids in the far wings
(options worth under a cent), one crossed quote, one very wide quote and one stale quote. These
are listed in the truth file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from optdesk.data.rates import treasury_curve
from optdesk.pricing.black_scholes import bs_price, forward_price

OUT = Path(__file__).parent
TICKER = "SPY"
ASOF = pd.Timestamp("2026-10-16 19:45", tz="UTC")  # Friday 3:45pm New York time
SPOT = 580.0
DIV_YIELD = 0.013
TREASURY_QUOTES_PCT = {"^IRX": 4.00, "^FVX": 3.90, "^TNX": 4.10}
EXPIRIES = ["2026-10-23", "2026-11-20", "2026-12-18", "2027-03-19"]
STRIKES = np.arange(480.0, 680.0 + 1, 5.0)
DEFECTS = {  # (kind, expiry, option_type, strike): all out of the money, so they'd otherwise pass
    "crossed": ("2026-11-20", "call", 600.0),
    "wide": ("2026-12-18", "put", 550.0),
    "stale": ("2027-03-19", "call", 620.0),
}


def smile(k: np.ndarray, T: float) -> np.ndarray:
    """Equity-style skew: vol rises for low strikes. Scaling k by sqrt(T) flattens long expiries."""
    x = k / np.sqrt(T)
    return np.maximum(0.16 - 0.10 * x + 0.06 * x**2, 0.06)


def contract_symbol(expiry: str, option_type: str, strike: float) -> str:
    """OCC symbol, e.g. SPY261023C00580000 (strike x 1000, zero-padded to 8 digits)."""
    yymmdd = pd.Timestamp(expiry).strftime("%y%m%d")
    return f"{TICKER}{yymmdd}{option_type[0].upper()}{round(strike * 1000):08d}"


def year_fraction(expiry: str) -> float:
    """ACT/365 from ASOF to 4pm New York time on the expiry date."""
    close = pd.Timestamp(f"{expiry} 16:00").tz_localize("America/New_York")
    return (close - ASOF).total_seconds() / (365 * 24 * 3600)


def main() -> None:
    rng = np.random.default_rng(20261016)
    curve = treasury_curve(TREASURY_QUOTES_PCT)
    rows, truth_expiries = [], {}

    for expiry in EXPIRIES:
        T = year_fraction(expiry)
        r = float(curve(T))
        F = float(forward_price(SPOT, T, r, DIV_YIELD))
        truth_expiries[expiry] = {"T": T, "r": r, "forward": F}
        sigma = smile(np.log(STRIKES / F), T)

        for option_type in ("call", "put"):
            is_call = option_type == "call"
            price = bs_price(SPOT, STRIKES, T, r, sigma, DIV_YIELD, is_call)
            half_spread = np.maximum(0.01, 0.04 * price)
            bid = price - half_spread
            ask = price + half_spread
            bid[bid < 0.005] = 0.0  # nobody bids for an option worth under half a cent
            n = len(STRIKES)
            rows.append(
                pd.DataFrame(
                    {
                        "expiry": expiry,
                        "option_type": option_type,
                        "contractSymbol": [
                            contract_symbol(expiry, option_type, K) for K in STRIKES
                        ],
                        "lastTradeDate": ASOF - pd.to_timedelta(rng.uniform(0, 6, n), unit="h"),
                        "strike": STRIKES,
                        "lastPrice": np.round(price + rng.normal(0, 0.5, n) * half_spread, 2),
                        "bid": bid,
                        "ask": ask,
                        "change": 0.0,
                        "percentChange": 0.0,
                        "volume": rng.integers(0, 5000, n),
                        "openInterest": rng.integers(0, 50000, n),
                        "impliedVolatility": sigma,
                        "inTheMoney": STRIKES < SPOT if is_call else STRIKES > SPOT,
                        "contractSize": "REGULAR",
                        "currency": "USD",
                    }
                )
            )

    chain = pd.concat(rows, ignore_index=True)

    defect_symbols = {}
    for kind, (expiry, option_type, strike) in DEFECTS.items():
        symbol = contract_symbol(expiry, option_type, strike)
        defect_symbols[kind] = symbol
        i = chain.index[chain["contractSymbol"] == symbol][0]
        bid, ask = chain.at[i, "bid"], chain.at[i, "ask"]
        if kind == "crossed":
            chain.loc[i, ["bid", "ask"]] = [ask, bid]
        elif kind == "wide":
            mid = (bid + ask) / 2
            chain.loc[i, ["bid", "ask"]] = [0.5 * mid, 2.0 * mid]
        elif kind == "stale":
            chain.at[i, "lastTradeDate"] = ASOF - pd.Timedelta(days=30)

    chain["lastTradeDate"] = chain["lastTradeDate"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    chain.to_csv(OUT / "spy_chain.csv", index=False, float_format="%.17g")

    truth = {
        "ticker": TICKER,
        "asof": ASOF.isoformat(),
        "spot": SPOT,
        "dividend_yield": DIV_YIELD,
        "treasury_quotes_pct": TREASURY_QUOTES_PCT,
        "expiries": truth_expiries,
        "defects": defect_symbols,
    }
    (OUT / "spy_chain_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"wrote {len(chain)} contracts to {OUT}")


if __name__ == "__main__":
    main()
