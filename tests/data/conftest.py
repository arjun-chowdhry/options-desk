"""Shared fixtures for the data tests: a fake yfinance backed by the saved SPY chain.

``FakeTicker`` imitates the small slice of ``yf.Ticker`` that ``fetch.py`` uses (``options``,
``option_chain`` and ``history``) and serves ``tests/fixtures/spy_chain.csv``. The
``fake_yf`` fixture swaps it in, so the real fetch code runs end to end without a network.
As a guard, every test in this folder also has real network connections disabled.
"""

from __future__ import annotations

import json
import socket
from collections import namedtuple
from pathlib import Path

import pandas as pd
import pytest

from optdesk.data import fetch
from optdesk.data.rates import treasury_curve

FIXTURES = Path(__file__).parents[1] / "fixtures"
CHAIN_CSV = FIXTURES / "spy_chain.csv"
TRUTH = json.loads((FIXTURES / "spy_chain_truth.json").read_text())

Options = namedtuple("Options", ["calls", "puts", "underlying"])


class FakeTicker:
    """Stands in for ``yf.Ticker``; records which expiries were requested."""

    chain = pd.read_csv(CHAIN_CSV, dtype={"expiry": str})
    requests: list[str] = []
    failing: set[str] = set()  # symbols whose history() should raise, to test error handling

    def __init__(self, symbol: str):
        self.symbol = symbol

    @property
    def options(self) -> tuple[str, ...]:
        if self.symbol.upper() != TRUTH["ticker"]:
            return ()
        return tuple(sorted(self.chain["expiry"].unique()))

    def option_chain(self, date: str) -> Options:
        FakeTicker.requests.append(date)
        rows = self.chain[self.chain["expiry"] == date].drop(columns="expiry")
        rows = rows.assign(lastTradeDate=pd.to_datetime(rows["lastTradeDate"], utc=True))
        calls, puts = (
            rows[rows["option_type"] == t].drop(columns="option_type").reset_index(drop=True)
            for t in ("call", "put")
        )
        return Options(calls, puts, {"regularMarketPrice": TRUTH["spot"]})

    def history(self, period: str) -> pd.DataFrame:
        if self.symbol in self.failing:
            raise ConnectionError("simulated download failure")
        close = TRUTH["treasury_quotes_pct"][self.symbol]
        return pd.DataFrame({"Close": [close - 0.05, close]})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test in this folder tries to open a real connection."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def fake_yf(monkeypatch):
    """Route ``fetch.py``'s yfinance calls to ``FakeTicker``."""
    FakeTicker.requests = []
    FakeTicker.failing = set()
    monkeypatch.setattr(fetch.yf, "Ticker", FakeTicker)
    return FakeTicker


@pytest.fixture
def source_chain() -> pd.DataFrame:
    """The saved chain exactly as stored on disk (yfinance column names)."""
    return pd.read_csv(CHAIN_CSV)


@pytest.fixture
def truth() -> dict:
    return TRUTH


@pytest.fixture
def raw_chain(fake_yf) -> pd.DataFrame:
    """The saved chain as ``fetch_chain`` returns it."""
    return fetch.fetch_chain(TRUTH["ticker"], asof=pd.Timestamp(TRUTH["asof"]))


@pytest.fixture
def curve():
    return treasury_curve(TRUTH["treasury_quotes_pct"])
