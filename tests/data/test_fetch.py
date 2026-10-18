"""Tests for the yfinance download layer, run against a fake yfinance (no network).

Checks that the per-expiry calls/puts tables become one long table with our column names and
unchanged values, that ``max_expiries`` limits the requests, that bad tickers fail clearly, and
that Treasury quotes turn into the expected rate curve even when one download fails.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.data.fetch import RAW_COLUMNS, fetch_chain, fetch_rate_curve, fetch_treasury_quotes
from optdesk.data.rates import treasury_curve


def test_chain_shape_and_columns(raw_chain, source_chain, truth):
    expected = source_chain
    assert list(raw_chain.columns) == RAW_COLUMNS
    assert len(raw_chain) == len(expected)
    assert set(raw_chain["option_type"]) == {"call", "put"}
    assert (raw_chain["ticker"] == "SPY").all()
    assert (raw_chain["spot"] == truth["spot"]).all()
    assert (raw_chain["asof"] == pd.Timestamp(truth["asof"])).all()
    assert sorted(raw_chain["expiry"].dt.strftime("%Y-%m-%d").unique()) == sorted(truth["expiries"])


def test_quotes_pass_through_unchanged(raw_chain, source_chain):
    """Fetching must not alter quotes: every value matches the source file by contract."""
    expected = source_chain.set_index("contractSymbol")
    got = raw_chain.set_index("contract").loc[expected.index]
    np.testing.assert_array_equal(got["strike"], expected["strike"])
    np.testing.assert_array_equal(got["bid"], expected["bid"])
    np.testing.assert_array_equal(got["ask"], expected["ask"])
    np.testing.assert_array_equal(got["last"], expected["lastPrice"])
    np.testing.assert_array_equal(got["volume"], expected["volume"])
    np.testing.assert_array_equal(got["open_interest"], expected["openInterest"])
    assert (got["option_type"].str[0] == expected["option_type"].str[0]).all()
    assert (got["last_trade"] == pd.to_datetime(expected["lastTradeDate"], utc=True)).all()


def test_max_expiries_limits_requests(fake_yf):
    chain = fetch_chain("SPY", max_expiries=2)
    assert fake_yf.requests == ["2026-10-23", "2026-11-20"]
    assert chain["expiry"].nunique() == 2


def test_lowercase_ticker_is_normalized(fake_yf):
    assert (fetch_chain("spy", max_expiries=1)["ticker"] == "SPY").all()


def test_asof_defaults_to_now(fake_yf):
    before = pd.Timestamp.now(tz="UTC")
    chain = fetch_chain("SPY", max_expiries=1)
    assert before <= chain["asof"].iloc[0] <= pd.Timestamp.now(tz="UTC")


def test_unknown_ticker_raises(fake_yf):
    with pytest.raises(ValueError, match="no listed options"):
        fetch_chain("NOTATICKER")


def test_empty_tables_are_skipped(fake_yf, monkeypatch):
    """An expiry whose calls table is empty still contributes its puts; all-empty is an error."""
    original = fake_yf.option_chain

    def no_calls(self, date):
        chain = original(self, date)
        return chain._replace(calls=chain.calls.iloc[0:0])

    monkeypatch.setattr(fake_yf, "option_chain", no_calls)
    assert set(fetch_chain("SPY")["option_type"]) == {"put"}

    def nothing(self, date):
        chain = original(self, date)
        return chain._replace(calls=chain.calls.iloc[0:0], puts=chain.puts.iloc[0:0])

    monkeypatch.setattr(fake_yf, "option_chain", nothing)
    with pytest.raises(ValueError, match="came back empty"):
        fetch_chain("SPY")


def test_treasury_quotes_and_curve(fake_yf, truth):
    assert fetch_treasury_quotes() == truth["treasury_quotes_pct"]  # the latest close
    assert fetch_rate_curve() == treasury_curve(truth["treasury_quotes_pct"])


def test_failed_treasury_download_is_skipped(fake_yf):
    fake_yf.failing = {"^FVX"}
    with pytest.warns(UserWarning, match=r"\^FVX"):
        quotes = fetch_treasury_quotes()
    assert np.isnan(quotes["^FVX"])
    with pytest.warns(UserWarning):
        assert len(fetch_rate_curve().tenors) == 2
