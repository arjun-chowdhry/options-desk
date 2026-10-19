"""Tests for ``scripts/snapshot.py``, the daily job Task Scheduler runs (offline, fake yfinance).

The fixture chain was downloaded at 3:45pm New York on Friday 16 Oct 2026 and its options
last traded that day. We pin the download time to that moment (a normal trading day) or to
the next Monday morning before anything has traded (what a holiday looks like).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from optdesk.data import fetch
from optdesk.data.store import list_snapshots

SCRIPT = Path(__file__).parents[2] / "scripts" / "snapshot.py"
spec = importlib.util.spec_from_file_location("snapshot_script", SCRIPT)
snapshot_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot_script)

FRIDAY = pd.Timestamp("2026-10-16 19:45", tz="UTC")  # the fixture's own download time
MONDAY = pd.Timestamp("2026-10-19 14:00", tz="UTC")  # 10am New York, before any of its trades


@pytest.fixture
def run(fake_yf, monkeypatch, tmp_path):
    """Run the script's ``main`` inside ``tmp_path`` with the download time pinned to ``asof``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snapshot_script, "REPO", tmp_path)
    real = fetch.fetch_chain

    def go(args, asof=FRIDAY):
        monkeypatch.setattr(fetch, "fetch_chain", lambda t, n=None: real(t, n, asof=asof))
        return snapshot_script.main([*args, "--root", str(tmp_path / "snaps"), "--european"])

    return go


def test_traded_today():
    chain = pd.DataFrame(
        {
            "asof": [FRIDAY] * 2,
            "last_trade": [pd.Timestamp("2026-10-15 15:00", tz="UTC"), pd.NaT],
        }
    )
    assert not snapshot_script.traded_today(chain)
    # 01:00 UTC on the 17th is still the 16th in New York
    chain.loc[1, "last_trade"] = pd.Timestamp("2026-10-17 01:00", tz="UTC")
    assert snapshot_script.traded_today(chain)


def test_saves_snapshot_and_logs(run, tmp_path):
    assert run(["SPY"]) == 0
    assert list_snapshots("SPY", tmp_path / "snaps") == [pd.Timestamp("2026-10-16").date()]
    log = (tmp_path / "data" / "logs" / "snapshot.log").read_text()
    assert "SPY: saved" in log and "zero_bid" in log and "clean OTM quotes" in log


def test_second_run_leaves_the_snapshot_alone(run, tmp_path):
    assert run(["SPY"]) == 0
    first = (tmp_path / "snaps" / "SPY" / "2026-10-16.parquet").stat().st_mtime_ns
    assert run(["SPY"]) == 0
    assert (tmp_path / "snaps" / "SPY" / "2026-10-16.parquet").stat().st_mtime_ns == first
    assert "already exists" in (tmp_path / "data" / "logs" / "snapshot.log").read_text()


def test_holiday_saves_nothing_and_fails(run, tmp_path):
    assert run(["SPY"], asof=MONDAY) == 1
    assert list_snapshots("SPY", tmp_path / "snaps") == []
    assert "market holiday" in (tmp_path / "data" / "logs" / "snapshot.log").read_text()


def test_one_bad_ticker_does_not_stop_the_rest(run, tmp_path):
    assert run(["NOPE", "SPY"]) == 1
    assert list_snapshots("SPY", tmp_path / "snaps") == [pd.Timestamp("2026-10-16").date()]
    assert "NOPE: no listed options" in (tmp_path / "data" / "logs" / "snapshot.log").read_text()
