"""Tests for Parquet snapshots, and the fetch-and-save command line (with the fake yfinance).

The key property is an exact round trip: a saved and reloaded snapshot is identical to the
original (values, types, time zones and the rate curve), so cleaning it later reproduces the
original result bit for bit. All files go to pytest's temporary folder.
"""

import datetime as dt

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from optdesk.data import fetch
from optdesk.data.clean import clean_chain
from optdesk.data.store import list_snapshots, load_snapshot, save_snapshot, snapshot_path


def test_round_trip_is_exact(raw_chain, curve, tmp_path):
    path = save_snapshot(raw_chain, curve, tmp_path)
    chain, loaded_curve = load_snapshot("SPY", "2026-10-16", tmp_path)
    pd.testing.assert_frame_equal(chain, raw_chain)
    assert loaded_curve == curve
    assert path == tmp_path / "SPY" / "2026-10-16.parquet"


def test_cleaning_a_loaded_snapshot_reproduces_the_result(raw_chain, curve, tmp_path):
    save_snapshot(raw_chain, curve, tmp_path)
    pd.testing.assert_frame_equal(
        clean_chain(*load_snapshot("SPY", "2026-10-16", tmp_path)), clean_chain(raw_chain, curve)
    )


def test_date_is_the_new_york_trading_day(raw_chain, curve, tmp_path):
    """10:30pm New York on 16 Oct is already 17 Oct in UTC; it still belongs to the 16th."""
    late = raw_chain.assign(asof=pd.Timestamp("2026-10-17 02:30", tz="UTC"))
    assert save_snapshot(late, curve, tmp_path).stem == "2026-10-16"


def test_existing_snapshot_is_protected(raw_chain, curve, tmp_path):
    save_snapshot(raw_chain, curve, tmp_path)
    with pytest.raises(FileExistsError):
        save_snapshot(raw_chain, curve, tmp_path)
    changed = raw_chain.assign(bid=raw_chain["bid"] + 0.01)
    save_snapshot(changed, curve, tmp_path, overwrite=True)
    chain, _ = load_snapshot("SPY", "2026-10-16", tmp_path)
    pd.testing.assert_series_equal(chain["bid"], changed["bid"])
    assert not list(tmp_path.rglob("*.tmp"))  # no leftover temporary file


def test_list_snapshots(raw_chain, curve, tmp_path):
    assert list_snapshots("SPY", tmp_path) == []
    for day in ("2026-10-16", "2026-10-14", "2026-10-15"):
        asof = pd.Timestamp(f"{day} 19:45", tz="UTC")
        save_snapshot(raw_chain.assign(asof=asof), curve, tmp_path)
    (tmp_path / "SPY" / "notes.parquet").touch()  # not a date: ignored
    assert list_snapshots("spy", tmp_path) == [
        dt.date(2026, 10, 14),
        dt.date(2026, 10, 15),
        dt.date(2026, 10, 16),
    ]


def test_snapshot_path_layout_and_validation(tmp_path):
    assert snapshot_path("brk-b", dt.date(2026, 1, 2), tmp_path) == (
        tmp_path / "BRK-B" / "2026-01-02.parquet"
    )
    for bad in ("../etc", "SPY/QQQ", ""):
        with pytest.raises(ValueError, match="invalid ticker"):
            snapshot_path(bad, "2026-01-02", tmp_path)


def test_save_rejects_bad_chains(raw_chain, curve, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_snapshot(raw_chain.iloc[0:0], curve, tmp_path)
    mixed = raw_chain.assign(ticker=["SPY", "QQQ"] * (len(raw_chain) // 2))
    with pytest.raises(ValueError, match="one ticker"):
        save_snapshot(mixed, curve, tmp_path)


def test_load_errors(raw_chain, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot("SPY", "2026-10-16", tmp_path)
    path = snapshot_path("SPY", "2026-10-16", tmp_path)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(raw_chain), path)  # a Parquet file, but not ours
    with pytest.raises(ValueError, match="not an optdesk snapshot"):
        load_snapshot("SPY", "2026-10-16", tmp_path)


# --- Command line ------------------------------------------------------------------------------


def test_command_line_fetch_saves_and_summarizes(fake_yf, tmp_path, capsys, truth, monkeypatch):
    # Pin the download time to the fixture's, so the test doesn't age as real time passes.
    real_fetch = fetch.fetch_chain
    asof = pd.Timestamp(truth["asof"])
    monkeypatch.setattr(fetch, "fetch_chain", lambda t, n=None: real_fetch(t, n, asof=asof))
    main = fetch.main

    # The fixture is priced as European options, so tell the cleaner (the default is American).
    assert main(["SPY", "--max-expiries", "2", "--root", str(tmp_path), "--european"]) == 0
    out = capsys.readouterr().out
    saved = list((tmp_path / "SPY").glob("*.parquet"))
    assert len(saved) == 1
    chain, curve = load_snapshot("SPY", saved[0].stem, tmp_path)
    assert chain["expiry"].nunique() == 2
    assert "2026-10-23" in out and "2026-11-20" in out and "ATM vol=" in out
    forward = truth["expiries"]["2026-10-23"]["forward"]
    assert f"F={forward:.2f}" in out

    with pytest.raises(FileExistsError):
        main(["SPY", "--root", str(tmp_path)])
    assert main(["SPY", "--root", str(tmp_path), "--overwrite"]) == 0
