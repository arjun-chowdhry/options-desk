"""Save and load daily option-chain snapshots as Parquet files.

Layout: ``data/snapshots/{TICKER}/{YYYY-MM-DD}.parquet``, one file per ticker per trading day
(the date is the New York date of the download). The folder is gitignored, because market data
is regenerable and too bulky for the repo.

Why Parquet
-----------
It is a compressed, columnar format: an options chain (thousands of rows of numbers)
shrinks to a fraction of its CSV size, loads much faster, and keeps exact types (floats stay
bit-for-bit identical, and timestamps keep their time zone). CSV would turn everything into
text and lose both.

Why store the raw chain, plus the rate curve
--------------------------------------------
We save quotes exactly as downloaded, before cleaning, so the cleaning rules can change later
and be re-run on old days. To make that re-run reproduce the original result, the snapshot
also needs the interest-rate curve from the same moment. It goes into the Parquet file's
metadata (a small JSON note stored in the header), so one file holds everything needed to
rebuild the day's surface.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from optdesk.data.rates import RateCurve

__all__ = ["DEFAULT_ROOT", "list_snapshots", "load_snapshot", "save_snapshot", "snapshot_path"]

DEFAULT_ROOT = Path("data/snapshots")
SNAPSHOT_TZ = "America/New_York"  # a snapshot belongs to the New York trading day
_METADATA_KEY = b"optdesk"
_FORMAT_VERSION = 1
_TICKER_RE = re.compile(r"[A-Za-z0-9.^=_-]+")  # e.g. SPY, BRK-B, ^SPX; no path separators


def snapshot_path(ticker: str, date: dt.date | str, root: Path | str = DEFAULT_ROOT) -> Path:
    """Where the snapshot for ``ticker`` on ``date`` lives: ``root/TICKER/YYYY-MM-DD.parquet``."""
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"invalid ticker {ticker!r}")
    day = pd.Timestamp(date).date()
    return Path(root) / ticker.upper() / f"{day:%Y-%m-%d}.parquet"


def save_snapshot(
    chain: pd.DataFrame,
    curve: RateCurve,
    root: Path | str = DEFAULT_ROOT,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a raw chain (from ``fetch_chain``) and its rate curve; return the file path.

    The ticker and date come from the chain's ``ticker`` and ``asof`` columns. An existing
    snapshot for the same day is only replaced if ``overwrite=True``, so a second fetch can't
    silently clobber the first.
    """
    if chain.empty:
        raise ValueError("cannot save an empty chain")
    tickers = chain["ticker"].unique()
    if len(tickers) != 1:
        raise ValueError(f"a snapshot holds one ticker, got {list(tickers)}")
    asof = pd.Timestamp(chain["asof"].iloc[0])
    path = snapshot_path(tickers[0], asof.tz_convert(SNAPSHOT_TZ).date(), root)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")

    table = pa.Table.from_pandas(chain, preserve_index=False)
    note = {"version": _FORMAT_VERSION, "rate_curve": curve.to_dict()}
    metadata = {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(note).encode()}
    table = table.replace_schema_metadata(metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and then rename it, so a crash mid-write never leaves a
    # half-written snapshot under the real name.
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


def load_snapshot(
    ticker: str, date: dt.date | str, root: Path | str = DEFAULT_ROOT
) -> tuple[pd.DataFrame, RateCurve]:
    """Load a saved snapshot: the raw chain and the rate curve stored with it."""
    path = snapshot_path(ticker, date, root)
    if not path.exists():
        raise FileNotFoundError(f"no snapshot at {path}")
    table = pq.read_table(path)
    raw_note = (table.schema.metadata or {}).get(_METADATA_KEY)
    if raw_note is None:
        raise ValueError(f"{path} is not an optdesk snapshot (no rate-curve metadata)")
    note = json.loads(raw_note)
    return table.to_pandas(), RateCurve.from_dict(note["rate_curve"])


def list_snapshots(ticker: str, root: Path | str = DEFAULT_ROOT) -> list[dt.date]:
    """Dates with a saved snapshot for ``ticker``, oldest first."""
    folder = snapshot_path(ticker, "2000-01-01", root).parent
    dates = []
    for f in folder.glob("*.parquet"):
        try:
            dates.append(dt.date.fromisoformat(f.stem))
        except ValueError:
            continue  # ignore files that aren't named by date
    return sorted(dates)
