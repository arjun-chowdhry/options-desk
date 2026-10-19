"""Save today's option-chain snapshot(s): the script Windows Task Scheduler runs every weekday.

Usage (from anywhere; the script moves to the repo folder itself)::

    .venv\\Scripts\\python.exe scripts\\snapshot.py              # SPY, every expiry
    .venv\\Scripts\\python.exe scripts\\snapshot.py SPY QQQ --max-expiries 12

Each ticker goes to ``data/snapshots/{TICKER}/{YYYY-MM-DD}.parquet`` (via ``optdesk.data``),
and every run appends to ``data/logs/snapshot.log``, so you can check later what happened on a
day you weren't watching.

Why these choices
-----------------
- *Every expiry by default.* SPY lists options for every weekday, so the nearest 6 expiries
  only reach one week out. A surface (and the market-making study's 2- to 6-month options)
  needs the whole term structure. A full SPY chain is ~30 web requests and ~1 MB on disk.
- *Skip days with no trading.* On a market holiday Yahoo still serves the last session's
  quotes, which would be saved under today's date. If no contract in the chain traded today
  (New York date), we log it and save nothing.
- *One ticker failing doesn't stop the others,* and the exit code is 1 if any failed, so Task
  Scheduler's "Last Run Result" column shows that something went wrong.
- *Run while the market is open.* After the close many bids come back as zero and the
  cleaner throws most quotes away (see ``optdesk.data.fetch``). The scheduled task ("optdesk
  SPY snapshot") runs weekdays at 2:15pm Central = 3:15pm New York, 45 minutes before the
  close: late enough to capture the day's surface, early enough that quotes are still live.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOG_FILE = Path("data/logs/snapshot.log")
NEW_YORK = "America/New_York"

log = logging.getLogger("snapshot")


def traded_today(chain: pd.DataFrame) -> bool:
    """True if any contract in the chain last traded on the download's New York date."""
    today = chain["asof"].iloc[0].tz_convert(NEW_YORK).date()
    last = chain["last_trade"].dropna()
    return bool(len(last)) and last.max().tz_convert(NEW_YORK).date() == today


def snapshot(ticker: str, max_expiries: int | None, root: Path, *, american: bool = True) -> Path:
    """Fetch, check and save one ticker's chain; return the snapshot path."""
    from optdesk.data.clean import clean_chain, quote_flags
    from optdesk.data.fetch import fetch_chain, fetch_rate_curve
    from optdesk.data.store import save_snapshot

    chain = fetch_chain(ticker, max_expiries)
    if not traded_today(chain):
        raise RuntimeError("no contract traded today (market holiday?); nothing saved")
    curve = fetch_rate_curve()
    path = save_snapshot(chain, curve, root)
    flags = quote_flags(chain)
    clean = clean_chain(chain, curve, american=american)
    log.info(
        "%s: saved %d quotes over %d expiries to %s; flagged %s; %d clean OTM quotes",
        ticker,
        len(chain),
        chain["expiry"].nunique(),
        path,
        ", ".join(f"{k} {v}" for k, v in flags.sum().items()),
        len(clean),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save today's option-chain snapshots.")
    parser.add_argument("tickers", nargs="*", default=["SPY"], help="symbols (default: SPY)")
    parser.add_argument("--max-expiries", type=int, default=None, help="nearest N only")
    parser.add_argument("--root", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--european", action="store_true", help="European options (^SPX)")
    args = parser.parse_args(argv)

    os.chdir(REPO)  # Task Scheduler starts in C:\Windows\System32; paths here are repo-relative
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    failed = 0
    for ticker in args.tickers:
        try:
            snapshot(ticker, args.max_expiries, args.root, american=not args.european)
        except FileExistsError:
            log.info("%s: today's snapshot already exists; left as is", ticker)
        except Exception as exc:  # log and carry on with the next ticker
            log.error("%s: %s", ticker, exc)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
