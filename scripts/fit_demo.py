"""Regenerate the saved fit of the demo snapshot (``data/demo/SPY/2026-10-22.fit.parquet``).

Usage (from anywhere; paths are resolved from the repo folder)::

    .venv\\Scripts\\python.exe scripts\\fit_demo.py

Run it whenever ``tests/surface/test_saved_fit.py`` says the saved fit is stale: after
changing the demo snapshot, a cleaning or fitting default, or any module listed in
``optdesk.surface.saved_fit.FIT_MODULES``. It fits the demo exactly as the app would
(American exercise on, every default setting), then writes the cleaned quotes, the SVI
fits and their fingerprint next to the snapshot. Takes about half a minute.

Why a script and not "fit on first load"
----------------------------------------
The hosted app must never write to its own repo, and a fit made on the server would be lost
whenever the free tier puts the app to sleep. Committing the fit means every cold start gets
it, and its fingerprint means the app ignores it the moment it stops matching.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO / "data" / "demo"


def main() -> int:
    from optdesk.data.store import load_snapshot, snapshot_path
    from optdesk.surface.market import demo_source, fit_quotes
    from optdesk.surface.saved_fit import (
        fit_fingerprint,
        load_fit,
        save_fit,
        saved_fit_path,
    )

    demo = demo_source(DEMO_ROOT)
    if demo is None:
        print(f"no demo snapshot under {DEMO_ROOT}", file=sys.stderr)
        return 1
    snapshot = snapshot_path(demo.ticker, demo.date, DEMO_ROOT)
    start = time.perf_counter()
    raw, curve = load_snapshot(demo.ticker, demo.date, DEMO_ROOT)
    clean, fits = fit_quotes(raw, curve, True)  # American: SPY options can be exercised early
    fingerprint = fit_fingerprint(snapshot, american=True)
    path = save_fit(saved_fit_path(snapshot), clean, fits, fingerprint)
    assert load_fit(path, fingerprint) is not None  # reads back and matches
    print(
        f"saved {path.relative_to(REPO)}: {len(fits)} expiries, {len(clean):,} quotes, "
        f"{path.stat().st_size / 1e3:.0f} KB, fitted in {time.perf_counter() - start:.0f} s"
    )
    print(
        f"fingerprint: snapshot {fingerprint['snapshot_sha256'][:12]}, "
        f"code {fingerprint['code_sha256'][:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
