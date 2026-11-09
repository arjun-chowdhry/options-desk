"""Tests for the saved demo fit (``surface/saved_fit.py``).

- The committed fit of ``data/demo/`` is current: its fingerprint matches today's snapshot,
  settings and code. If this fails, run ``python scripts/fit_demo.py``.
- The committed fit matches a fresh fit of the same snapshot: the cleaned quotes, the
  arbitrage flags, and the implied vols of both surfaces at every expiry.
- A current saved fit skips the fitter; any change to the quotes, the settings (``american``
  or a default), the code or the file format, or a missing or corrupt file, means a refit.
- Only the demo uses a saved fit: a saved snapshot with a fit file next to it still refits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optdesk.data.store import load_snapshot, save_snapshot
from optdesk.surface import saved_fit
from optdesk.surface.market import MarketSource, available_sources, fit_quotes, load_market
from optdesk.surface.saved_fit import (
    fit_fingerprint,
    load_fit,
    save_fit,
    saved_fit_path,
    with_saved_fit,
)
from optdesk.surface.source import fitted_surface

DEMO_ROOT = Path(__file__).parents[2] / "data" / "demo"
DEMO = DEMO_ROOT / "SPY" / "2026-10-22.parquet"
STALE = "the saved demo fit is stale: run  python scripts/fit_demo.py  and commit the file"
FLAGS = ["expiry", "n", "success", "butterfly_ok", "lee_ok", "calendar_ok"]


def test_committed_demo_fit_is_current():
    assert saved_fit_path(DEMO).name == "2026-10-22.fit.parquet"
    assert saved_fit_path(DEMO).stat().st_size < 500_000  # small enough for git
    assert load_fit(saved_fit_path(DEMO), fit_fingerprint(DEMO, american=True)) is not None, STALE


def test_committed_demo_fit_matches_a_fresh_fit():
    saved = load_fit(saved_fit_path(DEMO), fit_fingerprint(DEMO, american=True))
    assert saved is not None, STALE
    saved_clean, saved_fits = saved
    raw, curve = load_snapshot("SPY", "2026-10-22", DEMO_ROOT)
    clean, fits = fit_quotes(raw, curve, True)

    pd.testing.assert_frame_equal(saved_clean, clean, check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(saved_fits[FLAGS], fits[FLAGS])
    # The saved fit was made on Windows and CI refits on Linux, so the numbers differ by
    # floating-point noise. rmse and max_error are small (about 0.001), which makes a purely
    # relative test too strict for them, so they also get an absolute 0.0001 vol points.
    inputs = ["T", "forward"]
    np.testing.assert_allclose(saved_fits[inputs], fits[inputs], rtol=1e-6)
    errors = ["rmse", "max_error"]
    np.testing.assert_allclose(saved_fits[errors], fits[errors], rtol=1e-6, atol=1e-6)
    # what the app prices with: the vol of each fitted smile, compared on a strike grid
    k = np.linspace(-0.4, 0.2, 61)
    saved_surface, fresh_surface = fitted_surface(saved_fits), fitted_surface(fits)
    for T in fits["T"]:
        diff = saved_surface.implied_vol(k, T) - fresh_surface.implied_vol(k, T)
        assert np.max(np.abs(diff)) < 1e-6, T  # 0.0001 vol points


# --- refit rules, on a small synthetic snapshot (fits in a few seconds) --------------------


@pytest.fixture
def saved_snapshot(tmp_path, exact_market):
    """A European synthetic snapshot with a current saved fit next to it."""
    path = save_snapshot(exact_market.raw, exact_market.curve, tmp_path)
    raw, curve = load_snapshot("SPY", "2026-10-16", tmp_path)
    clean, fits = fit_quotes(raw, curve, False)
    save_fit(saved_fit_path(path), clean, fits, fit_fingerprint(path, american=False))
    return path


class Counting:
    """A stand-in fitter that records its calls and returns a marked result."""

    def __init__(self):
        self.calls = 0

    def __call__(self, raw, curve, american):
        self.calls += 1
        return pd.DataFrame(), pd.DataFrame({"refit": [True]})


def test_a_current_saved_fit_skips_the_fitter(saved_snapshot):
    raw, curve = load_snapshot("SPY", "2026-10-16", saved_snapshot.parent.parent)
    fitter = Counting()
    clean, fits = with_saved_fit(saved_snapshot, fitter)(raw, curve, False)
    assert fitter.calls == 0
    assert len(fits) > 3 and "refit" not in fits and len(clean) > 300
    # the saved fit doesn't show up as a snapshot of its own
    assert len(available_sources(saved_snapshot.parent.parent)) == 2  # synthetic + 1 snapshot


@pytest.mark.parametrize(
    "change", ["quotes", "american", "default", "code", "format", "missing", "corrupt"]
)
def test_a_changed_fingerprint_triggers_a_refit(saved_snapshot, change, monkeypatch):
    root = saved_snapshot.parent.parent
    raw, curve = load_snapshot("SPY", "2026-10-16", root)
    american = False
    if change == "quotes":  # one bid a cent higher: a different snapshot file
        raw.loc[raw.index[0], "bid"] += 0.01
        save_snapshot(raw, curve, root, overwrite=True)
    elif change == "american":
        american = True
    elif change == "default":  # e.g. someone loosens the spread filter in clean_chain
        settings = saved_fit.fit_settings

        def looser(american):
            s = settings(american)
            s["clean_chain"]["max_rel_spread"] = 0.75
            return s

        monkeypatch.setattr(saved_fit, "fit_settings", looser)
    elif change == "code":  # an edit to any module the fit runs through
        monkeypatch.setattr(saved_fit, "_code_hash", lambda: "edited")
    elif change == "format":
        monkeypatch.setattr(saved_fit, "FIT_FORMAT", saved_fit.FIT_FORMAT + 1)
    elif change == "missing":
        saved_fit_path(saved_snapshot).unlink()
    elif change == "corrupt":
        saved_fit_path(saved_snapshot).write_bytes(b"not a parquet file")

    fitter = Counting()
    _, fits = with_saved_fit(saved_snapshot, fitter)(raw, curve, american)
    assert fitter.calls == 1, change
    assert "refit" in fits  # the fresh result, not the saved one


def test_saved_snapshots_never_use_a_saved_fit(saved_snapshot):
    # a matching fit file sits next to this snapshot, but it is not the demo: fit as usual
    fitter = Counting()
    source = MarketSource("snapshot", "SPY", "2026-10-16")
    with pytest.raises(Exception):  # noqa: B017 - the stand-in's empty fit can't build a market
        load_market(source, saved_snapshot.parent.parent, american=False, fit=fitter)
    assert fitter.calls == 1


def test_line_endings_do_not_change_the_code_hash():
    # git checks files out with CRLF on Windows and LF on Linux; the fingerprint must agree
    assert saved_fit._normalized(b"a = 1\r\nb = 2\r\n") == saved_fit._normalized(b"a = 1\nb = 2\n")
