"""A saved SVI fit for the demo snapshot, so the hosted app opens in seconds, not a minute.

Why save a fit
--------------
Building a market from a snapshot means cleaning every quote, stripping the early-exercise
premium on a CRR tree, and fitting SVI to every expiry: about 25 seconds on a laptop for the
28-expiry demo, and longer on the free hosting tier, which also sleeps when idle and so
starts cold for most visitors. The demo snapshot never changes, so its fit can be computed
once (``python scripts/fit_demo.py``) and committed. Only the demo uses this: saved and live
snapshots are fitted fresh every time, because nobody has checked a saved fit for them.

Why a fingerprint
-----------------
A saved fit is only right if it came from the same quotes, the same settings and the same
code. So the file records a fingerprint of all three, and ``load_fit`` returns nothing
unless it matches exactly; the app then refits as usual. That way a stale fit is never
shown: it just costs the old load time until someone regenerates it, and a test fails in CI
when the committed fit is out of date. The fingerprint holds:

- the SHA-256 of the snapshot file's bytes (any change to the quotes or rate curve);
- the fit settings: ``american`` and every default argument of ``clean_chain``,
  ``fit_chain`` and ``check_surface`` (the filters, the SVI weighting, the arbitrage grid);
- a SHA-256 of the source code of every module the fit runs through, with line endings
  normalized so a Windows checkout and a Linux server agree. Even a docstring edit there
  invalidates the fit; that is deliberately strict, since regenerating it takes a minute.

File format
-----------
One Parquet file next to the snapshot, ``{date}.fit.parquet`` (about 200 KB, smaller than
the snapshot itself). Its table is the cleaned quotes; the fit table (28 rows) and the
fingerprint go in the file's metadata. Both tables round-trip bit for bit, the same as the
snapshot format, and one file means the quotes and the fit can't get out of step.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import inspect
import io
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "FIT_MODULES",
    "fit_fingerprint",
    "load_fit",
    "save_fit",
    "saved_fit_path",
    "with_saved_fit",
]

FIT_FORMAT = 1
_METADATA_KEY = b"optdesk_fit"
# Every module the fit runs through: snapshot loading, cleaning (parity forward, rates,
# de-Americanizing on CRR trees, implied vols), SVI fitting and the arbitrage checks.
FIT_MODULES = (
    "optdesk.data.store",
    "optdesk.data.rates",
    "optdesk.data.clean",
    "optdesk.pricing.black_scholes",
    "optdesk.pricing.greeks",
    "optdesk.pricing.implied_vol",
    "optdesk.pricing.binomial",
    "optdesk.surface.svi",
    "optdesk.surface.arbitrage",
    "optdesk.surface.market",  # fit_quotes, which chains the steps together
)

Fits = tuple[pd.DataFrame, pd.DataFrame]  # (clean quotes, fits), as fit_quotes returns
Fitter = Callable[[pd.DataFrame, object, bool], Fits]


def saved_fit_path(snapshot: Path | str) -> Path:
    """Where the saved fit of ``snapshot`` lives, e.g. ``SPY/2026-10-22.fit.parquet``.

    The ``.fit`` stem is not a date, so ``list_snapshots`` never mistakes it for a snapshot.
    """
    snapshot = Path(snapshot)
    return snapshot.with_name(f"{snapshot.stem}.fit.parquet")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized(source: bytes) -> bytes:
    """Source code with Windows line endings turned into Unix ones (git converts on checkout)."""
    return source.replace(b"\r\n", b"\n")


def _code_hash() -> str:
    h = hashlib.sha256()
    for name in FIT_MODULES:
        origin = importlib.util.find_spec(name).origin
        h.update(name.encode() + b"\0" + _normalized(Path(origin).read_bytes()) + b"\0")
    return h.hexdigest()


def _defaults(func) -> dict:
    """A function's default arguments, as JSON-friendly values (arrays become a summary)."""
    out = {}
    for name, p in inspect.signature(func).parameters.items():
        if p.default is inspect.Parameter.empty:
            continue
        value = p.default
        if isinstance(value, np.ndarray):
            value = {"min": float(value.min()), "max": float(value.max()), "n": int(value.size)}
        out[name] = value
    return out


def fit_settings(american: bool) -> dict:
    """Everything that sets how a chain is cleaned and fitted, besides the code itself."""
    from optdesk.data.clean import clean_chain
    from optdesk.surface.arbitrage import check_surface
    from optdesk.surface.svi import fit_chain

    return {
        "american": bool(american),
        "clean_chain": _defaults(clean_chain),
        "fit_chain": _defaults(fit_chain),
        "check_surface": _defaults(check_surface),
    }


def fit_fingerprint(snapshot: Path | str, american: bool) -> dict:
    """The fingerprint a saved fit of ``snapshot`` must carry to be used (see module notes)."""
    return {
        "format": FIT_FORMAT,
        "snapshot_sha256": _sha256(Path(snapshot).read_bytes()),
        "settings": fit_settings(american),
        "code_sha256": _code_hash(),
    }


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf, compression="zstd")
    return buf.getvalue()


def save_fit(path: Path | str, clean: pd.DataFrame, fits: pd.DataFrame, fingerprint: dict) -> Path:
    """Write a fit (from ``fit_quotes``) and its fingerprint to ``path``; return the path."""
    path = Path(path)
    note = {
        "fingerprint": fingerprint,
        "fits_parquet": base64.b64encode(_to_parquet_bytes(fits)).decode("ascii"),
    }
    table = pa.Table.from_pandas(clean)
    metadata = {**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(note).encode()}
    table = table.replace_schema_metadata(metadata)
    tmp = path.with_suffix(".parquet.tmp")  # write then rename, so a crash leaves no half file
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def load_fit(path: Path | str, fingerprint: dict) -> Fits | None:
    """The saved ``(clean, fits)`` at ``path`` if its fingerprint equals ``fingerprint``.

    Returns None, so the caller refits, if the file is missing, unreadable, or was made
    from other quotes, settings or code.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        table = pq.read_table(path)
        note = json.loads((table.schema.metadata or {})[_METADATA_KEY])
        # compare as JSON, the form it was saved in (tuples come back as lists, and so on)
        if note["fingerprint"] != json.loads(json.dumps(fingerprint)):
            return None
        fits = pq.read_table(io.BytesIO(base64.b64decode(note["fits_parquet"]))).to_pandas()
        return table.to_pandas(), fits
    except Exception:  # corrupt or foreign file: refitting is always safe
        return None


def with_saved_fit(snapshot: Path | str, fit: Fitter) -> Fitter:
    """A fitter that uses the saved fit of ``snapshot`` when it is current, else calls ``fit``."""

    def fitter(raw: pd.DataFrame, curve, american: bool) -> Fits:
        saved = load_fit(saved_fit_path(snapshot), fit_fingerprint(snapshot, american))
        return saved if saved is not None else fit(raw, curve, american)

    return fitter
