"""Tests for the surface plots: the figures show the right numbers, not just "a figure".

The 3D surface's z values must equal the SVI vol at each grid point, and be blank beyond
the quoted strikes; the smile panels must plot the market vols and the fitted curve. The
command line runs end to end on the synthetic market and writes both HTML files.
"""

import numpy as np
import pandas as pd
import pytest

from optdesk.surface import plot
from optdesk.surface.plot import (
    main,
    png_slices,
    smile_figure,
    surface_figure,
    surface_png_figure,
)
from optdesk.surface.svi import SVIParams


def test_surface_values_are_svi_vols(market_fits, market_clean):
    fig = surface_figure(market_fits, market_clean)
    surface, quotes = fig.data
    z_axis, T, vol, k = (
        np.asarray(surface.x),
        np.asarray(surface.y),
        np.asarray(surface.z, dtype=float),
        np.asarray(surface.customdata, dtype=float),
    )
    fits = market_fits.sort_values("T").reset_index(drop=True)
    np.testing.assert_allclose(T, fits["T"])
    for i, row in fits.iterrows():
        params = SVIParams.from_mapping(row)
        sd = np.sqrt(params.total_variance(0.0))
        np.testing.assert_allclose(k[i], sd * z_axis, rtol=1e-12)  # z is k in ATM std devs
        drawn = ~np.isnan(vol[i])
        np.testing.assert_allclose(vol[i, drawn], 100 * params.implied_vol(k[i, drawn], row["T"]))
        quoted = market_clean.loc[market_clean["expiry"] == row["expiry"], "k"]
        assert k[i, drawn].min() >= quoted.min() - 0.0201  # nothing drawn beyond the quotes
        assert k[i, drawn].max() <= quoted.max() + 0.0201
    assert len(quotes.x) == len(market_clean)
    np.testing.assert_allclose(np.asarray(quotes.z), 100 * market_clean["iv"])


def test_surface_without_quotes(market_fits):
    fig = surface_figure(market_fits)
    assert len(fig.data) == 1
    assert not np.isnan(np.asarray(fig.data[0].z, dtype=float)).all()


def test_smile_panels(market_fits, market_clean):
    fig = smile_figure(market_fits, market_clean)
    assert len(fig.data) == 2 * len(market_fits)  # dots + line per expiry
    assert sum(trace.showlegend for trace in fig.data) == 2  # one legend entry per series
    fits = market_fits.sort_values("T").reset_index(drop=True)
    for i, row in fits.iterrows():
        dots, line = fig.data[2 * i], fig.data[2 * i + 1]
        quotes = market_clean[market_clean["expiry"] == row["expiry"]]
        np.testing.assert_allclose(np.asarray(dots.y), 100 * quotes["iv"])
        expected = 100 * SVIParams.from_mapping(row).implied_vol(np.asarray(line.x), row["T"])
        np.testing.assert_allclose(np.asarray(line.y), expected)


def test_static_png_matches_the_interactive_surface(market_fits, market_clean):
    """The README image plots the same SVI vols as the Plotly surface, one smile per expiry."""
    interactive = np.asarray(surface_figure(market_fits, market_clean).data[0].z, dtype=float)
    fig = surface_png_figure(market_fits, market_clean)
    smiles = fig.axes[1].get_lines()[0::2]  # fit line, then quote dots, per expiry
    assert len(smiles) == len(market_fits)
    for i, line in enumerate(smiles):
        np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), interactive[i])
    dots = fig.axes[1].get_lines()[1::2]
    assert sum(len(d.get_ydata()) for d in dots) == len(market_clean)


def test_png_slices_on_a_live_style_chain():
    """31 expiries from 1 day to 2.3 years (like SPY): drop under a week, keep 8 spread in log T."""
    days = [1, 2, 3, 4, 7, 8, 9, 10, 11, 18, 25, 32, 39, 42, 60, 73, 88, 102, 123, 130, 151,
            163, 241, 254, 333, 346, 424, 459, 606, 788, 823]  # fmt: skip
    fits = pd.DataFrame({"T": np.array(days[::-1]) / 365, "expiry": range(len(days))})
    picked = png_slices(fits)
    assert len(picked) == plot.PNG_MAX_SLICES
    assert picked["T"].is_monotonic_increasing
    assert picked["T"].iloc[0] == 7 / 365 and picked["T"].iloc[-1] == 823 / 365  # both ends
    gaps = np.diff(np.log(picked["T"]))
    assert gaps.max() < 2.5 * gaps.min()  # roughly even in log maturity


def test_png_slices_keeps_short_chains_whole():
    few = pd.DataFrame({"T": [2 / 365, 30 / 365, 0.5], "expiry": range(3)})
    assert list(png_slices(few)["T"]) == [30 / 365, 0.5]  # only the 2-day one goes
    all_short = pd.DataFrame({"T": [1 / 365, 2 / 365], "expiry": range(2)})
    assert len(png_slices(all_short)) == 2  # nothing a week out: keep everything


def test_png_shows_only_quotes_inside_the_z_window(market_fits, market_clean, monkeypatch):
    monkeypatch.setattr(plot, "PNG_Z_RANGE", (-1.0, 1.0))
    fig = surface_png_figure(market_fits, market_clean)
    smiles, dots = fig.axes[1].get_lines()[0::2], fig.axes[1].get_lines()[1::2]
    for line in dots:
        z = np.asarray(line.get_xdata(), dtype=float)
        assert len(z) and z.min() >= -1.0 and z.max() <= 1.0
    assert sum(len(d.get_xdata()) for d in dots) < len(market_clean)
    assert len(smiles) == len(market_fits)


def test_command_line_writes_both_plots(tmp_path, capsys):
    png = tmp_path / "img" / "surface.png"
    assert main(["--out", str(tmp_path), "--png", str(png)]) == 0
    out = capsys.readouterr().out
    assert "SVI fits for synthetic SPY" in out and "Arbitrage checks" in out
    for name in ("surface.html", "smiles.html"):
        assert (tmp_path / name).stat().st_size > 100_000  # self-contained, includes plotly.js
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_command_line_from_a_saved_snapshot(market, tmp_path, capsys):
    """The Monday workflow: fetch saves a snapshot, then the plot command fits it."""
    from optdesk.data.store import save_snapshot

    path = save_snapshot(market.raw, market.curve, tmp_path / "snapshots")
    out_dir = tmp_path / "plots"
    args = ["--ticker", "SPY", "--date", path.stem, "--root", str(tmp_path / "snapshots")]
    assert main([*args, "--out", str(out_dir)]) == 0
    assert f"SVI fits for SPY {path.stem}" in capsys.readouterr().out
    assert (out_dir / "surface.html").exists() and (out_dir / "smiles.html").exists()
    with pytest.raises(SystemExit):  # --ticker without --date is a usage error
        main(["--ticker", "SPY", "--out", str(out_dir)])
