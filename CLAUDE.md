# CLAUDE.md

Options desk toolkit in Python, built by a college student (Arjun) to show in sales & trading
recruiting. Arjun is new to the terminal and medium-level in Python.

## Rules (non-negotiable)

1. **Every numerical function gets a unit test.** Test against known answers (textbook values,
   parity, finite differences, theoretical moments), not just "it runs". Tests must run offline.
2. **Every model choice gets a plain-English docstring explaining why.** E.g. why CRR over
   Jarrow-Rudd, why full-truncation Euler for Heston, why vega-weighted SVI fits. Write it so a
   recruiter or interviewer could follow it.
3. **Build one milestone at a time.** Finish a milestone (tests green, ruff clean, committed)
   before starting the next. Don't scaffold future milestones' code early.
4. **Explain each terminal command before running it.** Say what it does and why.

## Commands

```powershell
.venv\Scripts\Activate.ps1              # activate the project's virtual environment
pip install -e ".[dev]"                 # install package (editable) + dev tools
pytest                                  # run all tests
pytest --cov=optdesk                    # tests + coverage report
ruff check .                            # lint
ruff format .                           # auto-format
streamlit run src/optdesk/app/main.py   # dashboard (from M5)
python scripts/snapshot.py              # save today's SPY snapshot (Task Scheduler: weekdays 2:15pm Central)
```

## Conventions

- `src/` layout; package name `optdesk`. Tests in `tests/` mirror the package folders.
- T in years (ACT/365); r and q are continuous annual rates; sigma is annualized.
- Numerical functions accept NumPy arrays and broadcast.
- Greek units: vega/volga per 1.00 vol, theta/charm per year, rho per 1.00 rate (provide
  per-vol-point / per-day helpers separately).
- Surface work uses log-moneyness `k = ln(K/F)` and total variance `w = sigma^2 * T`.
- Randomness goes through a seeded `np.random.Generator`, so simulations are reproducible.
- The Streamlit app holds no math; it only calls tested modules.
- yfinance (`data/fetch.py`) is the only network code; mock it in tests.

## Structure

```
src/optdesk/
  pricing/  black_scholes.py, greeks.py, implied_vol.py, binomial.py, portfolio.py
  data/     fetch.py, clean.py, rates.py, store.py
  surface/  svi.py, arbitrage.py, plot.py, synthetic.py, source.py (swappable VolSurface),
            market.py (MarketData bundle: synthetic or snapshot, the app's data source)
  sim/      paths.py, hedging.py, plot.py, market_maker.py, mm_plot.py, lab.py (app experiments)
  app/      main.py (sidebar + tabs), views/ (one file per tab), charts.py (Plotly figures)
tests/      mirrors the above; fixtures/ holds a saved chain for offline tests
data/snapshots/   gitignored, {TICKER}/{YYYY-MM-DD}.parquet
data/plots/       gitignored, HTML output of python -m optdesk.surface.plot
```

## Milestones

- [x] **M0 Scaffolding:** pyproject, package skeleton, placeholder test.
- [x] **M1 Pricing:** BS price with q; delta, gamma, vega, theta, rho, vanna, volga, charm;
  implied vol (Newton with Brent fallback, NaN outside no-arb bounds); CRR binomial with
  American exercise. Tests: Hull values, put-call parity (Hypothesis), every Greek vs finite
  difference, price bounds, IV round-trip, binomial converges to BS, American put >= European.
- [ ] **M2 Data** (code complete, live test pending): fetch chain (yfinance), clean (drop zero
  bids, crossed/wide/stale quotes; mid; T; forward from parity; IV), Parquet snapshots. Tests
  use a fixture chain plus a mocked yfinance.
  - [x] Code, offline tests, README section, study notes (merged to main).
  - [x] Live-fetch test on Monday 2026-10-19 at 3:40pm New York (branch `m7-live`): 6-expiry
    snapshot in `data/snapshots/`, full chain (32 expiries) in `data/snapshots_full/`. Fix:
    same-day (0DTE) expiries are now dropped (`clean_chain(min_days=1)`). Daily job:
    `scripts/snapshot.py`.
- [ ] **M3 Surface** (code complete and merged, live fit pending): raw SVI fit per expiry
  with parameter bounds; butterfly check (Gatheral g(k) >= 0) and calendar check (w
  non-decreasing in T); Plotly 3D surface + smiles; de-Americanizing for SPY (early-exercise
  premium stripped on a CRR tree in `clean_chain(american=True)`); synthetic SSVI chain
  (`surface/synthetic.py`) as the answer key. Tests: parameter recovery, Axel Vogt arbitrage
  example flagged, crossed slices flagged.
  - [x] Code, tests, README section, study notes (sections 9-12) (merged to main).
  - [x] Fit Monday's live snapshot (branch `m7-live`): RMSE 0.05-0.23 vol points; every
    butterfly flag is in the extrapolated call wing; 6 calendar crossings touch the quoted
    range at its edge (0.02-0.65 vol points; independent per-slice fits). PNGs in
    `docs/images/live/`.
- [x] **M4a Hedging sim:** GBM (exact) and Heston (full-truncation Euler) paths; delta-hedge
  P&L with transaction costs. Tests: theoretical moments, martingale property, P&L std shrinks
  as 1/sqrt(rebalances), costs reduce P&L, wrong-vol P&L sign. README section, study notes
  (sections 13-15), charts in `docs/images/` (`python -m optdesk.sim.plot`). Merged to main.
- [x] **M4b Market maker** (merged to main): Avellaneda-Stoikov reservation price and spread,
  Poisson fills,
  inventory limits. Tests: symmetric quotes at zero inventory, skew direction, spread grows with
  gamma and sigma, reproducibility, skewing reduces inventory variance.
  - [x] A-S in vol space (skew by book vega), delta hedging, pre-trade vega/gamma limits, P&L
    split into spread / hedging / inventory; swappable surface (`surface/source.py`, synthetic
    by default). Study + charts (`python -m optdesk.sim.mm_plot`), README section, study
    notes (sections 16-18).
  - [x] Merged to main.
  - [ ] **Results are on the synthetic SSVI surface only.** Every number in the README, the
    study notes (section 18) and `docs/images/mm_*.png` comes from the synthetic surface. Rerun
    on live data once Monday's snapshot is fitted (M3 live fit), compare, and update the docs:
    `python -m optdesk.sim.mm_plot --ticker SPY --date 2026-10-19 --out data/plots`.
    Rerun done on branch `m7-live` (full-chain snapshot, charts in `docs/images/live/`, and
    the hedging study now takes `--ticker/--date` too). README and study notes not yet updated.
- [x] **M5 App** (merged to main): Streamlit tabs
  Surface, Hedging simulator (adjustable parameters), Market maker, Portfolio Greeks, on a
  swappable market (`surface/market.py`: synthetic by default, saved snapshots appear in the
  sidebar dropdown). Math lives in `pricing/portfolio.py` and `sim/lab.py`. Smoke tests with
  `streamlit.testing.v1.AppTest`. README section, study notes (sections 19-20).
  - [x] Code, tests, README section, study notes.
  - [x] Merged to main. Run: `streamlit run src/optdesk/app/main.py`.
- [ ] **M6 Polish:** README with screenshots, design decisions, CI (GitHub Actions: pytest +
  ruff), coverage badge, talking-points doc.
  - [x] README rewritten (summary, results with screenshots and numbers, install/run,
    assumptions and limitations, module reference); `docs/images/surface.png` from
    `python -m optdesk.surface.plot --png docs/images/surface.png`; study notes section 21
    (15 whole-project interview questions). Merged to main.
  - [x] CI: `.github/workflows/ci.yml` runs ruff check, ruff format --check and pytest on
    every push, on the Python in `.python-version` (3.13); ruff pinned in pyproject.
    `tests/conftest.py` blocks all non-loopback network for every test. Badge in README.
    Merged to main.
  - [ ] Coverage badge, talking-points doc.

Each milestone is done when: tests green, `ruff check .` clean, README section written, committed.
