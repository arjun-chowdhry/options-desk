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
  pricing/  black_scholes.py, greeks.py, implied_vol.py, binomial.py
  data/     fetch.py, clean.py, rates.py, store.py
  surface/  svi.py, arbitrage.py, plot.py
  sim/      paths.py, hedging.py, market_maker.py
  app/      main.py, pages/
tests/      mirrors the above; fixtures/ holds a saved chain for offline tests
data/snapshots/   gitignored, {TICKER}/{YYYY-MM-DD}.parquet
```

## Milestones

- [x] **M0 Scaffolding:** pyproject, package skeleton, placeholder test.
- [ ] **M1 Pricing:** BS price with q; delta, gamma, vega, theta, rho, vanna, volga, charm;
  implied vol (Newton with Brent fallback, NaN outside no-arb bounds); CRR binomial with
  American exercise. Tests: Hull values, put-call parity (Hypothesis), every Greek vs finite
  difference, price bounds, IV round-trip, binomial converges to BS, American put >= European.
- [ ] **M2 Data:** fetch chain (yfinance), clean (drop zero bids, crossed/wide/stale quotes;
  mid; T; forward from parity; IV), Parquet snapshots. Tests use a fixture chain plus a mocked
  yfinance.
- [ ] **M3 Surface:** raw SVI fit per expiry with parameter bounds; butterfly check (Gatheral
  g(k) >= 0) and calendar check (w non-decreasing in T); Plotly 3D surface + smiles. Tests:
  parameter recovery, Axel Vogt arbitrage example flagged, crossed slices flagged.
- [ ] **M4a Hedging sim:** GBM (exact) and Heston (full-truncation Euler) paths; delta-hedge
  P&L with transaction costs. Tests: theoretical moments, martingale property, P&L std shrinks
  as 1/sqrt(rebalances), costs reduce P&L, wrong-vol P&L sign.
- [ ] **M4b Market maker:** Avellaneda-Stoikov reservation price and spread, Poisson fills,
  inventory limits. Tests: symmetric quotes at zero inventory, skew direction, spread grows with
  gamma and sigma, reproducibility, skewing reduces inventory variance.
- [ ] **M5 App:** Streamlit pages: Pricer, Chain & IV, Surface, Hedging, Market Maker. Smoke
  tests with `streamlit.testing.v1.AppTest`.
- [ ] **M6 Polish:** README with screenshots, design decisions, CI (GitHub Actions: pytest +
  ruff), coverage badge, talking-points doc.

Each milestone is done when: tests green, `ruff check .` clean, README section written, committed.
