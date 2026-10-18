# options-desk

An options desk toolkit in Python: pricing and Greeks, implied vol, volatility surfaces, and
hedging and market-making simulators. Every numerical function is unit-tested against known
answers (textbook values, no-arbitrage identities, finite differences).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Pricing (`optdesk.pricing`)

| Module | What it does | Key design choice |
| --- | --- | --- |
| `black_scholes.py` | European call/put price with continuous dividend yield `q` | Forward form `e^(−rT)[F·N(d1) − K·N(d2)]`, consistent with the log-moneyness surface work later; exact zero-vol / expiry limit |
| `greeks.py` | Delta, gamma, vega, theta, rho, vanna, volga, charm, plus per-vol-point and per-day helpers | Closed forms (exact, fast); finite differences only in tests as an independent check |
| `implied_vol.py` | Implied vol from prices, vectorized over a chain | Newton from the Manaster-Koehler inflection point, Brent fallback where vega is tiny, inverts the OTM side via parity, NaN outside no-arbitrage bounds |
| `binomial.py` | CRR tree for American and European options | Recombining `u·d = 1` grid; converges to Black-Scholes at O(1/N) |

```python
from optdesk.pricing.black_scholes import bs_price
from optdesk.pricing.greeks import delta, vega_per_vol_point
from optdesk.pricing.implied_vol import implied_vol
from optdesk.pricing.binomial import binomial_price

px = bs_price(S=100, K=105, T=0.5, r=0.04, sigma=0.22, q=0.01, call=True)
implied_vol(px, S=100, K=105, T=0.5, r=0.04, q=0.01)  # -> 0.22
(
    delta(S=100, K=105, T=0.5, r=0.04, sigma=0.22, q=0.01),
    vega_per_vol_point(S=100, K=105, T=0.5, r=0.04, sigma=0.22),
)
binomial_price(S=100, K=105, T=0.5, r=0.04, sigma=0.22, call=False, american=True)
```

How it's tested: Hull and Haug textbook values; put-call parity, price bounds and Greek parity
identities with Hypothesis (hundreds of random inputs); every Greek against a finite
difference; the Black-Scholes PDE; implied-vol round trips on both solver paths; binomial
convergence to Black-Scholes, American ≥ European, and no early exercise of calls without
dividends.

Conventions: `T` in years (ACT/365); `r`, `q` continuously compounded; vega per 1.00 vol,
theta per year. Interview prep for this module: [`docs/STUDY_NOTES.md`](docs/STUDY_NOTES.md).
