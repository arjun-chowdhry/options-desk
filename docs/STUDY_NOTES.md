# Study notes: M1 pricing

One section per piece of the pricing module: why it works, the formulas you should be able
to write from memory, and interview-style questions with answers. Code lives in
`src/optdesk/pricing/`; the tests in `tests/pricing/` are the proof each claim holds.

Notation: spot `S`, strike `K`, expiry `T` in years, rate `r`, dividend yield `q`, vol `σ`,
forward `F = S·e^((r−q)T)`, `N` the standard normal CDF, `n` its density.

---

## 1. Black-Scholes-Merton price (`black_scholes.py`)

### Why it works
If the stock follows geometric Brownian motion with constant vol, you can replicate an option
by continuously trading the stock and a bond. A portfolio of the option and the right amount
of stock (delta) has no exposure to small stock moves, so it must earn the risk-free rate, or
there would be an arbitrage. That condition is a PDE whose solution is the Black-Scholes
formula. The stock's expected return drops out entirely, which is why the price only needs
`r`, `q` and `σ`, not anyone's view on where the stock is going.

Equivalently, the price is the discounted expected payoff under the *risk-neutral* measure,
where the stock drifts at `r − q`. `N(d2)` is the risk-neutral probability of finishing in the
money.

The dividend yield `q` (Merton's extension) matters because the most liquid listed options are
on indices and ETFs. The same formula covers FX options (`q` = foreign rate) and futures
options (`q = r`, i.e. Black-76).

### Key formulas
```
d1 = [ln(F/K) + ½σ²T] / (σ√T)        d2 = d1 − σ√T
C  = e^(−rT) [F·N(d1) − K·N(d2)]
P  = e^(−rT) [K·N(−d2) − F·N(−d1)]
Put-call parity:  C − P = S·e^(−qT) − K·e^(−rT)
Bounds:  max(S·e^(−qT) − K·e^(−rT), 0) ≤ C ≤ S·e^(−qT)
```
With `σ√T = 0` there's no uncertainty, so the price is `e^(−rT)·max(F − K, 0)`.

### Interview questions
1. **What assumptions does Black-Scholes make, and which fails most in practice?**
   Constant vol and rate, lognormal prices (no jumps), continuous frictionless trading, no
   arbitrage. Constant vol fails most visibly: implied vol varies by strike (the smile/skew)
   and by expiry, which is why desks build vol surfaces.
2. **Why doesn't the stock's expected return appear in the formula?**
   Because the option can be hedged with the stock. The hedged portfolio is riskless, so it
   earns `r` regardless of the stock's drift. Anyone who disagrees on the drift still agrees
   on the hedge cost.
3. **What does N(d2) mean? N(d1)?**
   `N(d2)` is the risk-neutral probability the call ends in the money. `N(d1)` is that
   probability under the stock-numeraire measure; times `e^(−qT)` it is the call's delta.
4. **Derive put-call parity without a model.**
   Long call + short put pays `S_T − K` at expiry in every state: the same as a forward. A
   forward is worth `S·e^(−qT) − K·e^(−rT)` today (buy `e^(−qT)` shares, reinvesting dividends,
   and borrow the PV of K). Same payoff means same price, or there's an arbitrage.
5. **What happens to a call's value as vol goes to infinity? To zero?**
   Infinite vol: the call is worth the whole (dividend-discounted) stock, `S·e^(−qT)`. Zero vol:
   the stock ends at the forward for sure, so the call is worth `e^(−rT)·max(F − K, 0)`.

---

## 2. Greeks (`greeks.py`)

### Why it works
The Greeks are partial derivatives of the price. Black-Scholes is smooth in every input, so
each has a closed form. They're how a desk decomposes P&L and decides hedges: a Taylor
expansion of the option's value over a short interval is approximately
`ΔV ≈ Δ·dS + ½Γ·dS² + vega·dσ + θ·dt`.

The deepest point is the **gamma-theta trade-off**. For a delta-hedged option the PDE says
`θ + ½σ²S²Γ ≈ rV`: a long-gamma position pays theta every day and earns `½Γ·dS²` from moves.
It makes money if realized vol beats the implied vol you paid. That's what the M4a hedging
simulator will show.

### Key formulas (call; `e^(−qT)` factors from dividends)
```
Δ     = e^(−qT) N(d1)                       put: −e^(−qT) N(−d1)
Γ     = e^(−qT) n(d1) / (S σ √T)            same for put
vega  = S e^(−qT) n(d1) √T                  same for put
θ     = −S e^(−qT) n(d1) σ / (2√T) − rK e^(−rT) N(d2) + qS e^(−qT) N(d1)
ρ     = K T e^(−rT) N(d2)                   put: −K T e^(−rT) N(−d2)
vanna = ∂Δ/∂σ = ∂vega/∂S = −e^(−qT) n(d1) d2 / σ
volga = ∂vega/∂σ = vega · d1 d2 / σ
charm = −∂Δ/∂T   (delta drift per year of time passing)
PDE:   θ + (r − q) S Δ + ½ σ² S² Γ = r V
```
Units here: vega per 1.00 vol (divide by 100 for per vol point), theta per year (divide by
365 for per day).

### Interview questions
1. **Why are gamma and vega the same for a call and a put with the same strike?**
   By put-call parity, call minus put is a forward, which is linear in `S` and doesn't depend
   on `σ`. A linear, vol-free position has zero gamma and zero vega, so the call's and put's
   must be equal.
2. **Where is gamma largest?**
   At the money, and it gets more concentrated as expiry approaches. Short-dated ATM options
   have huge gamma and huge theta. That's the trade-off from both sides.
3. **You're long a delta-hedged straddle. What has to happen for you to make money?**
   Realized vol over the holding period has to exceed the implied vol you bought at. You pay
   theta each day and earn roughly `½Γ·S²·(realized daily variance)` from rehedging.
4. **What is vanna, and why do equity desks care?**
   The sensitivity of delta to vol (equivalently, of vega to spot). In equities spot and vol
   are negatively correlated (spot down, vol up), so vanna determines how your hedge ratio
   changes in a selloff, beyond what gamma alone predicts.
5. **What is charm, and when does it bite?**
   How delta drifts as time passes with spot unchanged. Near expiry, OTM deltas decay to 0
   and ITM deltas go to ±1 quickly. Hedging at the close, you wake up with a different delta
   even if nothing moved, especially over weekends and near expiry.

---

## 3. Implied volatility (`implied_vol.py`)

### Why it works
The Black-Scholes price is strictly increasing in vol (vega > 0), so for any price strictly
between the no-arbitrage bounds there's exactly one vol that reproduces it. Implied vol is the
market's price expressed in a unit that's comparable across strikes and expiries.

There's no closed-form inverse, so we solve `BS(σ) − price = 0` numerically:

- **Newton:** `σ ← σ − (BS(σ) − price) / vega(σ)`. Quadratic convergence (correct digits
  roughly double per step), vectorized over a whole chain. In tests it handles about 99% of
  random options on its own.
- **Starting point (Manaster-Koehler):** `σ₀ = √(2|ln(F/K)|/T)` is the inflection point of
  price as a function of vol. Price is convex below and concave above it, and starting there
  makes Newton move monotonically toward the root instead of overshooting.
- **Brent fallback:** where vega is tiny (deep OTM, very short-dated), Newton steps blow up. Brent's
  method needs only a bracket where the error changes sign, and it is guaranteed to converge.
  It's slower and runs one option at a time, so it only handles the leftovers.
- **Solve on the OTM side:** an ITM call is mostly intrinsic value. Parity converts it to the
  OTM put with the same vol, whose price is all time value, so no digits are lost.
- **NaN outside the bounds:** below intrinsic or above the upper bound, no vol works. That's
  a bad quote or an arbitrage, and NaN is more honest than a made-up number.

### Key formulas
```
Newton:      σ_{n+1} = σ_n − [BS(σ_n) − P_mkt] / vega(σ_n)
Start:       σ₀ = max( √(2|ln(F/K)| / T), 0.05 )
Parity flip: P = C − (S e^(−qT) − K e^(−rT))
Valid iff:   0 < (OTM price) < S e^(−qT) (call) or K e^(−rT) (put)
Brenner-Subrahmanyam ATM approximation:  σ ≈ √(2π/T) · C / S
```

### Interview questions
1. **Why do traders quote implied vol instead of price?**
   Price depends on strike, expiry, spot and rates, so prices aren't comparable. Vol
   normalizes all that. A 20-vol option is cheap or rich relative to other vols and to
   realized vol, whatever its dollar price.
2. **Why Newton's method, and when does it fail?**
   Vega is the exact derivative, available in closed form, so Newton converges in a few steps.
   It fails when vega is near zero (deep OTM or very short-dated), where one step can jump to a
   negative or absurd vol. That's what the Brent fallback is for.
3. **Can an option have no implied vol?**
   Yes, if its price is outside the no-arbitrage bounds: e.g. a call below
   `S·e^(−qT) − K·e^(−rT)` or above `S·e^(−qT)`. In real data this usually means a stale or
   crossed quote, or a wrong dividend/rate assumption.
4. **Why invert the OTM option rather than the ITM one?**
   An ITM option's price is mostly intrinsic. The vol information is in the small time value,
   so small price errors become large vol errors. Parity gives the OTM option with the same
   implied vol and a price that's all time value.
5. **Quick ATM vol from a price, no calculator?**
   `σ ≈ 2.5 · C / (S √T)` (since √(2π) ≈ 2.5). A 1-year ATM call on a $100 stock at $8 is
   about 20 vol.
