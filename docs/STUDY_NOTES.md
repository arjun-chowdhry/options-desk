# Study notes

One section per piece of the toolkit: why it works, the formulas you should be able to write
from memory, and interview-style questions with answers. Code lives in `src/optdesk/`; the
tests in `tests/` are the proof each claim holds.

- **M1 Pricing** (`pricing/`): sections 1–4
- **M2 Data** (`data/`): sections 5–8

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

---

## 4. CRR binomial tree (`binomial.py`)

### Why it works
Chop time to expiry into `N` steps. In each step the stock goes up by `u` or down by `d`. In
a one-step world, an option can be replicated exactly with stock and cash, so its price is a
discounted average of its two possible values with weights `p` and `1 − p`. Those weights are
the risk-neutral probabilities: the ones that make the stock earn `r − q` on average. Chain
those one-step problems backward from expiry and you price any payoff.

**American exercise** fits in naturally: at every node the holder chooses the better of
*hold* (the discounted average of the children) and *exercise now* (intrinsic value). That
`max` is the whole difference between the American and European tree.

**Why CRR:** `u = e^(σ√dt)`, `d = 1/u`. Since `u·d = 1` the tree recombines, so up-then-down
lands where down-then-up does. That gives `i + 1` nodes after `i` steps, not `2^i`, so a 2,000-step
tree is cheap. The grid is symmetric around today's spot, and the drift only enters through
`p`. As `N → ∞` the tree converges to Black-Scholes, with error O(1/N) that oscillates between
odd and even `N` (averaging `N` and `N + 1` damps it).

**When early exercise matters:**
- **American put:** deep in the money with `r > 0`, exercising now gets you `K` today to earn
  interest, which can beat the remaining optionality.
- **American call, no dividends:** never exercise early (Merton). You'd pay `K` sooner and throw
  away time value, so it's worth exactly the European.
- **American call with dividends:** exercising just before a big dividend can pay, because the
  holder of the stock gets the dividend and the option holder doesn't.

### Key formulas
```
dt = T/N     u = e^(σ√dt)     d = 1/u
p  = (e^((r−q)dt) − d) / (u − d)          must lie in [0, 1]
node (i, j) price:   S · u^(2j − i)        (j up-moves out of i steps)
European:  V(i, j) = e^(−r dt) [p V(i+1, j+1) + (1 − p) V(i+1, j)]
American:  V(i, j) = max( that,  intrinsic at S·u^(2j−i) )
Error vs Black-Scholes (European):  O(1/N), oscillating
```

### Interview questions
1. **Why use a binomial tree when we have Black-Scholes?**
   Black-Scholes is for European exercise only. American options (most single-stock listed
   options) can be exercised early, which has no closed form. The tree checks the exercise
   decision at every node.
2. **When should you exercise an American put early?**
   When it's deep enough in the money that getting `K` now and earning interest on it is
   worth more than the remaining time value. That's more likely with higher rates and lower
   vol. With `r = 0` (and no dividends) it's never optimal.
3. **Why is an American call on a non-dividend stock worth the same as the European?**
   A call is always worth at least `S − K·e^(−rT)`, which is more than the `S − K` you'd get
   by exercising. Selling the call always beats exercising it, so the early-exercise right is
   worthless.
4. **What does `p` represent, and what if it's outside [0, 1]?**
   The risk-neutral up-probability, chosen so the stock's expected growth per step is
   `e^((r−q)dt)`. It's not a real-world probability. If it's outside [0, 1] the steps are too
   coarse (the drift per step exceeds the vol per step) and the tree admits arbitrage. Use
   more steps.
5. **How fast does the tree converge, and how can you speed it up?**
   Error is O(1/N), with an odd/even oscillation because the strike falls at different places
   relative to the nodes. Averaging `N` and `N + 1` steps helps, as does Richardson
   extrapolation or replacing the last step with Black-Scholes values (the BBS method).
   Doubling `N` roughly halves the error but quadruples the work, since an `N`-step tree has
   about `N²/2` nodes.

---

# M2: Market data

Notation as above, plus discount factor `DF = e^(−rT)`, mid `= (bid + ask)/2`, and
log-moneyness `k = ln(K/F)`.

## 5. Risk-free rates (`rates.py`)

### Why it works
Every price and implied vol needs a discount rate. We take it from US Treasuries, the standard
proxy for the risk-free dollar rate, using three yield indices Yahoo publishes: `^IRX`
(13-week bill), `^FVX` (5-year) and `^TNX` (10-year). The package uses continuous compounding
everywhere, but each market quotes its own way, so every quote is converted first:

- **T-bills** trade at a *discount*: the quote `d` says the price is `1 − d·days/360`.
  We find the continuous `r` that discounts 1 to that price over `days/365` years.
- **Notes** are quoted as a semiannual *bond-equivalent yield* `y`. Equal growth means
  `(1 + y/2)^(2T) = e^(rT)`, so `r = 2·ln(1 + y/2)`.

Between maturities we interpolate the rate linearly in `T`, and we hold it flat beyond the
first and last points. Rates are known to within a few basis points while vol is uncertain
by whole points, so a rate error moves option prices much less than vol uncertainty does and
this simple curve is good enough. A rates desk would bootstrap a full
SOFR/OIS curve instead.

### Key formulas
```
T-bill:     price = 1 − d·days/360          r = −ln(price) / (days/365)
Note:       r = 2·ln(1 + y/2)               (general: r = m·ln(1 + y/m))
Discount:   DF(T) = e^(−r(T)·T)
Interp:     r(T) linear between knots, flat outside
```

### Interview questions
1. **Why is a T-bill's discount yield lower than its true yield?**
   It divides the dollar discount by *face value*, but you actually invest the (lower) price,
   and it uses a 360-day year. Both conventions make the quoted number understate the return.
2. **Continuous vs semiannual: which number is bigger for the same investment?**
   The semiannual yield. Compounding more often needs a lower stated rate to produce the
   same growth, and continuous is the limit, so `r_cont < y_semi` (4% semiannual ≈ 3.96%
   continuous).
3. **Why use continuous compounding in derivatives pricing?**
   Discount factors multiply cleanly (`e^(−r₁t₁)·e^(−r₂t₂)`), forwards are `S·e^((r−q)T)`,
   and the math of Black-Scholes is in continuous time. One convention avoids mistakes.
4. **How much does a rate error matter for an option price?**
   About rho × error. A 3-month ATM option on a $100 stock has rho ≈ 12.5 (per 1.00 rate)
   and vega ≈ 20 (per 1.00 vol), so a 10bp rate error moves it ~1.2¢, less than a 0.1
   vol-point error (~2¢). Vol is uncertain by whole points, rates by a few bp. Rates matter
   more for long expiries and deep ITM options.
5. **What would a real desk use instead of Treasuries?**
   An OIS/SOFR discount curve (the rate collateralized trades actually earn), bootstrapped from
   many instruments. For equity options it can also *imply* the rate from box spreads or from
   put-call parity across strikes.

---

## 6. Fetching chains (`fetch.py`)

### Why it works
yfinance scrapes Yahoo Finance: `Ticker("SPY").options` lists the expiries, and
`option_chain(expiry)` returns a calls table, a puts table and the underlying's quote. We stack
every expiry's calls and puts into one long table (`expiry`, `option_type`, `strike`, `bid`,
`ask`, `last`, `last_trade`, `volume`, `open_interest`, plus `spot` and the download time
`asof`). The quotes stay **raw**: no filtering at download time, so cleaning rules can change
later and be re-run on old snapshots.

It's the only module that touches the network. In tests a fake `Ticker` serves a saved chain,
and real network connections are blocked, so the suite proves it runs offline.

**Why the saved chain is synthetic.** A real chain has no answer key. Ours is generated from
Black-Scholes with a known forward, rate and smile, so the tests can check the cleaner finds
the *right* forward and vols, not just *some* numbers. Bad quotes (zero bids, crossed, wide,
stale) are planted on purpose.

**Data caveats to know cold:** Yahoo quotes can be ~15 minutes delayed. Outside market hours
many bids and asks show as 0. Bid, ask and last trade aren't synchronized (the last trade may
be hours old). US single-stock and ETF options (SPY) are *American*, while SPX is European and
cash-settled.

### Key facts
```
OCC symbol:  SPY261023C00580000  = root, YYMMDD, C/P, strike × 1000 (8 digits)
Expiry:      equity options stop trading at 4:00pm New York time on expiry day
Contract:    100 shares per contract (dollar price × 100 = premium paid)
```

### Interview questions
1. **Why not use the `last` price to compute implied vol?**
   The last trade can be hours old, so it doesn't match the current spot, and it was at the bid
   or the ask, not fair value. The mid of the current bid and ask is simultaneous with the
   underlying quote and unbiased between buyer and seller.
2. **What's the difference between SPY and SPX options?**
   SPY options are American, physically settled in ETF shares, and the ETF pays dividends.
   SPX options are European, cash-settled, 10× bigger notional, and mostly PM-settled with
   some AM-settled expiries. SPX has Section 1256 tax treatment. European exercise makes SPX
   the cleaner input for Black-Scholes and vol surfaces.
3. **Why do bids show as zero after hours?**
   Market makers pull their quotes when the market closes, so there's no live bid. Data
   vendors show 0 or the last value. That's why snapshots should be taken during the session.
4. **What's open interest vs volume?**
   Volume is the number of contracts traded today. Open interest is the number of contracts
   outstanding, updated once a day by the OCC. High OI with low volume means a lot of existing
   positions but little trading today.
5. **Why store raw data rather than cleaned data?**
   Cleaning involves judgment calls (spread thresholds, staleness). Keeping the raw quotes lets
   you change those rules and re-run history. You can't recover quotes you threw away.
