# Study notes

One section per piece of the toolkit: why it works, the formulas you should be able to write
from memory, and interview-style questions with answers. Code lives in `src/optdesk/`; the
tests in `tests/` are the proof each claim holds.

- **M1 Pricing** (`pricing/`): sections 1–4
- **M2 Data** (`data/`): sections 5–8
- **M3 Surface** (`surface/`, plus de-Americanizing in `data/clean.py`): sections 9–12
- **M4a Hedging** (`sim/paths.py`, `sim/hedging.py`, `sim/plot.py`): sections 13–15
- **M4b Market making** (`surface/source.py`, `sim/market_maker.py`, `sim/mm_plot.py`):
  sections 16–18

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
   SPX options are European, cash-settled and 10× bigger notional; standard monthlies settle
   on the opening print (AM), weeklies (SPXW) on the close (PM). SPX also gets Section 1256
   tax treatment. European exercise makes SPX
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

---

## 7. Cleaning a chain (`clean.py`)

### Why it works
A raw chain is full of quotes whose mid isn't a real price. The cleaner drops them, values the
rest at mid, finds each expiry's forward from put-call parity, keeps one OTM option per strike,
and computes implied vol and log-moneyness.

**Filters (a quote goes if any is true):**
- *Zero bid:* nobody will pay for it, so the "mid" is half the ask, which isn't a price.
- *Crossed (ask < bid):* impossible live. The two sides came from different moments.
- *Wide (spread/mid > 50%):* the mid could be anywhere in a huge range. Relative, because a
  5¢ spread is tight on a $10 option and enormous on a 6¢ one. This also removes most
  penny wings.
- *Stale (no trade in 5 days):* illiquid contracts tend to have leftover quotes.

**Forward from parity.** Long call + short put at strike `K` pays `S_T − K`, exactly a forward.
So `C − P = DF·(F − K)`, giving `F = K + (C − P)/DF` at every strike. We take the median over
the 5 strikes with the smallest `|C − P|` (nearest the money: most liquid, tightest spreads,
smallest American early-exercise effect). The median means one bad pair can't move it. Only
`DF` comes from outside (the Treasury curve). The dividend yield and the stock's borrow cost are
already *in* the market's forward, so we never have to estimate them.

**Why OTM only.** By parity a call and a put at the same strike have the same implied vol,
but the ITM one is mostly intrinsic with a wider spread, so its vol is noisier. Puts below `F`
and calls above give one clean vol per strike. The VIX calculation uses the same convention.

**Pricing on the forward.** With `F` known, IV is solved with `S = F` and `q = r`, which makes
Black-Scholes' `S·e^((r−q)T)` equal `F`. No dividend assumption is needed.

**Time to expiry** is ACT/365 to 4pm New York time on expiry day, in actual seconds. A weekly
on expiry morning has hours left, not 0 or 1 day, and the time zone handles daylight saving.

### Key formulas
```
mid = (bid + ask)/2                      rel_spread = (ask − bid)/mid
T   = seconds(asof → expiry 16:00 NY) / (365·86400)
F   = median over 5 nearest strikes of  K + (C_mid − P_mid)·e^(rT)
OTM: put if K < F, call if K ≥ F         k = ln(K/F)
Implied dividend yield (if wanted):  q = r − ln(F/S)/T
```

### Interview questions
1. **How would you get the forward for SPY options without knowing the dividend?**
   Put-call parity: at each strike `F = K + (C − P)·e^(rT)`. Use near-the-money strikes and a
   robust average such as the median. It already includes dividends and borrow costs, and you
   can back out the implied dividend yield from it.
2. **Why does parity only hold approximately for SPY?**
   SPY options are American. The early-exercise right is worth something (more for deep ITM
   puts, and for calls just before an ex-dividend date), so `C − P` can drift from the European
   value. Near the money the effect is small, which is one reason to use those strikes.
3. **Why use OTM options to build a smile?**
   They're the liquid side with tighter spreads, and their price is all time value, so the vol
   is well determined. The ITM option at the same strike should give the same vol by parity, but
   it is noisier.
4. **What does a crossed or locked market mean?**
   Crossed (bid > ask) would be a free arbitrage, so in data it means stale or unsynchronized
   quotes. Locked (bid = ask) can happen briefly on fast moves. We drop crossed quotes and keep
   locked ones.
5. **If the wide-spread threshold is too strict or too loose, what happens?**
   Too strict and you lose the wings, so the surface fit in M3 has nothing to anchor the tails.
   Too loose and penny options with meaningless mids put noise in the wings. The 50% default is
   a judgment call, which is why it's a parameter and raw data is stored.

---

## 8. Snapshots (`store.py`)

### Why it works
Each day's raw chain is saved as one Parquet file, `data/snapshots/{TICKER}/{YYYY-MM-DD}.parquet`
(the New York trading date, so a late-evening fetch still files under that day). The rate
curve fetched at the same moment goes into the file's metadata as a small JSON note, so one
file holds everything needed to rebuild the day's cleaned chain and, later, its surface. The
tests check the round trip is exact and that cleaning a reloaded snapshot reproduces the
original result bit for bit.

**Why Parquet over CSV:** columnar and compressed (smaller, faster), and it keeps types:
floats are bit-identical and timestamps keep their time zone. CSV turns everything into text.

**Safety details:** saving refuses to overwrite an existing day unless asked
(`overwrite=True`), and it writes to a temporary file and renames it, so a crash can't leave a
half-written snapshot. Snapshots are gitignored, since they're regenerable data.

**Command line:** `python -m optdesk.data.fetch SPY` fetches the chain and curve, saves the
snapshot and prints forward, ATM vol and strike count per expiry.

### Key facts
```
Path:      data/snapshots/SPY/2026-10-16.parquet
Contents:  raw chain (RAW_COLUMNS) + metadata {"version": 1, "rate_curve": {...}}
Load:      chain, curve = load_snapshot("SPY", "2026-10-16");  clean_chain(chain, curve)
```

### Interview questions
1. **Why build your own history of snapshots instead of downloading history later?**
   Free sources only give *today's* chain; historical option quotes are expensive. Saving a
   snapshot each day builds a private dataset for backtesting surfaces and hedges.
2. **Why is a columnar format good for this data?**
   Analysis usually reads a few columns (strike, mid, iv) across many rows. Columnar files
   read only those columns and compress well, because each column has similar values.
3. **What does "reproducible" mean for a data pipeline here?**
   Same inputs give the same outputs: the raw quotes and the rate curve are stored together,
   so re-running the cleaner on an old snapshot gives the same forwards and vols, unless you
   deliberately change a cleaning rule.
4. **What time-of-day issues matter for snapshots?**
   Take them during market hours (after hours, bids vanish). Make sure the underlying price and
   option quotes are from the same moment, and avoid the open and close, when spreads are widest
   and quotes jump around.

---

# M3: Volatility surface

Notation as above, plus total implied variance `w = σ_impl²·T`, log-moneyness `k = ln(K/F)`,
and the early-exercise premium `EEP = American price − European price`.

## 9. SVI smile fitting (`svi.py`)

### Why it works
Implied vol isn't constant across strikes, so a desk needs a smooth curve through each
expiry's quotes. That curve is used to price strikes nobody quoted, compute Greeks consistently,
and spot quotes that are out of line. SVI (Gatheral, Merrill Lynch 2004) describes one expiry
with five numbers:

```
w(k) = a + b·( ρ(k − m) + √((k − m)² + σ²) )
```

- `a`: overall variance level. `b`: how steep the wings are. `ρ` (−1 to 1): tilt; negative
  for equities, because puts are richer. `m`: slides the smile left/right. `σ`: how rounded
  the bottom is (small `σ` = sharp V).
- **Why SVI rather than a polynomial or spline:** far from the money, SVI's total variance grows
  *linearly* (slopes `b(1−ρ)` left, `b(1+ρ)` right). Roger Lee proved no arbitrage-free smile
  can grow faster than that, with slope ≤ 2. A quadratic grows faster and must be wrong in
  the wings, and a spline says nothing past the last strike. SVI is also the long-maturity limit
  of the Heston model, and its arbitrage conditions are checkable in closed form.
- **Why fit in vol, weighted by vega:** errors read as vol points, the unit traders quote.
  Vega measures how much vol information a price carries. A 3-cent wing option has tiny vega,
  so half a cent of quote noise there is several vol points. Weighting by vega trusts the
  near-the-money quotes most but leaves the wings enough weight to anchor the tails. (Fitting
  prices would amount to weighting by vega², which nearly ignores the wings.) The test: one
  wing quote 3 vol points off moves the vega-weighted ATM vol less than a third as much as
  an equal-weighted fit.
- **Fitting without local minima ("quasi-explicit", Zeliade 2009):** fix `m` and `σ`, set
  `y = (k − m)/σ`, and SVI becomes `w = a + (bρσ)·y + (bσ)·√(y² + 1)`, which is *linear* in three
  unknowns. Ordinary least squares solves that exactly. So we scan a 9×9 grid of `(m, σ)`, solve
  the linear problem at each point, and polish the best 3 with a full nonlinear least-squares
  fit. A single-start 5-parameter optimizer often gets stuck. Constraints: box bounds
  (`b ≥ 0`, `|ρ| < 1`, `σ > 0`, `m` near the strikes) plus penalties keeping the minimum
  variance `a + bσ√(1−ρ²) ≥ 0` and Lee's wing bound `b(1+|ρ|) ≤ 2`.

**Identifiability (an important real-world lesson).** On exact data, all five parameters are
recovered to ~1e-13. With realistic noise, the *smile* is recovered within 0.08 vol points on
every expiry, but on the 1–2 week expiries the *parameters* wander (e.g. `ρ` off by 0.26, `a`
off by 40%). Over the narrow strike range of a weekly, quite different parameter sets draw
almost the same curve. From 1 month out the parameters are pinned down (`b` within ~4%, `ρ`
within ~0.06). So: trust the curve, not the individual parameters, for short expiries, and be
careful when trading on "ρ changed" for weeklies.

### Key formulas
```
w(k)   = a + b(ρ(k − m) + √((k − m)² + σ²))        σ_impl(k) = √(w/T)
w'(k)  = b(ρ + x/R)     w''(k) = bσ²/R³          x = k − m, R = √(x² + σ²)
min w  = a + bσ√(1 − ρ²)                          wing slopes: b(1 − ρ), b(1 + ρ)
Lee:   b(1 + |ρ|) ≤ 2
SSVI slice → raw: a = θ(1−ρ²)/2, b = θφ/2, m = −ρ/φ, σ = √(1−ρ²)/φ
```

### Interview questions
1. **What is SVI and why is it popular?**
   A 5-parameter formula for one expiry's total implied variance in log-moneyness. Its wings
   are linear in `k`, as Lee's moment formula requires, it fits equity smiles well, its
   parameters have meanings, and its no-arbitrage conditions can be checked in closed form.
2. **Why parametrize by total variance and log-moneyness instead of vol and strike?**
   Total variance `σ²T` adds up over time, so calendar arbitrage becomes "w must not decrease in
   T". Log-moneyness relative to the *forward* removes rates and dividends, so smiles from
   different expiries line up and the formulas are clean.
3. **What does ρ < 0 mean economically?**
   Downside puts trade at higher vol than upside calls: investors pay up for crash protection,
   and vol tends to rise when the market falls (spot-vol correlation is negative).
4. **How do you fit five parameters robustly?**
   Exploit the structure: for fixed `m` and `σ`, the other three enter linearly, so a cheap
   grid over two parameters with exact linear solves gives good starting points, then a
   bounded nonlinear least-squares fit finishes. Weight by vega so noisy wing quotes don't
   dominate.
5. **Your fitted ρ for the weekly jumped from −0.7 to −0.95 overnight. Is the market telling you
   something?**
   Maybe not. For short expiries, several parameter combinations draw nearly the same smile
   over the quoted strikes. Compare fitted *vols* (ATM level, 25-delta skew) day over day; those
   are well determined.

---

## 10. Static arbitrage checks (`arbitrage.py`)

### Why it works
A fitted surface is only usable if no combination of options built from it is a free lunch.
There are two kinds to check.

**Butterfly (within one expiry).** A butterfly spread (long `K−h` and `K+h` calls, short two
`K` calls) never pays less than zero, so it can't cost less than zero. In the limit that means
`∂²C/∂K² ≥ 0`. By Breeden-Litzenberger, that second derivative times `e^(rT)` *is* the
risk-neutral density of the stock at expiry. So "no butterfly arbitrage" means "the implied
probability density is never negative". Gatheral rewrote the density in terms of `w(k)`;
everything except one factor `g(k)` is positive, so the test is just `g(k) ≥ 0` for all `k`,
exact in closed form for SVI. We check on `k ∈ [−1.5, 1.5]`, far beyond the quotes, because a
fit can look fine where there's data and still imply negative probabilities in the wings.

**The Vogt example** (Axel Vogt, published by Gatheral & Jacquier 2014): a smooth, sensible-
looking SVI slice (`a=−0.041, b=0.1331, ρ=0.306, m=0.3586, σ=0.4153`, T=1) whose `g` dips to
−0.033 for `k` between about 0.64 and 1.26. You can't see it by looking at the smile, which is
why checks are code, not eyeballs. Our test checks the formula two independent ways:
Gatheral's `g` density must match `K·∂²C/∂K²` from finite differences of Black-Scholes prices.

**Calendar (across expiries).** At a fixed `k`, total variance must not decrease with expiry.
At fixed forward-moneyness, a call's normalized price increases with total variance, and a
longer option must be worth at least a shorter one. If two fitted slices cross, a calendar
spread at the crossing strikes is an arbitrage. Comparing at the same `k` (not the same dollar
strike) makes rates and dividends drop out. The checker reports *where* (the `k` ranges) and
*how badly* (the worst gap), because a desk needs to know which strikes to look at.

### Key formulas
```
g(k) = (1 − k·w'/(2w))² − (w'²/4)·(1/w + 1/4) + w''/2          need g ≥ 0
density in k:  p(k) = g(k)/√(2πw) · exp(−d₂²/2),   d₂ = −k/√w − √w/2
Breeden-Litzenberger:  density in K = e^(rT) ∂²C/∂K²     (density in k = K × that)
Calendar:  w(k, T₂) ≥ w(k, T₁)  for all k,  T₂ > T₁
Lee:       wing slope of w ≤ 2   ⇔   b(1 + |ρ|) ≤ 2 for SVI
```

### Interview questions
1. **What is butterfly arbitrage and how do you detect it in a vol surface?**
   A negative-cost butterfly, equivalent to a negative risk-neutral density. For an SVI slice,
   compute `g(k)` from `w, w', w''` on a fine grid and flag any `k` with `g < 0`.
2. **What is calendar arbitrage?**
   Total implied variance decreasing with maturity at fixed log-moneyness. Then you can buy the
   long-dated option and sell the short-dated one for a credit with no downside.
3. **Why compare expiries at the same log-moneyness rather than the same strike?**
   With dividends and rates, the forward differs by expiry. At a fixed `k = ln(K/F)` the
   comparison is clean. At a fixed dollar strike, total variance can legitimately decrease
   (e.g. across a big dividend).
4. **What is Lee's moment formula?**
   For large `|k|`, implied total variance grows at most linearly, with slope ≤ 2. Lee showed
   the wing slope is set by how many moments of the stock price are finite: fatter tails
   (fewer finite moments) mean steeper wings, and slope 2 is the extreme where barely the
   mean exists. Since every arbitrage-free model needs a finite mean, any slope above 2 is
   impossible.
5. **Your fit has a tiny negative `g` far out in the wing where there are no quotes. Do you care?**
   Yes, if anything is priced there: a variance swap or a deep-OTM structure integrates over
   the whole density. Fix it by constraining the fit (e.g. tighter wing bounds, or SSVI, which
   has arbitrage-free conditions built in) rather than ignoring it.

---

## 11. American exercise: how SPY affects implied vols (`data/clean.py`, `pricing/binomial.py`)

### Why it matters
SVI, put-call parity and the arbitrage checks are all European theory. SPY options are
*American*: they can be exercised any day. So every SPY price contains an early-exercise
premium (EEP). Treat it as European and you make two errors:

1. **Vols too high.** Inverting `mid` (which includes the EEP) with Black-Scholes gives a vol
   that has to explain extra value. The EEP is biggest for puts (deep ITM, getting `K` early
   to earn interest beats waiting) and grows with maturity and rates.
2. **Forward too low.** Parity `F = K + (C − P)/DF` assumes European prices. The American put
   is worth more than the European one, so `C − P` is too small and the forward comes out low.
   That shifts every `k = ln(K/F)` and the OTM/ITM split.

Measured on a SPY-like setup (spot 580, r = 4%, q = 1.3%, near-the-money OTM put):

| Expiry | Vol error if ignored | Forward error if ignored |
| --- | --- | --- |
| 1 week | < 0.01 vol pts | −0.3 bp |
| 1 month | ~0.04 vol pts | −1.7 bp |
| 3 months | ~0.14 vol pts | −6 bp |
| 6 months | ~0.29 vol pts | −15 bp |
| 1 year | ~0.6 vol pts | −38 bp |

Weeklies barely care. Long-dated SPY vols are off by more than half a point, which is bigger
than a typical bid-ask spread in vol terms, so it has to be fixed.

### How we handle it (`clean_chain(..., american=True)`)
**De-Americanize:** turn each American price into the European price with the same vol, then
use the European machinery unchanged.

1. For each quote, price its EEP on a CRR tree (American minus European, same tree, averaged
   over N and N+1 steps) at the vol the quote implies. Subtract it:
   `European-equivalent = mid − EEP`. The EEP depends on the vol, which depends on the EEP, so
   alternate twice; it settles immediately because the EEP barely moves with vol.
2. The forward and the EEP depend on each other too (the tree needs a dividend yield, which
   comes from the forward). Iterate 3 times: parity forward → implied yield
   `q = r − ln(F/S)/T` → de-Americanize the near-the-money call/put pairs → parity forward.
3. De-Americanize the OTM quotes with the final forward and invert with Black-Scholes as before.
   The `eep` column shows how much was removed from each quote.

**Result** on a synthetic American chain with noise: forwards within 0.6 bp and fitted smiles
within 0.08 vol points of the truth. Without de-Americanizing: forward 46 bp low and vols
0.57 points high at 14 months. Command-line tools default to American; pass `--european` for
index options such as SPX.

**Why this way:** inverting an American tree for every quote (a root search of full trees)
is slower. Subtracting the premium needs a few batched trees and reuses the fast
Black-Scholes solver. The result is a *European-equivalent* vol, the right input for SVI and
the arbitrage checks.

**What we measured about the tree:** averaging N and N+1 steps halves the premium's error, and
American-minus-European on the same tree beats tree-American minus exact Black-Scholes. At 200
steps the premium is good to about a cent on SPY (~0.005 vol points). We had first assumed the
tree error would cancel completely in the difference. Measuring showed it only partly does,
which is why there's averaging.

**Known approximations:** SPY pays *discrete* quarterly dividends while the tree uses a
continuous yield. Discrete dividends matter for calls just before ex-dividend dates, but those
are ITM calls, which we don't use. Using OTM quotes keeps us where the EEP is smallest anyway.

### Key formulas
```
EEP = American(σ) − European(σ)               (same CRR tree, avg of N and N+1 steps)
European-equivalent price = mid − EEP(σ_impl)  (iterate twice)
q_implied = r − ln(F/S)/T                       (feeds the tree)
American parity bounds:  S·e^(−qT) − K ≤ C − P ≤ S − K·e^(−rT)     (an inequality, not =)
```

### Interview questions
1. **SPY options are American. Does that matter for implied vol?**
   Yes, mostly for puts and longer maturities. The early-exercise premium makes Black-Scholes
   vols too high (about 0.6 vol points for a 1-year near-the-money put at 4% rates) and biases
   the parity forward low. For weeklies it's negligible.
2. **Why does the American feature bias the parity forward downward?**
   The American put is worth more than the European put (early exercise is valuable when rates
   are positive), so `C − P` is smaller than the European `DF·(F − K)`, and
   `F = K + (C − P)/DF` comes out too low.
3. **How would you get European-equivalent vols from American prices?**
   Compute the early-exercise premium with a tree at the option's vol, subtract it, and invert
   Black-Scholes; iterate because the premium depends on the vol. Or find the vol that makes an
   American tree match the price directly (what data vendors like OptionMetrics do, with
   discrete dividends).
4. **Why not just use SPX options?**
   On a desk you often would: SPX is European and cash-settled, so no de-Americanizing is
   needed. But single stocks and ETFs only have American options, so the technique is needed
   anyway, and SPY data is free on Yahoo.
5. **When is an American call on SPY exercised early?**
   Only deep ITM, just before an ex-dividend date, when the dividend you'd collect as a
   stockholder exceeds the call's remaining time value. That's why OTM calls carry essentially
   no premium. Our test checks that the removed EEP on OTM calls is under a cent.

---

## 12. Synthetic market and plots (`synthetic.py`, `plot.py`)

### Why it works
**A test needs an answer key.** A real chain can show the fit *runs*, not that it finds the
*right* smile. The synthetic SPY chain is generated from a known surface. It's priced as
American on a CRR tree, given penny-wide spreads, random noise in the mids, and quotes rounded
out to the cent, then pushed through the real cleaner and fitter. The tests compare what comes
out with what went in. With noise switched off, the fit returns the parameters to ~1e-13.

**The true surface is SSVI** (Gatheral & Jacquier 2014), a family of SVI slices tied together
by ATM total variance `θ(T)` and a skew function `φ(θ) = η/(θ^γ (1+θ)^(1−γ))`. SSVI has proven
no-arbitrage conditions: `η(1+|ρ|) ≤ 2` rules out butterflies and `0 < γ ≤ ½` with rising `θ`
rules out calendars. Our values (`ρ = −0.7, η = 1.1, γ = 0.45`) satisfy both, so the true
surface is a known-clean control for the arbitrage checks. Levels mimic SPY in a calm market:
ATM vol ~14% at 1 week rising to ~17%, and skew about −1 vol point per 1% of moneyness at 1
month.

**Plots.** Smiles are small multiples (one panel per expiry: market dots, SVI line, RMSE in
the title), plotted against `k`. The 3D surface uses *standardized moneyness*
`k/(σ_ATM√T)`: a 1-week smile spans a few percent of `k` while a 1-year smile spans 50%, so on
a `k` axis the short expiries shrink to slivers. In ATM standard deviations every expiry's
quotes cover roughly −4 to +2.5, so the surface is a full sheet and skews are comparable across
maturities (like quoting by delta). Nothing is drawn beyond the quoted strikes. SVI would
happily extrapolate, but a surface painted past the data invites over-reading.

**Command line:** `python -m optdesk.surface.plot` fits the synthetic market, prints the
parameter and arbitrage tables, and writes `data/plots/surface.html` and `smiles.html`. After
Monday's fetch: `python -m optdesk.surface.plot --ticker SPY --date 2026-10-19`.

### Key formulas
```
SSVI:  w(k, θ) = θ/2 · (1 + ρφk + √((φk + ρ)² + 1 − ρ²)),    θ = σ_ATM(T)²·T
φ(θ) = η / (θ^γ (1 + θ)^(1−γ))
No arbitrage (power-law φ):  η(1 + |ρ|) ≤ 2  and  0 < γ ≤ ½
Standardized moneyness:  z = k / √w(0)
```

### Interview questions
1. **How do you test a calibration routine?**
   Generate data from known parameters, fit, and check you get them back: first exact, then with
   realistic noise, where you check the quantities that are actually identified (the smile)
   and accept looser parameter recovery where the data can't pin them down.
2. **What's SSVI and why use it?**
   A surface-wide SVI with three global parameters plus the ATM variance curve, with simple
   proven conditions for no static arbitrage. It's a common way to get an arbitrage-free
   surface by construction, at the cost of flexibility per expiry.
3. **Why plot vol against standardized moneyness?**
   It puts different maturities on the same footing. A 5% OTM strike is far out for a weekly
   and near the money for a 1-year option. Dividing by `σ√T` measures distance in standard
   deviations, the way delta does.
4. **What does an upward-sloping ATM term structure mean? When does it invert?**
   Calm markets: near-term vol is low and longer-dated vol prices in the chance of future
   turbulence. In a selloff, near-term vol spikes above long-dated vol and the term structure
   inverts (backwardation), a classic stress signal.

---

# M4a: Hedging simulator

Notation as above, plus `Γ` gamma, `σ_i` the implied (hedge) vol, `σ_r` the realized vol,
`N` the number of hedges, and `v` the instantaneous variance in Heston.

Charts: `docs/images/` (regenerate with `python -m optdesk.sim.plot`).

## 13. Simulating prices: GBM and Heston (`sim/paths.py`)

### Why it works
**GBM** is the Black-Scholes world: `dS/S = μ dt + σ dW`. It has an exact solution,
`S_(t+dt) = S_t · exp((μ − σ²/2)dt + σ√dt·Z)`, so we simulate with that and there's no
discretization error at all. One big step to expiry gives the same distribution as a thousand
small ones (a test checks this). The grid only decides how often the hedger *looks*.

The `−σ²/2` is volatility drag: the average log return is lower than the average return,
because ups and downs of the same size don't cancel in price terms (+10% then −10% leaves you
at 99). It's what makes `E[S_T] = S_0·e^(μT)` come out right.

**Heston** lets variance move: `dv = κ(θ − v)dt + ξ√v dW_v`, with `corr(dW_S, dW_v) = ρ`.
Variance mean-reverts to `θ` at speed `κ`, has its own vol `ξ` ("vol of vol"), and for
equities `ρ < 0`: prices fall when vol spikes. That negative correlation is what produces
the equity skew we fitted in M3.

The catch: the true `v` never goes negative, but a simulated step can. **Full-truncation
Euler** lets the simulated `v` dip below zero but uses `v⁺ = max(v, 0)` wherever it enters the
drift or the noise. Lord, Koekkoek & van Dijk (2010) compared the simple fixes, and full
truncation had the least bias. Resetting to zero (absorption) or flipping the sign
(reflection) both push variance up. Exact schemes (Broadie-Kaya, Andersen's QE) are more
accurate per step but much more code, and buy little when we take 441 small steps anyway.
The price uses a log-Euler step, which keeps it positive.

**Feller condition** `2κθ ≥ ξ²`: when it holds, the true `v` never touches zero. Fitted equity
parameters often break it (our study uses `κ = 2, θ = 0.04, ξ = 0.5`: 0.16 < 0.25), which is
exactly why the scheme has to handle `v` hitting zero.

### Key formulas
```
GBM exact step:   S_(t+dt) = S_t · exp((μ − σ²/2)dt + σ√dt·Z)
GBM moments:      E[S_T] = S_0 e^(μT),   Var[S_T] = S_0² e^(2μT)(e^(σ²T) − 1)
Martingale:       with μ = r − q, E[S_t e^(−(r−q)t)] = S_0 for every t
Heston variance:  E[v_t] = θ + (v_0 − θ)e^(−κt)        (half-life ln2/κ)
Full truncation:  v_(t+dt) = v_t + κ(θ − v_t⁺)dt + ξ√(v_t⁺ dt)·Z_v
```

### Interview questions
1. **Why is the expected log return `μ − σ²/2` and not `μ`?**
   Jensen's inequality: log is concave, so `E[ln S] < ln E[S]`. A path that goes +10%, −10%
   ends below where it started. The drag grows with variance, which is why high-vol assets
   compound more slowly than their average return suggests.
2. **What's a martingale and why test for it?**
   A process whose expected future value is today's value. Under the risk-neutral measure the
   discounted price (with dividends reinvested) must be one, or you could make money on
   average by holding stock and shorting bonds. If a simulator fails this test, every price
   it produces is wrong.
3. **How do you simulate Heston, and what goes wrong?**
   Euler on `v` can go negative, where `√v` doesn't exist. Full truncation (use `max(v, 0)`)
   is simple and has the least bias of the quick fixes. Andersen's QE scheme is the industry
   standard when you need accuracy with big steps.
4. **What does each Heston parameter do to the smile?**
   `ρ` tilts it (negative → put skew); `ξ` curves it (more vol of vol → fatter wings); `κ`
   controls how fast the smile flattens with maturity; `θ` and `v_0` set the long- and
   short-dated ATM levels.

## 14. Delta hedging and the gamma P&L formula (`sim/hedging.py`)

### Why it works
Buy one call at 20% implied vol and sell `Δ` shares against it. Over a short step the option
changes by `Δ·dS + ½Γ·dS² + θ·dt` and the hedge by `−Δ·dS`, so the `dS` terms cancel and you're
left with gamma and theta. In Black-Scholes, theta is exactly what it costs to own the gamma:
`θ ≈ −½ΓS²σ_i²` (ignoring rates). So each step earns

```
½ΓS² · (dS²/S² − σ_i² dt)
```

`dS²/S²` is what the stock *actually* moved, squared: realized variance. If the stock
realizes more variance than the 20% you paid for, every step earns a little; if less, every
step loses a little. Add it up over the option's life and you get the **gamma P&L formula**:

```
P&L ≈ ∫ e^(−rt) · ½ Γ_t S_t² (σ_r² − σ_i²) dt
```

So buying an option and delta-hedging it is a bet that realized vol will beat implied vol,
not a bet on direction. That's what "trading vol" means on a desk.

Results (5,000 paths, hourly hedging, 20% implied; `gamma_pnl.png`):

| Realized vol | Simulated mean P&L | Gamma formula | BS(σ_r) − BS(σ_i) | Path correlation |
| --- | --- | --- | --- | --- |
| 10% | −1.972 ± 0.007 | −1.971 | −1.971 | 0.99 |
| 15% | −0.992 ± 0.005 | −0.993 | −0.987 | 0.95 |
| 20% | +0.004 ± 0.002 | 0 | 0 | |
| 25% | +0.993 ± 0.007 | +0.991 | +0.987 | 0.86 |
| 30% | +2.002 ± 0.013 | +1.992 | +1.975 | 0.94 |
| Heston (avg 19.9%) | −0.163 ± 0.014 | −0.164 | no closed form | 0.98 |

(± is one standard error.) Three things to take from it:

- **On average, the answer is a price difference.** With risk-neutral paths, the average P&L
  of hedging at `σ_i` when the world realizes `σ_r` is exactly `BS(σ_r) − BS(σ_i)`: you paid
  the 20% price for something worth the 25% price.
- **Path by path, it depends on *when* the variance came.** The formula weights each moment by
  `ΓS²` (dollar gamma), which is biggest near the strike and close to expiry. Realize your
  excess variance while the stock is far from the strike and you collect little. That's why
  the formula and the P&L vary from path to path, and why variance swaps exist: they pay
  realized minus implied variance with no path weighting. The leftover scatter around the
  formula is discrete-hedging noise (section 15), which is bigger when the stock moves more.
- **The drift doesn't matter.** A hedged option has no `dS` exposure, so the stock's expected
  return drops out. A test hedges paths drifting at 15% a year and still gets zero mean P&L.
  This is the Black-Scholes insight in simulation form.

**Why hedge at the implied vol, not the realized?** You don't know realized vol until it's
over. Hedging at realized would give a smooth P&L equal to `BS(σ_r) − BS(σ_i)` but needs
foresight. Hedging at implied makes the P&L path-dependent but has the same average.

**Stochastic vol (Heston).** The formula still holds path by path if you plug in the
variance each path actually had (`v_t`) instead of a constant `σ_r²`: correlation 0.98. The
surprise is the mean. Average variance is exactly 0.04 (20% vol), yet the hedged call *lost*
$0.16. The reason is Jensen's inequality again. An ATM option's value is roughly linear in
*vol*, not variance, and when variance is random, the average of `√(path variance)` is below
`√(average variance)`. In a check with 20,000 paths, the average per-path realized vol was
19.1%, and `vega × (19.1% − 20%) = −$0.17` matched the simulated loss (−$0.16). Setting
`ρ = 0` gave almost the same loss (−$0.17), so it's the randomness of vol, not its
correlation with spot, that costs money here. In Heston terms the 3-month ATM call is worth
less than Black-Scholes at `√θ`, i.e. Heston's ATM implied vol sits below `√θ`.

### Key formulas
```
Hedged step:      dΠ = ½ Γ S² (dS²/S² − σ_i² dt)       (long option, short Δ shares)
Gamma P&L:        P&L = ∫ e^(−rt) ½ Γ S² (σ_r² − σ_i²) dt
Mean (μ = r − q): E[P&L] = BS(σ_r) − BS(σ_i) ≈ vega · (σ_r − σ_i)
Break-even move:  daily |move| ≈ σ_i / √252   (≈ 1.26% a day at 20% vol)
```

### Interview questions
1. **You buy an option at 20% implied and realized vol comes in at 25%. Did you make money?**
   If you delta-hedged: on average yes, roughly `vega × 5 points` (+$0.99 here). But it's
   path-dependent; if the big moves came far from the strike, or early when gamma was small,
   you can still lose. Unhedged, it depends on direction, which is a different trade.
2. **Explain the gamma-theta trade-off.**
   Long gamma earns `½ΓS²·(move/S)²` on every move, up or down; theta is the rent, `½ΓS²σ_i²`
   per unit time. You break even if the stock moves about one implied standard deviation per
   period: `σ_i/√252 ≈ 1.26%` a day at 20% vol.
3. **What's the difference between a delta-hedged option and a variance swap?**
   Both pay realized minus implied variance, but the hedged option weights each moment by
   dollar gamma `ΓS²`, so it's path-dependent and concentrated near the strike. A variance
   swap weights every moment equally. You can build one from a strip of options across all
   strikes weighted `1/K²`.
4. **Your hedged long call lost money even though average variance equalled implied. How?**
   Discrete hedging noise (std ~$0.17 hourly here), transaction costs, jumps through the
   strike, or vol convexity: with random vol, the average realized *vol* is below the square
   root of the average variance, and an ATM option pays on vol (the Heston result above).
5. **Who is short gamma and why would they be?**
   Option sellers (market makers net short customer demand, covered-call writers). They
   collect theta and lose when realized beats implied. It pays on average because implied vol
   usually trades above subsequent realized vol (the variance risk premium), but it loses big
   in selloffs.

## 15. Hedging frequency and transaction costs (`sim/hedging.py`, `sim/plot.py`)

### Why it works
Black-Scholes assumes you hedge continuously. Hedging `N` times instead leaves, on each
interval, an unhedged `½ΓS²(ΔS²/S² − σ²Δt)`. It's zero on average, but each one is
nonzero. These errors are roughly independent from interval to interval, so they add like
noise: `N` errors of size `~1/N` give a total spread of `~1/√N`. Derman & Kamal (1999) worked
out the constant for an ATM option:

```
std(P&L) ≈ √(π/4) · vega · σ / √N
```

Our 5,000-path study (3-month ATM call, `S = K = 100`, `r = 4%`, `q = 1.3%`, premium $4.31,
realized = implied = 20%, same paths at every frequency; `hedge_frequency.png`,
`hedge_pnl_histograms.png`):

| Frequency | Hedges N | Mean P&L | Std P&L | Derman-Kamal | Mean / SE |
| --- | --- | --- | --- | --- | --- |
| monthly | 3 | −0.015 | 1.822 | 2.021 | −0.6 |
| biweekly | 7 | −0.013 | 1.229 | 1.323 | −0.7 |
| weekly | 13 | −0.004 | 0.902 | 0.971 | −0.3 |
| every 2 days | 32 | −0.000 | 0.587 | 0.619 | −0.0 |
| daily | 63 | −0.004 | 0.431 | 0.441 | −0.6 |
| every 3 hours | 147 | −0.003 | 0.286 | 0.289 | −0.8 |
| hourly | 441 | −0.002 | 0.166 | 0.167 | −0.7 |

Log-log slope of std vs `N`: −0.48 (theory −0.5). Derman-Kamal is an asymptotic formula, so
it's 10% high at 3 hedges and within 1% at 441. Every mean is within one standard error of
zero.

**Transaction costs push the other way** (`hedge_costs.png`). Each rehedge trades `|ΔΔ|`
shares and pays the half-spread on them. Each delta change is `~1/√N` and there are `N` of
them, so turnover and cost grow like `√N` while risk falls like `1/√N`. At 5bp per share
traded: expected cost $0.05 monthly → $0.15 daily → $0.36 hourly, while risk falls from $1.82
to $0.20. (Cost grows a bit slower than √N here because the opening hedge, a fixed cost, is a
large share of the total at low frequencies.) Somewhere in between is the frequency where one
more hedge removes less risk than it costs; where exactly depends on how much the desk will
pay to cut risk. In practice desks hedge on a delta band (rehedge when delta drifts by more
than some amount), not on a clock. Leland (1985) folds costs into an adjusted vol: someone
short an option should charge a *higher* vol to cover hedging costs.

**Design choice: same paths at every frequency.** Simulating new paths per frequency would
mix hedging differences with simulation luck. Hedging one set of hourly paths at every
frequency (common random numbers) keeps the comparison clean; the monthly hedger just looks
at the price less often.

### Key formulas
```
Derman-Kamal:     std(P&L) ≈ √(π/4) · vega · σ / √N     (ATM, realized = implied)
Cost of a trade:  cost · |shares traded| · S             (cost = half-spread, e.g. 5bp)
Scaling:          hedging error ~ 1/√N,  total cost ~ √N
```

### Interview questions
1. **If you hedge 4× as often, how much does your P&L uncertainty fall?**
   By half: it scales like `1/√N`. Here 3 → 13 hedges cuts the std from $1.82 to $0.90, and
   63 → 441 (7×) cuts it from $0.43 to $0.17 (÷2.6 ≈ √7).
2. **Why not just hedge continuously?**
   Costs. Each rehedge pays the bid-ask spread, and total cost grows like `√N`. At some
   frequency the next hedge costs more than the risk it removes is worth. Markets also close
   overnight, so there's always gap risk you can't hedge.
3. **Is discrete hedging error biased?**
   To first order no: it's zero on average, just noisy. Every mean in the table is within one
   SE of zero. Costs, by contrast, are a pure drag.
4. **How would you hedge in practice?**
   Rehedge when delta leaves a band rather than at fixed times. Make the band narrower where
   gamma is big (near the strike, near expiry), because delta moves faster there, and wider
   when trading is expensive.
5. **Short-dated ATM option: what's the bigger risk, hedging error or vol?**
   Near expiry, gamma explodes near the strike, so hedging error and pin risk dominate. For
   longer-dated options vega dominates, and the question is realized vs implied vol.

---

# M4b: Market-making simulator

Notation as above, plus `γ` the dealer's risk aversion (per dollar; not the Greek, which is
`Γ`), `σ_v` the vol of vol (how much implied vol moves, in vol per √year), `H` the risk
horizon, `V` the book's vega, `ν` the vega one fill adds, `A` and `k` the customer arrival rate
and how fast it decays with distance from theo.

Charts: `docs/images/mm_*.png` (regenerate with `python -m optdesk.sim.mm_plot`).

## 16. Where theo comes from: swappable surfaces (`surface/source.py`)

### Why it works
A market maker quotes around **theoretical value** (theo): what the desk thinks the option is
worth. For a listed option that's Black-Scholes at the vol the *surface* gives that strike and
expiry. The simulator needs only one thing from a surface, `implied_vol(k, T)`, so anything
with that method plugs in (a Python `Protocol`: duck typing with a name):

- `synthetic_surface()`: the synthetic SSVI surface from M3, analytic in `T`. The default
  until we have live data.
- `fitted_surface(fit_chain(clean_chain(snapshot)))`: SVI fits of a real snapshot. Between
  fitted expiries it interpolates **total variance linearly in T at fixed k**. If every slice
  has `w` rising with `T`, so does every interpolated point, so the interpolation can't
  create calendar arbitrage. Outside the fitted range each strike's vol is held constant.
- `FlatSurface(0.2)`: the Black-Scholes world, for tests.

`python -m optdesk.sim.mm_plot --ticker SPY --date 2026-10-19` reruns the whole study on
Monday's fitted surface without touching the simulator.

### Key formulas
```
Theo:                 V_i = BS(S, K_i, T_i, r, σ_surface(k_i, T_i) + x)
Log-moneyness:        k = ln(K/F),  F = S e^((r−q)T)
Linear-in-w interp:   w(k,T) = w_j(k) + (T − T_j)/(T_(j+1) − T_j) · (w_(j+1)(k) − w_j(k))
```

### Interview questions
1. **Why interpolate total variance rather than vol between expiries?**
   No calendar arbitrage means total variance `σ²T` can't fall as `T` grows. A weighted average
   of two non-decreasing functions is non-decreasing, so linear-in-`w` interpolation keeps that
   property for free. Linear in vol doesn't, and it makes forward vols jump at each expiry.
2. **What are "sticky strike" and "sticky moneyness"?**
   Two assumptions about how the smile moves when spot moves. Sticky strike: the vol at each
   strike stays put. Sticky moneyness (or sticky delta): the smile slides with spot, so the vol
   at a fixed `K/F` stays put. Our simulated surface is sticky-moneyness. Equity index reality
   is somewhere in between, with vols also rising as spot falls.
3. **Why write the simulator against an interface rather than the synthetic surface?**
   We don't have live fits yet, and when we do the simulator shouldn't change. It also makes
   tests sharp: on a flat surface, both delta methods must agree exactly.

## 17. Quoting: Avellaneda-Stoikov in vol space (`sim/market_maker.py`)

### Why it works
A market maker earns the bid-ask spread but has to hold whatever customers trade: its
**inventory**. Avellaneda & Stoikov (2008) solve this for a dealer with exponential utility.
Fills arrive as Poisson processes with rate `A e^(−kδ)` for a quote `δ` away from fair value:
quote tighter and you trade more but earn less per trade. The optimal policy has two pieces:

1. **Reservation price** `s − q γ σ² H`: your own value of the asset, given what you hold. If
   you're long, it's *below* fair value, because one more unit adds risk you don't want.
   Both quotes are centred on it, so a long book shows a lower bid (buys less) and a lower ask
   (sells more): inventory mean-reverts.
2. **Spread** `γσ²H + (2/γ) ln(1 + γ/k)` around it. The log term is the monopolist's markup
   (it tends to `2/k` as `γ → 0`); the first term is extra pay for the risk each fill adds.

**The options twist.** An options desk hedges delta, so spot moves aren't its inventory risk.
What's left is **vega**: if implied vols move by `dσ`, the book makes `V · dσ`. So we treat the
vol level as the risky asset. Every option loads on that one vol factor, so in the
multi-asset A-S reservation price `s_i − γH·Cov(dV_i, dBook)`, the covariance is
`vega_i · V · σ_v²`. Divide by `vega_i` to quote in vol, and every option's reservation vol is
**theo minus the same shift**:

```
reservation vol = theo vol − γ σ_v² H V
```

That's how vol traders talk: "I'm long vega, lower my vols." The spread in vol for a fill
adding vega `ν` is `γνσ_v²H + (2/(γν))·ln(1 + γν/k)`.

**The numbers here.** `γ = 5×10⁻⁵`, `σ_v` = 10 vol points a year, `H` = 1 day, `k = 200` (fills
drop by a factor of `e` for every half vol point further from theo), `A = 1,000` a year per
side per contract, 10-lot trades. For a 10-lot of the 2-month ATM (~$950 of vega per vol
point) the half-spread is **0.50 vol points**, and a book long $10k per vol point shades every
quote down by **0.2 vol points**. At the $25k vega limit the shift equals the half-spread, so
the ask sits exactly on theo (`mm_frontier.png`, left).

**Design choices.**
- *Constant risk horizon `H` (1 day)* instead of A-S's time to the close. A-S assume you end
  the session flat, so their quotes collapse to symmetric at the close. A desk carrying a book
  for weeks doesn't. Guéant, Lehalle & Fernandez-Tapia (2013) show the optimal quotes settle to
  stationary values far from the end, and a fixed `H` is the practical stand-in.
- *The same A-S spread with and without skew,* so the comparison isolates the skew.
- *Delta hedged every 15 minutes* with Black-Scholes delta at each option's smile vol
  (sticky-strike delta), paying 1 bp per share.

### Key formulas
```
Fill rate:            λ(δ) = A e^(−kδ),  fills in dt ~ Poisson(λ dt)
A-S reservation:      r = s − q γ σ² H
A-S spread:           δ_a + δ_b = γ σ² H + (2/γ) ln(1 + γ/k)       → 2/k as γ → 0
In vol (options):     shift = γ σ_v² H V;   half-spread = ½γνσ_v²H + ln(1 + γν/k)/(γν)
Spread rises with γ   whenever σ² H k² ≥ 1 (here σ_v² H k² = 1.6)
```

### Interview questions
1. **You're a market maker and you're long a lot of vega. What do you do with your quotes?**
   Lower both your bid and offer vol. You become the best offer (customers buy from you) and a
   worse bid (they stop selling to you), so the position works itself off without you crossing
   the spread. That's the A-S reservation price. If it's urgent, hedge directly: sell vega in
   another strike or expiry, or a variance swap.
2. **Why does the spread widen when volatility is high?**
   Each fill adds inventory whose value moves by `σ` per √time, so more vol means each fill
   carries more risk and needs more compensation (`γσ²H`). Wider spreads in high vol are
   rational, not greed. A test checks the spread rises with `σ_v`.
3. **What does the `(2/γ) ln(1 + γ/k)` term mean?**
   With no risk aversion it's `2/k`: a monopolist facing exponential demand maximizes
   `δ · e^(−kδ)` at `δ = 1/k`. It's the spread you'd quote purely to make money on flow.
   Risk aversion trims it slightly, because a risk-averse dealer values trading more.
4. **Why quote in vol rather than in price?**
   Vol is comparable across strikes and expiries; a $0.10 edge means very different things on
   a $0.50 wing put and a $20 ATM call. The skew also becomes one number for the whole book,
   because all options load on the same vol factor.
5. **What does A-S leave out?**
   Adverse selection (informed customers who know where vol is going), jumps, competition from
   other dealers, a smile that changes shape (our shift is parallel), and gamma. Our desk skews
   on vega only; the gamma it's left with is handled by the gamma limit (section 18).

## 18. P&L attribution, risk limits and what skewing changes (`sim/market_maker.py`, `sim/mm_plot.py`)

### Why it works
**Three buckets that add up exactly.** Each step, the desk's mark-to-market P&L splits into:

- **Spread capture**: at each fill, the traded price minus theo, from the desk's side. This is
  the desk's edge. With skew, some trades happen on the "wrong" side of theo (a long desk's
  shaded ask can sit below theo), so skewing gives some edge back.
- **Hedging**: option value change from spot and time at *unchanged* vols (gamma, theta, the
  smile rolling with spot), plus the share hedge, dividends, interest and hedge costs. When
  realized vol tracks implied, this is noise around minus the hedge costs (a test checks it on
  a flat surface).
- **Inventory**: option value change from the vol move, `≈ V·dσ`. This is the cost of carrying
  inventory, and it's what skewing targets.

The order matters: we revalue at the new spot and time with the *old* vol, then apply the vol
move. Tests check that the buckets add up to the mark-to-market P&L (cash + options at theo +
shares) within 1e-9 of its size, and that with no vol moves the inventory bucket is exactly 0.

**Risk limits** (vega $25k per vol point; gamma $1m of delta per 1% move) work like a
pre-trade check: a quote is shown only if one more fill there would keep the book inside both
limits. Near the long-vega limit the desk stops bidding but keeps offering. It never trades
*out* of a breach, so the market can still push exposure past a limit (the gamma of a
near-dated option near its strike), but only briefly.

**The study.** 12 SPY options (2-, 3- and 6-month; strikes at 90/95/100/105% of the forward),
one month of 15-minute steps, 1,000 paths. Both desks see the *same* market paths and the
*same* customer random numbers, so every difference comes from the quoting rule
(`mm_attribution.png`, `mm_inventory.png`):

| $ per month | No skew | A-S skew |
| --- | --- | --- |
| Spread capture (mean) | 313,491 | 314,186 |
| Hedging (mean) | −9,885 | −9,319 |
| Inventory (mean) | +669 | +766 |
| **Total (mean ± SE)** | **304,275 ± 1,124** | **305,633 ± 796** |
| Spread capture std | 19,739 | 20,154 |
| Hedging std | 28,150 | 21,269 |
| Inventory std | 35,371 | 13,510 |
| **Total std** | **35,544** | **25,162** |
| Customer fills | 720 | 737 |
| Std of ending vega ($/vol pt) | 13,474 | 3,917 |
| Average \|vega\| ($/vol pt) | 10,367 | 2,994 |
| Average \|gamma\| ($ delta per 1%) | 294,394 | 220,666 |
| Steps with a quote pulled by a limit | 4.5% | 0.6% |

What changed when we turned skewing on:

- **Inventory risk fell by almost two thirds** (std $35k → $14k) because vega inventory stays
  near zero: average |vega| fell from $10.4k to $3.0k per vol point. Without skew, inventory
  is a random walk that only the limit stops; with skew, it mean-reverts.
- **Total risk fell by 29%** ($35.5k → $25.2k), and mean/std rose from 9 to 12.
- **The average didn't fall:** $305.6k vs $304.3k is within one standard error. Skewing gives
  up some edge per trade, but here it gets *more* fills (737 vs 720). Fill rates are convex in
  the quote (`e^(−k(δ−s)) + e^(−k(δ+s)) ≥ 2e^(−kδ)`), so the side shaded towards theo gains
  more fills than the far side loses.
- **The limits almost stopped binding** (4.5% → 0.6% of steps).
- **Hedging risk fell too** ($28k → $21k): a smaller book has less gamma to hedge.

**The risk-aversion sweep** (`mm_frontier.png`, right) shows the trade-off. A little skew
(`γ = 1–2×10⁻⁵`) slightly *raises* mean P&L while cutting risk. Past `γ ≈ 5×10⁻⁵` risk stops
falling (it levels off near $24k) and the desk starts paying for it: spread capture falls to
$301k at `γ = 2×10⁻⁴`.

**Why does risk level off at ~$24k?** Two risks are left that a parallel-vega skew can't see:

1. **Gamma and term structure.** A book can be vega-flat and still short gamma: short
   front-month, long back-month. In `mm_path.png` the skewed desk sits around −$400k of gamma
   on a nearly flat vega book. The gamma limit is the only control on that.
2. **Strike-level carry.** Each option earns `½ΓS²(σ_realized² − σ_its_implied²)`. Wing puts
   are marked at 20%+ vol while spot realizes ~15%, so a vega-flat book that's long wing puts
   and short ATM calls bleeds, and one short wing puts earns. That's smile risk, not
   vol-level risk.

**Which delta?** `delta_method="sticky_moneyness"` hedges with the surface-consistent delta
(bump spot, reprice on the moving smile). Without skew it halves the *hedging* bucket's noise
($36k → $20k in a 200-path check) but *raises* total risk ($42k → $53k). The reason: vols
here fall when spot rises (correlation −0.7), and the plain sticky-strike delta happens to lean
against that, a crude minimum-variance hedge (Hull & White 2017). The best delta depends on how
vols really move with spot, which is an empirical question for the live data.

### Key formulas
```
Spread capture:   Σ fills · size · mult · (ask − theo)  or  (theo − bid)
Hedging:          Σ pos · [V(S₁, t₁, x₀) − V(S₀, t₀, x₀)] + shares·ΔS + carry − costs
Inventory:        Σ pos · [V(S₁, t₁, x₁) − V(S₁, t₁, x₀)]  ≈ V · Δσ
Check:            spread + hedging + inventory = cash + Σ pos·theo + shares·S
Pre-trade limit:  show the bid on i only if V + ν_i ≤ V_max and Γ + Γ_i ≤ Γ_max
```

### Interview questions
1. **How does a market maker make money, and how does it lose it?**
   It earns the bid-ask spread on customer flow (about $435 per trade here) and loses on the
   inventory it's forced to hold when prices (for options: vols) move against it. Good market
   making is spread minus inventory losses; the A-S skew is about keeping the second small.
2. **Your desk made $300k this month. How do you know whether it was skill or luck?**
   Attribute it. Spread capture is repeatable edge. Inventory P&L is a vol bet that happened
   to pay (or not). Hedging P&L is noise plus costs. A month that's mostly inventory P&L wasn't
   market making, it was a position. Here spread capture is 103% of P&L; hedging costs eat 3%.
3. **Skewing cut risk by 29% and didn't lower the mean. Isn't that free money?**
   Nearly, at this level: you're refusing to let inventory pile up, so you don't carry risk you
   aren't paid for. But the sweep shows it isn't unlimited. Past a point the desk gives up
   more edge per trade than it saves in risk, and mean P&L falls. Choosing `γ` is choosing a
   point on that frontier, usually from the desk's risk budget.
4. **You're vega-flat. Are you flat?**
   No. You can be short front-month gamma against long back-month vega (term-structure risk),
   long downside vega against short upside vega (skew risk), or exposed to vols moving
   differently by strike. Vega is one number summarizing a whole surface of risks. Our skewed
   desk shows it: vega near zero, gamma −$400k.
5. **Why use a pre-trade check instead of a hard cap on the book?**
   Because the trade is the thing you control. Checking "would this fill breach?" stops you
   walking into a breach. Exposure can still drift past the limit as markets move the Greeks,
   which is why real desks also have a process to trade *out* of breaches (here the skew does
   that gently).
