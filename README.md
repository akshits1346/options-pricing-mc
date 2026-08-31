# options-pricing-mc

A complete options pricing and risk engine: Black-Scholes closed-form
pricing with analytic Greeks, a Cox-Ross-Rubinstein binomial tree
(European and American exercise), plain and path-dependent (Asian,
barrier) Monte Carlo pricing with antithetic variates, and an implied
volatility solver (Newton-Raphson with a bisection fallback).

## Architecture

```
black_scholes.py       -- closed-form European pricing + analytic Greeks
                           (delta, gamma, vega, theta, rho)

binomial_tree.py        -- CRR binomial tree, European and American exercise

monte_carlo_vanilla.py  -- plain vanilla Monte Carlo (antithetic variates supported)

gbm_paths.py            -- shared GBM path simulator used by every MC pricer

asian_option.py         -- arithmetic-average Asian call via MC, validated
                           against a self-derived exact geometric-average
                           closed form

barrier_option.py       -- down-and-out call via MC, validated against
                           vanilla Black-Scholes as the barrier -> unreachable

implied_vol.py          -- Newton-Raphson implied vol solver with
                           bisection fallback for low-vega regimes

vol_surface.py           -- fits implied vol pointwise across a grid of
                           (strike, maturity) quotes, then checks the
                           resulting surface for calendar-spread and
                           butterfly (strike-convexity) static
                           no-arbitrage violations
```

## Build & test

```bash
pip3 install numpy scipy

python3 tests/test_black_scholes.py     # 14 checks (incl. dividend yield q)
python3 tests/test_binomial_tree.py     # 10 checks (incl. dividend yield q)
python3 tests/test_asian_option.py      # 8 checks (incl. control variates)
python3 tests/test_barrier_option.py    # 3 checks
python3 tests/test_implied_vol.py       # 5 checks
python3 tests/test_vol_surface.py       # 10 checks
```

50 checks total, all passing.

## Validation strategy

Every pricer here is checked against either an EXACT mathematical
identity (holds with certainty, any seed) or a KNOWN closed form
(statistical convergence, stated in standard errors or discretization
error, not an arbitrary tolerance):

- **Black-Scholes**: put-call parity (`C - P = S - Ke^-rT`) holds to
  machine precision; every Greek cross-checked against its own
  finite-difference definition.
- **Binomial tree**: European price converges to Black-Scholes as steps
  increase (error: 0.20 -> 0.001 from 10 to 2000 steps); American call
  equals European call exactly with no dividend (no early-exercise
  incentive); American put is strictly greater than European put (a
  real, nonzero early-exercise premium).
- **Vanilla Monte Carlo**: converges to Black-Scholes within ~1.6
  standard errors at 200k paths.
- **Asian option**: arithmetic MC price is guaranteed `>=` geometric MC
  price on the *same* simulated paths (AM-GM, exact); geometric MC
  converges to a self-derived exact closed form within ~1.6 SE; the
  control-variate estimator (below) agrees with the plain estimator
  within 3 SE of the far noisier plain price.
- **Barrier option**: down-and-out price is guaranteed `<=` vanilla
  price on the same paths (exact); converges to vanilla Black-Scholes
  as the barrier becomes unreachable (~1.7 SE).
- **Implied volatility**: round-trip test -- price at a known sigma,
  recover that exact sigma from the price, to near machine precision.
- **Vol surface**: round-trip test on a flat, known-sigma surface
  (guaranteed arbitrage-free -- it's just Black-Scholes with one sigma)
  for "does it correctly find nothing wrong," PLUS deliberately
  hand-constructed calendar and butterfly violations (built directly
  via `VolSurface.from_grid()`, bypassing the solver) for "does it
  actually catch a real violation" -- see below.

## Control variates: a 99.9% variance reduction, and why it's so much bigger than antithetic's

`monte_carlo_asian_call_control_variate` uses the geometric Asian's
EXACT closed form as a control variate for the arithmetic Asian: on
the SAME simulated paths, `Y_cv_i = Y_i - beta*(X_i - E[X])`, where
`Y_i`/`X_i` are the arithmetic/geometric discounted payoffs on path
`i`, `E[X]` is the geometric price's known exact value (not estimated),
and `beta` is fit from the sample to minimize variance. `mean(Y_cv)` is
still an unbiased estimator of the arithmetic price for any beta, since
`(X_i - E[X])` has mean zero by construction.

At 100,000 paths, same seed, same everything else: **99.92% variance
reduction** (a ~36x smaller standard error) versus antithetic variates'
~51% found earlier. The gap makes sense once you look at WHY each
technique works: antithetic variates exploit negative correlation
between a path and its mirror image (a real but limited effect,
correlation ~ -0.50); control variates here exploit correlation between
TWO DIFFERENT AVERAGES OF THE SAME PATH -- the arithmetic and geometric
averages of the identical simulated price sequence, which are far more
tightly correlated with each other than any path is with its
antithetic pair, because they're not even trying to be different random
variables, just different (and very similar) functions of the same one.
The two techniques are independent and combine: control variate +
antithetic together beats antithetic alone by a further wide margin
(see `tests/test_asian_option.py`).

## A volatility surface, not just a point solve

Solving implied vol at one (strike, maturity) point at a time, as
`implied_vol.py` does, doesn't guarantee those points are *consistent*
with each other. `vol_surface.py` fits a full grid and checks it for the
two classic static no-arbitrage conditions:

- **No calendar arbitrage**: total implied variance `sigma^2 * T` must
  be non-decreasing in `T` at fixed strike -- otherwise a calendar
  spread (long the longer-dated option, short the shorter, same
  strike) is risk-free money.
- **No butterfly arbitrage**: price must be convex in strike at fixed
  maturity (equivalently, the Breeden-Litzenberger risk-neutral density
  must stay non-negative) -- otherwise a butterfly spread (long the
  wings, short 2x the body) has negative cost and a non-negative
  payoff.

The detector is tested on both sides: a flat, genuinely arbitrage-free
surface produces zero flagged violations, and two deliberately broken
surfaces (one with total variance decreasing across maturities, one
with an artificially spiked mid-strike vol that breaks strike-convexity)
are each caught with exactly one flagged violation, verified to be the
correct one by construction. A checker that never fires is
indistinguishable from a checker that isn't wired up correctly without
this half of the test.

## Dividends

`black_scholes_price` / `black_scholes_greeks` / `binomial_tree_price`
all take an optional continuous dividend yield `q` (0.0 by default,
reproducing the original no-dividend formulas EXACTLY -- verified in
each test file, not just assumed). With `q=0`, American call value
always equals European call value (no dividend, no reason to exercise
early). The entire point of adding `q` is that this stops being true
once `q > 0`: giving up a big enough future dividend stream can make
exercising a call early worth more than holding it. Concretely, at
`S=120, K=100, r=0.05, sigma=0.20, T=1.0, q=0.06`:

| | European | American |
|---|---|---|
| price | 20.12 | **21.05** |

A real, nonzero ~$0.94 early-exercise premium that the same tree, same
strikes, same everything, shows as EXACTLY zero at `q=0` -- the
dividend is what creates it, not a modeling artifact. The European
tree price also still converges to the dividend-adjusted Black-Scholes
closed form (`q` passed to both), same convergence check as the
no-dividend case.

## Two real bugs, found and fixed

**1. Antithetic variance reduction initially showed ~0% improvement.**
Antithetic variates are well-established theory, so a ~0% result
demanded investigation rather than acceptance. Checking the actual
correlation between antithetic path pairs directly showed -0.50 --
strongly negative, exactly as expected -- meaning the "0% reduction"
number was a bug in the standard error calculation, not a real finding.
The bug: treating every path as an independent sample when computing
standard error, which is correct for plain Monte Carlo but wrong for
antithetic sampling (half the paths are deliberately correlated with
the other half). Fix: form pair-averages first, then compute standard
error across those. After the fix: 51.4% variance reduction, matching
what the -0.50 correlation implies.

**2. The implied volatility solver falsely reported convergence with a
wrong answer in deep-OTM/near-expiry cases.** The original convergence
check was `abs(model_price - target_price) < tol`, on raw price. That
silently fails whenever prices are astronomically small for a wide
range of sigma (e.g. a 50%-OTM option with ~2.5 days to expiry: sigma
0.2 gives price 2e-92, sigma 0.3 gives price 2e-42 -- both "zero" to 8
decimal places, but 0.1 apart in actual sigma). The fix: converge on
the Newton STEP SIZE (`price_diff / vega`) instead of the raw price
difference, which is scale-invariant and doesn't share this failure
mode. Separately, some sufficiently extreme cases are genuinely
ill-posed (the price surface really is numerically flat across a wide
sigma range there) -- no solver fixes that, and the tests use a
realistic near-expiry case instead of that degenerate extreme.

## Honest limitations

- Barrier monitoring checks only the discretized points, not
  continuous-time breaches between them (standard, usually small,
  discretization bias in MC barrier pricing).
- The control variate here uses a fixed beta fit once per call (not
  re-estimated adaptively), and only covers the arithmetic Asian --
  the barrier and vanilla MC pricers have no control variate at all
  yet (a similar geometric-average or Black-Scholes-based control
  could plausibly help the barrier pricer too).
- The dividend yield `q` is a flat CONTINUOUS yield, not a schedule of
  discrete cash dividends on specific dates -- the standard simplifying
  assumption for both Black-Scholes and a CRR tree, but a real
  dividend-paying stock pays discrete amounts on discrete dates, which
  a continuous yield only approximates.
- The vol surface's arbitrage checks are static (calendar and
  butterfly conditions on the fitted grid), not a full arbitrage-free
  parametric fit (e.g. SVI) -- it flags violations in a surface built
  from independently-solved points, it doesn't yet produce a smoothed,
  guaranteed-arbitrage-free surface from sparse/noisy market quotes.

## What I'd build next

- A control variate for the barrier and/or vanilla MC pricers
  (Black-Scholes itself is a natural control for vanilla MC)
- Discrete (not just continuous-yield) dividend dates for the binomial
  tree -- the harder, more realistic version of the dividend work above
- Fit a parametric arbitrage-free surface (e.g. SVI) through noisy
  synthetic market quotes, rather than only checking a grid of
  independently-solved points for violations after the fact
