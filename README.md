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
```

## Build & test

```bash
pip3 install numpy scipy

python3 tests/test_black_scholes.py     # 8 checks
python3 tests/test_binomial_tree.py     # 7 checks
python3 tests/test_asian_option.py      # 4 checks
python3 tests/test_barrier_option.py    # 3 checks
python3 tests/test_implied_vol.py       # 5 checks
```

27 checks total, all passing.

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
  converges to a self-derived exact closed form within ~1.6 SE.
- **Barrier option**: down-and-out price is guaranteed `<=` vanilla
  price on the same paths (exact); converges to vanilla Black-Scholes
  as the barrier becomes unreachable (~1.7 SE).
- **Implied volatility**: round-trip test -- price at a known sigma,
  recover that exact sigma from the price, to near machine precision.

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
- No control variates (e.g. using the geometric Asian's exact price to
  reduce the arithmetic Asian's variance further).
- The binomial tree doesn't model dividends; American-call-equals-
  European-call would no longer hold with a dividend-paying underlying.

## What I'd build next

- Control variates for the arithmetic Asian option
- Dividend-adjusted binomial tree and Black-Scholes
- A full implied volatility surface (across strikes and maturities)
  rather than single-point solves
