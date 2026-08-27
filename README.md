# options-pricing (Monte Carlo extension)

Monte Carlo pricing for path-dependent options that don't have simple
closed forms: Asian options (arithmetic-average payoff) and a
down-and-out barrier call, with antithetic variates for variance
reduction.

## Architecture

```
gbm_paths.py       -- shared GBM path simulator, with optional
                       antithetic sampling (each draw Z paired with -Z)

asian_option.py    -- Monte Carlo arithmetic-average Asian call, plus
                       an exact closed-form GEOMETRIC-average Asian
                       call (self-derived, see the module docstring)
                       used purely to validate the simulation machinery

barrier_option.py  -- Monte Carlo down-and-out call, plus a standard
                       closed-form vanilla Black-Scholes call used as
                       a convergence reference
```

## Build & test

```bash
pip3 install numpy scipy
python3 tests/test_asian_option.py     # 4 checks
python3 tests/test_barrier_option.py   # 3 checks
```

## Validation strategy

Arithmetic-average Asian options have no simple closed form -- that's
exactly why Monte Carlo is used for them. So instead of checking the
arithmetic price against nothing, this project validates the
*machinery* two ways:

1. **Exact, non-statistical checks** that must hold with certainty,
   regardless of random seed:
   - Arithmetic average >= geometric average, always, by AM-GM --
     verified on the *same* simulated paths for both payoffs.
   - Down-and-out price <= vanilla call price, always -- knocking out
     only ever removes payoff, never adds it -- also verified on
     shared paths.
2. **Statistical convergence checks**, stated in standard errors (not
   an arbitrary tolerance):
   - Geometric-average Asian MC price converges to an exact,
     self-derived closed form (within ~1.6 SE at 100k paths).
   - Down-and-out price with an unreachable barrier converges to the
     vanilla Black-Scholes price (within ~1.7 SE).

## A real bug, found and fixed

The first version of the antithetic variance-reduction code showed
**~0% variance reduction** -- no improvement at all over plain Monte
Carlo, despite antithetic variates being a well-established technique.
Rather than accept that number, I checked the actual correlation
between antithetic path pairs directly: it was **-0.50**, strongly
negative, exactly as the underlying theory predicts. That meant the
"0% reduction" result was a bug in *my* code, not a real finding about
the method.

The bug: `std_error` was computed by treating every simulated path as
an independent sample -- correct for plain Monte Carlo, but wrong for
antithetic sampling, where half the paths are deliberately correlated
with the other half. The fix: form pair-averages (`(Y_i + Y_i') / 2`
for each antithetic pair) *first*, then compute standard error across
those pair-averages, not across all raw paths.

After the fix: **51.4% variance reduction** on the Asian call, which is
what the -0.50 correlation actually implies. This is exactly the kind
of thing worth catching by checking a result against known theory
rather than trusting a plausible-looking number.

## Honest limitations

- The Asian and barrier pricers use *discrete* monitoring (n equally
  spaced points), matching the closed-form derivation exactly -- results
  would differ from continuous-monitoring closed forms found in most
  textbooks, which is why the closed form here was re-derived for the
  discrete case rather than borrowed directly from a reference.
- No variance reduction technique beyond antithetic variates
  (e.g. control variates, using the geometric Asian's exact price as a
  control for the arithmetic one, would likely help further).
- Barrier monitoring only checks the n discretized points, not
  continuous-time barrier breaches between them -- a real path could
  cross the barrier between monitored points without being detected
  here, which is a standard (and usually small, for reasonably fine
  discretization) discretization bias in Monte Carlo barrier pricing.

## What I'd build next

- Control variates using the geometric Asian closed form to further
  reduce arithmetic Asian variance
- Brownian bridge correction for the barrier monitoring gap
- Compare antithetic variance reduction across moneyness (ATM vs. deep
  ITM/OTM) to see how much it depends on payoff linearity
