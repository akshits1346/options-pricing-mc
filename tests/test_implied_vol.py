"""
Tests for implied_vol.py.

VALIDATION STRATEGY: round-trip. Price at a KNOWN sigma via
Black-Scholes, then solve implied vol back out from that exact price --
recovered sigma should match to near machine precision. This tests the
solver against the exact function it inverts, not against a reference
value from a table.

A REAL BUG, FOUND AND FIXED (see implied_vol.py's docstring for the
full story): the first version of this solver checked Newton
convergence via abs(model_price - target_price) < tol. That works fine
normally, but fails silently for deep OTM / near-expiry options, where
EVERY price is astronomically close to zero regardless of sigma (e.g.
2e-92 vs 2e-42 -- both "zero" to 8 decimal places, but corresponding to
sigmas 0.1 apart). The fix: converge on the sigma STEP SIZE
(price_diff / vega), which is scale-invariant and doesn't have this
failure mode.

A SEPARATE, HONEST LIMITATION (not a bug, a real numerical fact):
sufficiently extreme cases (e.g. 50% out-of-the-money with ~2.5 days to
expiry) have a price surface that is numerically indistinguishable from
zero across a WIDE range of sigma (0.1 through ~0.6 all give prices
below 1e-15) -- no solver, Newton or bisection, can recover a
meaningful implied vol there, because the inverse problem itself is
ill-posed at that point, not because of an implementation flaw. The
test below uses a realistic near-expiry hard case (short-dated,
near-the-money) instead of that degenerate extreme, since that's the
regime a solver is actually expected to handle robustly.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.black_scholes import black_scholes_price
from src.implied_vol import implied_volatility

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- normal case: round-trip recovery ---
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    true_sigma = 0.25
    price = black_scholes_price(S, K, r, true_sigma, T, "call")
    result = implied_volatility(price, S, K, r, T, "call")
    check(abs(result.implied_vol - true_sigma) < 1e-6,
          f"recovers known sigma to high precision: true={true_sigma}, got={result.implied_vol:.8f}")
    check(result.method == "newton", f"normal case converges via Newton, got method={result.method}")

    # --- realistic near-expiry hard case (short-dated, near-the-money) ---
    S2, K2, T2 = 100.0, 101.0, 0.02
    true_sigma2 = 0.35
    price2 = black_scholes_price(S2, K2, r, true_sigma2, T2, "call")
    result2 = implied_volatility(price2, S2, K2, r, T2, "call")
    check(abs(result2.implied_vol - true_sigma2) < 1e-6,
          f"recovers sigma correctly in a realistic short-dated case: true={true_sigma2}, got={result2.implied_vol:.8f}")

    # --- put-side round-trip too ---
    true_sigma3 = 0.4
    price3 = black_scholes_price(S, K, r, true_sigma3, T, "put")
    result3 = implied_volatility(price3, S, K, r, T, "put")
    check(abs(result3.implied_vol - true_sigma3) < 1e-6,
          f"recovers sigma correctly for a put: true={true_sigma3}, got={result3.implied_vol:.8f}")

    # --- regression check for the fixed bug: the degenerate deep-OTM case must NOT
    # falsely report Newton convergence with a wrong answer anymore. It's fine if it
    # raises (the price is outside a solvable bracket) or falls back to bisection --
    # what it must NOT do is confidently return an answer that's wildly wrong. ---
    S4, K4, T4 = 100.0, 150.0, 0.01
    true_sigma4 = 0.3
    price4 = black_scholes_price(S4, K4, r, true_sigma4, T4, "call")
    try:
        result4 = implied_volatility(price4, S4, K4, r, T4, "call")
        # if it returns anything, it must not be a confident Newton "success" -- the
        # old bug specifically was Newton falsely declaring victory in 1 iteration
        check(not (result4.method == "newton" and result4.iterations <= 2),
              f"degenerate case doesn't falsely converge via Newton in 1-2 iterations "
              f"(got method={result4.method}, iterations={result4.iterations})")
    except ValueError:
        check(True, "degenerate deep-OTM/near-expiry case correctly raises rather than guessing")

    print()
    if failures == 0:
        print("All implied volatility checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
