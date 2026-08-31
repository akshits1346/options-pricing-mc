"""
Tests for svi.py.

VALIDATION STRATEGY, three parts:

1. PARAMETER RECOVERY: generate synthetic (strike, implied_vol) quotes
   from a KNOWN SVI parameter set, add small noise (simulating noisy
   market quotes), fit SVI back out, and check the fitted parameters
   land close to the true ones. This is the strongest form of check
   used throughout this project (the same round-trip philosophy as
   implied_vol.py's test) -- it validates the fit against the exact
   generating process, not an arbitrary reference value.

2. NO FALSE POSITIVE: the fit recovered from realistic, well-behaved
   true parameters should pass check_svi_butterfly_arbitrage() cleanly
   (zero violations) when evaluated on a fine strike grid.

3. NOT A NO-OP: deliberately extreme SVI parameters (found by directly
   trying them and observing the ALREADY-VALIDATED grid checker fire --
   not by citing an unverified analytic arbitrage-free formula from
   memory, see svi.py's module docstring for why) DO get flagged. A
   checker that never fires on anything is indistinguishable from one
   that isn't wired up, without this.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.svi import SVIParams, svi_implied_vol, fit_svi, check_svi_butterfly_arbitrage

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S, r, T = 100.0, 0.03, 0.5
    true_params = SVIParams(a=0.015, b=0.12, rho=-0.35, m=0.0, sigma=0.12)

    forward = S * np.exp(r * T)
    k_obs = np.linspace(-0.4, 0.4, 15)
    strikes = forward * np.exp(k_obs)
    true_iv = svi_implied_vol(k_obs, T, true_params)

    rng = np.random.RandomState(0)
    noisy_iv = true_iv + rng.normal(0, 0.002, size=len(true_iv))  # ~20bp iv noise, realistic quote noise

    fitted = fit_svi(strikes, noisy_iv, T, S, r)

    check(abs(fitted.a - true_params.a) < 0.01, f"fitted a close to true: {fitted.a:.4f} vs {true_params.a}")
    check(abs(fitted.b - true_params.b) < 0.03, f"fitted b close to true: {fitted.b:.4f} vs {true_params.b}")
    check(abs(fitted.rho - true_params.rho) < 0.1, f"fitted rho close to true: {fitted.rho:.4f} vs {true_params.rho}")
    check(abs(fitted.sigma - true_params.sigma) < 0.03, f"fitted sigma close to true: {fitted.sigma:.4f} vs {true_params.sigma}")

    # the fitted curve itself should reproduce the noisy quotes closely
    # (a fit that recovers "close" parameters but fits the data badly
    # would be a red flag on its own)
    fitted_iv_at_obs = svi_implied_vol(k_obs, T, fitted)
    max_fit_error = float(np.max(np.abs(fitted_iv_at_obs - noisy_iv)))
    check(max_fit_error < 0.01, f"fitted curve tracks the noisy input quotes closely, max error {max_fit_error:.4f}")

    # --- no false positive: a realistic, well-behaved fit is clean ---
    clean_violations = check_svi_butterfly_arbitrage(fitted, T, S, r)
    check(len(clean_violations) == 0,
          f"the realistic fitted SVI curve has zero butterfly-arbitrage violations, got {len(clean_violations)}")

    # --- not a no-op: deliberately extreme parameters (huge slope b,
    # near-degenerate rho, tiny curvature sigma) DO get flagged ---
    bad_params = SVIParams(a=0.01, b=5.0, rho=-0.99, m=0.0, sigma=0.01)
    bad_violations = check_svi_butterfly_arbitrage(bad_params, T, S, r)
    check(len(bad_violations) > 0,
          f"deliberately extreme SVI parameters ARE flagged as butterfly-arbitrage violations, "
          f"got {len(bad_violations)}")

    # --- input validation ---
    try:
        fit_svi([100.0, 105.0], [0.2, 0.21], T, S, r)
        check(False, "should reject fewer than 5 quotes (can't fit 5 parameters)")
    except ValueError:
        check(True, "correctly rejects fewer than 5 quotes")

    try:
        fit_svi([100.0, 105.0, 110.0, 95.0, 90.0], [0.2, 0.21, 0.22], T, S, r)
        check(False, "should reject mismatched strikes/implied_vols lengths")
    except ValueError:
        check(True, "correctly rejects mismatched strikes/implied_vols lengths")

    print()
    if failures == 0:
        print("All SVI checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
