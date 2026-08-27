"""
Tests for asian_option.py.

Two flavors of check here, deliberately:
  - EXACT (non-statistical): arithmetic average >= geometric average
    pointwise, always, by AM-GM. Computed on the SAME simulated paths,
    arithmetic price must be >= geometric price with certainty -- no
    tolerance needed, no random seed can make this fail.
  - STATISTICAL (convergence): geometric MC price should land within a
    few standard errors of the exact closed-form price. This can't be
    an exact check (Monte Carlo has irreducible sampling noise), so the
    tolerance is stated in standard errors, not an arbitrary epsilon.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.gbm_paths import simulate_gbm_paths
from src.asian_option import geometric_asian_call_closed_form, monte_carlo_asian_call

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S0, K, r, sigma, T, n_steps = 100.0, 100.0, 0.05, 0.2, 1.0, 50

    # --- statistical: geometric MC converges to the exact closed form ---
    closed_form = geometric_asian_call_closed_form(S0, K, r, sigma, T, n_steps)
    mc_result = monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=100000,
                                       average_type="geometric", seed=1)
    n_std_errors = abs(mc_result.price - closed_form) / mc_result.std_error
    check(n_std_errors < 3.0,
          f"geometric MC price ({mc_result.price:.4f}) within 3 SE of closed form "
          f"({closed_form:.4f}), got {n_std_errors:.2f} SE")

    # --- exact: arithmetic >= geometric on the SAME paths, always (AM-GM) ---
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths=10000, seed=2)
    arithmetic = monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=10000,
                                        average_type="arithmetic", paths=paths)
    geometric = monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=10000,
                                       average_type="geometric", paths=paths)
    check(arithmetic.price >= geometric.price,
          f"arithmetic price ({arithmetic.price:.4f}) >= geometric price ({geometric.price:.4f}) "
          "on the same paths, guaranteed by AM-GM -- not a statistical claim")

    # --- antithetic variates: real, substantial variance reduction ---
    # (this specific 51% figure came from FIXING a bug: the first version
    # of _standard_error() treated all antithetic paths as independent
    # samples, which showed ~0% reduction despite the underlying paths
    # having a real -0.50 correlation. The fix -- averaging each
    # antithetic pair before computing standard error -- is what actually
    # surfaces the benefit. See _standard_error()'s docstring.)
    n_paths = 20000
    plain = monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=n_paths,
                                    average_type="arithmetic", antithetic=False, seed=5)
    antithetic_result = monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=n_paths,
                                               average_type="arithmetic", antithetic=True, seed=5)
    variance_reduction_pct = (1 - (antithetic_result.std_error / plain.std_error) ** 2) * 100
    check(variance_reduction_pct > 30.0,
          f"antithetic variates give substantial variance reduction, got {variance_reduction_pct:.1f}%")

    # --- input validation ---
    try:
        monte_carlo_asian_call(S0, K, r, sigma, T, n_steps, n_paths=100, average_type="bogus")
        check(False, "should reject an invalid average_type")
    except ValueError:
        check(True, "correctly rejects an invalid average_type")

    print()
    if failures == 0:
        print("All Asian option checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
