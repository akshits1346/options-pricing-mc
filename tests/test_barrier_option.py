"""
Tests for barrier_option.py -- same exact/statistical split as the
Asian option tests.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gbm_paths import simulate_gbm_paths
from src.barrier_option import black_scholes_call, monte_carlo_down_and_out_call

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S0, K, r, sigma, T, n_steps = 100.0, 100.0, 0.05, 0.2, 1.0, 50

    # --- statistical: barrier far below any plausible path -> converges to vanilla BS ---
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    low_barrier = monte_carlo_down_and_out_call(S0, K, r, sigma, T, barrier=1.0,
                                                n_steps=n_steps, n_paths=100000, seed=1)
    n_std_errors = abs(low_barrier.price - bs_price) / low_barrier.std_error
    check(n_std_errors < 3.0,
          f"down-and-out with unreachable barrier ({low_barrier.price:.4f}) within 3 SE of "
          f"vanilla BS ({bs_price:.4f}), got {n_std_errors:.2f} SE")

    # --- exact: down-and-out <= vanilla on the SAME paths, always ---
    # (barrier knock-out only ever zeroes payoffs the vanilla call would
    # have paid -- it can never add value -- so this holds with certainty,
    # not just in expectation)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths=10000, seed=2)
    barrier_result = monte_carlo_down_and_out_call(S0, K, r, sigma, T, barrier=85.0,
                                                    n_steps=n_steps, n_paths=10000, paths=paths)
    vanilla_via_unreachable_barrier = monte_carlo_down_and_out_call(
        S0, K, r, sigma, T, barrier=0.01, n_steps=n_steps, n_paths=10000, paths=paths
    )
    check(barrier_result.price <= vanilla_via_unreachable_barrier.price,
          f"down-and-out price ({barrier_result.price:.4f}) <= vanilla price "
          f"({vanilla_via_unreachable_barrier.price:.4f}) on the same paths, guaranteed, not statistical")

    # --- input validation ---
    try:
        monte_carlo_down_and_out_call(S0, K, r, sigma, T, barrier=150.0, n_steps=n_steps, n_paths=100)
        check(False, "should reject a barrier at or above spot for a down-and-out option")
    except ValueError:
        check(True, "correctly rejects a barrier at or above spot")

    print()
    if failures == 0:
        print("All barrier option checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
