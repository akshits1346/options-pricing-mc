"""
Tests for binomial_tree.py.

  - Convergence (statistical/numerical, not exact): error vs.
    Black-Scholes should shrink as n_steps grows. Checked as a monotonic
    trend across several step counts, not just one pass/fail threshold.
  - American call == European call for a non-dividend-paying underlying
    (exact, up to tree discretization -- there's never a reason to
    exercise a call early with no dividend, so the tree should never
    find the early-exercise branch worth taking).
  - American put >= European put, always (the early-exercise option can
    only add value).
  - WITH a dividend yield q > 0, that call symmetry breaks: American
    call > European call becomes possible (and, deep ITM with a high
    enough q, actual) -- giving up a big enough future dividend stream
    can make early exercise worth more than holding, which is exactly
    the classical justification for why dividend-adjusted trees matter.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.black_scholes import black_scholes_price
from src.binomial_tree import binomial_tree_price

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    bs_call = black_scholes_price(S, K, r, sigma, T, "call")

    # --- convergence: error should shrink as steps increase ---
    errors = []
    for n in [10, 50, 200, 1000]:
        bt_price = binomial_tree_price(S, K, r, sigma, T, n, "call", "european")
        errors.append(abs(bt_price - bs_call))

    check(errors[0] > errors[1] > errors[2] > errors[3],
          f"error shrinks monotonically as steps increase: {[f'{e:.4f}' for e in errors]}")
    check(errors[-1] < 0.01, f"error at n=1000 is small, got {errors[-1]:.4f}")

    # --- American call == European call (no dividend -> no early exercise premium) ---
    euro_call = binomial_tree_price(S, K, r, sigma, T, 500, "call", "european")
    amer_call = binomial_tree_price(S, K, r, sigma, T, 500, "call", "american")
    check(abs(euro_call - amer_call) < 1e-6,
          f"American call == European call with no dividend: {amer_call:.6f} vs {euro_call:.6f}")

    # --- American put >= European put, always ---
    euro_put = binomial_tree_price(S, K, r, sigma, T, 500, "put", "european")
    amer_put = binomial_tree_price(S, K, r, sigma, T, 500, "put", "american")
    check(amer_put >= euro_put,
          f"American put ({amer_put:.4f}) >= European put ({euro_put:.4f}), early-exercise premium exists")
    check(amer_put > euro_put,
          "American put is STRICTLY greater (a real, nonzero early-exercise premium, not just >=)")

    # --- input validation ---
    try:
        binomial_tree_price(S, K, r, sigma, T, 100, "invalid")
        check(False, "should reject invalid option_type")
    except ValueError:
        check(True, "correctly rejects invalid option_type")

    try:
        binomial_tree_price(S, K, r, sigma, T, 100, "call", "invalid")
        check(False, "should reject invalid exercise type")
    except ValueError:
        check(True, "correctly rejects invalid exercise type")

    # --- dividend yield q: q=0.0 (the default) must reproduce every
    # value above EXACTLY -- backward compatible, not a separate path ---
    euro_call_q0 = binomial_tree_price(S, K, r, sigma, T, 500, "call", "european", q=0.0)
    check(euro_call_q0 == euro_call, f"q=0.0 reproduces the no-dividend European call price exactly: {euro_call_q0} == {euro_call}")

    # --- WITH a dividend, European tree price still converges to the
    # dividend-adjusted Black-Scholes closed form ---
    q = 0.06
    bs_call_div = black_scholes_price(S, K, r, sigma, T, "call", q=q)
    euro_call_div = binomial_tree_price(S, K, r, sigma, T, 1000, "call", "european", q=q)
    check(abs(euro_call_div - bs_call_div) < 0.01,
          f"European call WITH dividend converges to dividend-adjusted Black-Scholes: "
          f"tree={euro_call_div:.4f}, BS={bs_call_div:.4f}")

    # --- the actual point of this section: WITH a high enough dividend
    # yield, American call now has a REAL, nonzero early-exercise
    # premium over European -- the exact opposite of the q=0 case above,
    # and the textbook reason dividend-adjusted trees matter at all
    # (giving up a big enough future dividend stream can make exercising
    # a call early worth more than holding it) ---
    S_itm = 120.0  # deep ITM: early exercise is where the premium shows up most clearly
    euro_call_itm_div = binomial_tree_price(S_itm, K, r, sigma, T, 500, "call", "european", q=q)
    amer_call_itm_div = binomial_tree_price(S_itm, K, r, sigma, T, 500, "call", "american", q=q)
    check(amer_call_itm_div > euro_call_itm_div,
          f"WITH a dividend, American call ({amer_call_itm_div:.4f}) is STRICTLY greater than "
          f"European ({euro_call_itm_div:.4f}) -- a real early-exercise premium that q=0 never shows")

    print()
    if failures == 0:
        print("All binomial tree checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
