"""
Tests for black_scholes.py.

Two validation strategies:
  - Put-call parity (C - P = S - K*e^-rT) is an EXACT mathematical
    identity, not an approximation -- any implementation bug in either
    the call or put formula shows up as a nonzero difference here, to
    machine precision.
  - Every Greek is cross-checked against a finite-difference
    approximation of its own definition (e.g. delta = dV/dS, checked
    via (price(S+eps)-price(S-eps))/(2*eps)). This validates the
    calculus was implemented correctly independent of any reference
    values from a textbook or another library.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from src.black_scholes import black_scholes_price, black_scholes_greeks

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    eps = 0.01

    # --- exact: put-call parity ---
    call_price = black_scholes_price(S, K, r, sigma, T, "call")
    put_price = black_scholes_price(S, K, r, sigma, T, "put")
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    check(abs(lhs - rhs) < 1e-9, f"put-call parity holds exactly: C-P={lhs:.6f}, S-Ke^-rT={rhs:.6f}")

    # --- Greeks vs finite difference (call) ---
    analytic = black_scholes_greeks(S, K, r, sigma, T, "call")

    delta_fd = (black_scholes_price(S + eps, K, r, sigma, T, "call")
                - black_scholes_price(S - eps, K, r, sigma, T, "call")) / (2 * eps)
    check(abs(analytic.delta - delta_fd) < 1e-4, f"delta matches finite difference: {analytic.delta:.6f} vs {delta_fd:.6f}")

    gamma_fd = (black_scholes_price(S + eps, K, r, sigma, T, "call")
                - 2 * black_scholes_price(S, K, r, sigma, T, "call")
                + black_scholes_price(S - eps, K, r, sigma, T, "call")) / (eps ** 2)
    check(abs(analytic.gamma - gamma_fd) < 1e-3, f"gamma matches finite difference: {analytic.gamma:.6f} vs {gamma_fd:.6f}")

    vega_fd = (black_scholes_price(S, K, r, sigma + eps, T, "call")
               - black_scholes_price(S, K, r, sigma - eps, T, "call")) / (2 * eps)
    check(abs(analytic.vega - vega_fd) < 1e-2, f"vega matches finite difference: {analytic.vega:.6f} vs {vega_fd:.6f}")

    theta_fd = -(black_scholes_price(S, K, r, sigma, T + eps, "call")
                 - black_scholes_price(S, K, r, sigma, T - eps, "call")) / (2 * eps)
    check(abs(analytic.theta - theta_fd) < 1e-2, f"theta matches finite difference: {analytic.theta:.6f} vs {theta_fd:.6f}")

    rho_fd = (black_scholes_price(S, K, r + eps, sigma, T, "call")
              - black_scholes_price(S, K, r - eps, sigma, T, "call")) / (2 * eps)
    check(abs(analytic.rho - rho_fd) < 1e-1, f"rho matches finite difference: {analytic.rho:.6f} vs {rho_fd:.6f}")

    # --- input validation ---
    try:
        black_scholes_price(S, K, r, sigma, T, "invalid")
        check(False, "should reject invalid option_type")
    except ValueError:
        check(True, "correctly rejects invalid option_type")

    try:
        black_scholes_price(S, K, r, sigma, -1.0, "call")
        check(False, "should reject non-positive T")
    except ValueError:
        check(True, "correctly rejects non-positive T")

    # --- dividend yield q: q=0.0 (the default) must reproduce every
    # value above EXACTLY -- q is additive, not a replacement path ---
    call_q0 = black_scholes_price(S, K, r, sigma, T, "call", q=0.0)
    check(call_q0 == call_price, f"q=0.0 reproduces the no-dividend call price exactly: {call_q0} == {call_price}")
    greeks_q0 = black_scholes_greeks(S, K, r, sigma, T, "call", q=0.0)
    check(greeks_q0.delta == analytic.delta, "q=0.0 reproduces the no-dividend delta exactly")

    # --- dividend-adjusted put-call parity: C - P = S*e^-qT - K*e^-rT,
    # an exact identity just like the q=0 case above ---
    q = 0.03
    call_div = black_scholes_price(S, K, r, sigma, T, "call", q=q)
    put_div = black_scholes_price(S, K, r, sigma, T, "put", q=q)
    lhs_div = call_div - put_div
    rhs_div = S * math.exp(-q * T) - K * math.exp(-r * T)
    check(abs(lhs_div - rhs_div) < 1e-9,
          f"dividend-adjusted put-call parity holds exactly: C-P={lhs_div:.6f}, S*e^-qT-Ke^-rT={rhs_div:.6f}")
    check(call_div < call_price, f"a dividend REDUCES call value vs no dividend: {call_div:.4f} < {call_price:.4f}")
    check(put_div > put_price, f"a dividend INCREASES put value vs no dividend: {put_div:.4f} > {put_price:.4f}")

    # --- dividend-adjusted delta vs finite difference, same check as
    # above but with q>0 -- validates the div_discount factor in the
    # Greeks, not just the price ---
    analytic_div = black_scholes_greeks(S, K, r, sigma, T, "call", q=q)
    delta_div_fd = (black_scholes_price(S + eps, K, r, sigma, T, "call", q=q)
                     - black_scholes_price(S - eps, K, r, sigma, T, "call", q=q)) / (2 * eps)
    check(abs(analytic_div.delta - delta_div_fd) < 1e-4,
          f"dividend-adjusted delta matches finite difference: {analytic_div.delta:.6f} vs {delta_div_fd:.6f}")

    print()
    if failures == 0:
        print("All Black-Scholes checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
