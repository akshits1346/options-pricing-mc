"""
Tests for vol_surface.py.

VALIDATION STRATEGY, in two independent parts:

1. fit() correctness via round-trip: generate quotes from a KNOWN flat
   vol surface (same sigma at every strike and maturity), fit implied
   vols back out, and check they recover the known sigma. A flat
   surface is mathematically guaranteed arbitrage-free (it's just
   Black-Scholes with one sigma), so this also doubles as a "no false
   positives" check for both arbitrage detectors.

2. Arbitrage detection correctness via DELIBERATELY CONSTRUCTED bad
   surfaces, built directly through VolSurface.from_grid() (bypassing
   the solver entirely) so each detector is tested in isolation against
   a shape with a KNOWN, hand-verified violation:
       - a calendar violation: total variance decreasing from a shorter
         to a longer maturity at the same strike
       - a butterfly violation: an artificially spiked vol at a middle
         strike, which prices that option far enough above the chord
         connecting its neighbors to break strike-convexity
   Testing the detector on cases it MUST flag is just as important as
   testing it doesn't false-positive on a clean surface -- a checker
   that never fires is indistinguishable from a checker that isn't
   wired up correctly, without this.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.black_scholes import black_scholes_price
from src.vol_surface import VolSurface, VolQuote

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    S, r = 100.0, 0.05

    # --- part 1: flat, genuinely arbitrage-free surface ---
    flat_sigma = 0.22
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.25, 0.5, 1.0, 2.0]

    quotes = []
    for T in maturities:
        for K in strikes:
            price = black_scholes_price(S, K, r, flat_sigma, T, "call")
            quotes.append(VolQuote(K=K, T=T, price=price, option_type="call"))

    flat_surface = VolSurface(S, r).fit(quotes)

    max_err = max(abs(flat_surface.iv(K, T) - flat_sigma) for T in maturities for K in strikes)
    check(max_err < 1e-6, f"fit() recovers the known flat sigma at every grid point (max err {max_err:.2e})")

    calendar_violations = flat_surface.check_calendar_arbitrage()
    butterfly_violations = flat_surface.check_butterfly_arbitrage()
    check(len(calendar_violations) == 0, f"flat surface has zero calendar-arbitrage violations, got {len(calendar_violations)}")
    check(len(butterfly_violations) == 0, f"flat surface has zero butterfly-arbitrage violations, got {len(butterfly_violations)}")
    check(flat_surface.is_arbitrage_free(), "is_arbitrage_free() is True for a flat, genuinely clean surface")

    # --- part 2a: deliberately construct a calendar-arbitrage violation ---
    # Same strike, two maturities, with total variance DECREASING from the
    # shorter to the longer maturity: w(K, 0.5) = 0.30^2 * 0.5 = 0.045,
    # w(K, 1.0) = 0.10^2 * 1.0 = 0.010 -- 0.010 < 0.045, a violation by
    # construction.
    bad_calendar = VolSurface.from_grid(S, r, {
        (100.0, 0.5): 0.30,
        (100.0, 1.0): 0.10,
    })
    cal_violations = bad_calendar.check_calendar_arbitrage()
    check(len(cal_violations) == 1, f"deliberately-broken calendar surface flags exactly 1 violation, got {len(cal_violations)}")
    if cal_violations:
        v = cal_violations[0]
        check(v["w_next"] < v["w_prev"],
              f"flagged violation actually has decreasing total variance: w_prev={v['w_prev']:.4f}, w_next={v['w_next']:.4f}")
    check(len(bad_calendar.check_butterfly_arbitrage()) == 0,
          "the calendar-broken surface has too few strikes per maturity to even run the butterfly check (correctly finds none, not falsely finds one)")

    # --- part 2b: deliberately construct a butterfly-arbitrage violation ---
    # Three strikes at the same maturity, with vol artificially spiked at
    # the middle strike. This inflates the middle option's price well
    # above the chord connecting its neighbors' prices, breaking
    # convexity in strike.
    bad_butterfly = VolSurface.from_grid(S, r, {
        (90.0, 1.0): 0.20,
        (100.0, 1.0): 1.50,  # spike
        (110.0, 1.0): 0.20,
    })
    bfly_violations = bad_butterfly.check_butterfly_arbitrage()
    check(len(bfly_violations) == 1, f"deliberately-spiked butterfly surface flags exactly 1 violation, got {len(bfly_violations)}")
    if bfly_violations:
        v = bfly_violations[0]
        check(v["slope_right"] < v["slope_left"],
              f"flagged violation actually has slope_right < slope_left: {v['slope_right']:.4f} < {v['slope_left']:.4f}")
    check(not bad_butterfly.is_arbitrage_free(), "is_arbitrage_free() is False for the spiked surface")

    print()
    if failures == 0:
        print("All vol surface checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
