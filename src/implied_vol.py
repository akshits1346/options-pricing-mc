"""
Implied volatility solver: given an observed option price, find the
sigma that makes Black-Scholes reproduce it.

Newton-Raphson is fast (uses vega, the price's sensitivity to sigma,
as the derivative) but can fail or oscillate wildly when vega is very
small -- which happens deep ITM/OTM or very close to expiry, exactly
where the price surface is nearly flat in sigma and a small price
change implies a huge, poorly-determined change in sigma. The fallback
here: if Newton-Raphson doesn't converge within max_iterations, or a
step would push sigma negative, switch to bisection, which is slower
but can't diverge -- it just repeatedly halves a bracket known to
contain the root.

VALIDATION STRATEGY: round-trip test. Price an option at a KNOWN sigma
via Black-Scholes, then solve implied vol from that exact price -- the
recovered sigma should match the original to high precision. This is a
much stronger check than comparing to a textbook's stated answer: it
tests the solver against the exact function it's supposed to invert.
"""
import math
from dataclasses import dataclass

from src.black_scholes import black_scholes_price, black_scholes_greeks


@dataclass
class ImpliedVolResult:
    implied_vol: float
    iterations: int
    method: str  # "newton" or "bisection" -- which one actually converged


def implied_volatility(price: float, S: float, K: float, r: float, T: float,
                       option_type: str = "call", initial_guess: float = 0.2,
                       tol: float = 1e-8, max_iterations: int = 100) -> ImpliedVolResult:
    """
    NOTE ON THE CONVERGENCE CRITERION -- this matters more than it looks:
    Newton's step here converges on the SIGMA STEP SIZE (diff/vega), not
    on the raw price difference. An earlier version of this function
    checked abs(model_price - price) < tol directly, which FAILS
    silently for deep OTM / near-expiry options: there, every price is
    astronomically close to zero regardless of sigma (e.g. 2e-92 vs
    2e-42 -- both "zero" in absolute terms, but 50 orders of magnitude
    apart, and corresponding to completely different implied vols). A
    price-space tolerance is trivially satisfied by two wildly different
    sigmas whenever both give near-zero prices, causing false
    convergence. Checking the sigma STEP size instead is scale-invariant
    and doesn't have this failure mode -- and naturally forces a
    fallback to bisection when vega is too small to trust (a huge step
    size from a tiny vega correctly signals "not converged" rather than
    the previous version's false "converged").
    """
    sigma = initial_guess

    for i in range(max_iterations):
        try:
            model_price = black_scholes_price(S, K, r, sigma, T, option_type)
        except ValueError:
            break  # sigma went non-positive or otherwise invalid -- stop and fall back to bisection

        diff = model_price - price
        vega = black_scholes_greeks(S, K, r, sigma, T, option_type).vega

        if vega < 1e-8:
            break  # vega too small to trust a Newton step -- fall back to bisection

        step = diff / vega
        if abs(step) < tol:
            return ImpliedVolResult(implied_vol=sigma, iterations=i + 1, method="newton")

        sigma = sigma - step
        if sigma <= 0:
            break  # stepped into invalid territory -- fall back to bisection

    # --- bisection fallback ---
    lo, hi = 1e-6, 5.0  # 500% vol is a safe upper bound for basically any real option
    price_lo = black_scholes_price(S, K, r, lo, T, option_type) - price
    price_hi = black_scholes_price(S, K, r, hi, T, option_type) - price

    if price_lo * price_hi > 0:
        raise ValueError(
            f"no implied volatility in [{lo}, {hi}] brackets the target price {price} -- "
            "the price may be outside any arbitrage-free range for these parameters"
        )

    for i in range(max_iterations):
        mid = (lo + hi) / 2
        price_mid = black_scholes_price(S, K, r, mid, T, option_type) - price

        if abs(price_mid) < tol or (hi - lo) / 2 < tol:
            return ImpliedVolResult(implied_vol=mid, iterations=i + 1, method="bisection")

        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo = mid
            price_lo = price_mid

    return ImpliedVolResult(implied_vol=(lo + hi) / 2, iterations=max_iterations, method="bisection")
