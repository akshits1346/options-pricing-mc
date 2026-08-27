"""
Monte Carlo pricer for a down-and-out call: a standard European call
that pays nothing if the underlying ever touches or drops below a
barrier level at any monitored point, and pays the usual max(S_T-K,0)
otherwise.

VALIDATION STRATEGY: down-and-out price can never EXCEED the vanilla
European call price -- knocking out only ever zeroes out payoffs that
the vanilla call would otherwise have paid, it never adds anything.
On the SAME simulated paths, this is an exact (not statistical)
pointwise inequality: down_and_out_payoff <= vanilla_payoff for every
single path, always. That gives a rigorous test independent of any
statistical convergence argument. Separately, as the barrier is pushed
far below any plausible path (i.e. essentially unreachable), the price
should converge toward the vanilla Black-Scholes price -- a genuine
statistical convergence check, since at that point almost no simulated
path ever gets knocked out.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.gbm_paths import simulate_gbm_paths


@dataclass
class MonteCarloResult:
    price: float
    std_error: float


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Standard closed-form vanilla European call, used only as a
    reference point for the barrier-option convergence check below."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def monte_carlo_down_and_out_call(S0: float, K: float, r: float, sigma: float, T: float,
                                   barrier: float, n_steps: int, n_paths: int,
                                   antithetic: bool = False, seed: int = None,
                                   paths: np.ndarray = None) -> MonteCarloResult:
    if barrier >= S0:
        raise ValueError("barrier must be below the current spot price S0 for a down-and-out option")

    if paths is None:
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=antithetic, seed=seed)

    monitored = paths[:, 1:]  # S_1 .. S_n, same monitoring convention as the Asian pricer
    knocked_out = (monitored <= barrier).any(axis=1)

    terminal_prices = paths[:, -1]
    vanilla_payoffs = np.maximum(terminal_prices - K, 0.0)
    payoffs = np.where(knocked_out, 0.0, vanilla_payoffs)

    discounted_payoffs = math.exp(-r * T) * payoffs
    price = float(discounted_payoffs.mean())
    std_error = _standard_error(discounted_payoffs, antithetic)

    return MonteCarloResult(price=price, std_error=std_error)


def _standard_error(discounted_payoffs: np.ndarray, antithetic: bool) -> float:
    """Same fix as in asian_option.py's _standard_error -- see that
    function's docstring for the full explanation of why antithetic
    sampling needs pair-averaged standard error, not a naive std over
    all paths treated as independent."""
    n = len(discounted_payoffs)
    if not antithetic:
        return float(discounted_payoffs.std(ddof=1) / math.sqrt(n))

    half = n // 2
    pair_averages = (discounted_payoffs[:half] + discounted_payoffs[half:]) / 2.0
    return float(pair_averages.std(ddof=1) / math.sqrt(half))
