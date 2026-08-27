"""
Plain vanilla European option pricing via Monte Carlo -- the simplest
case, used mainly to confirm the whole path-simulation/discounting
pipeline agrees with Black-Scholes before trusting the more complex
Asian/barrier pricers built on the same machinery (see asian_option.py,
barrier_option.py).
"""
import math
from dataclasses import dataclass

import numpy as np

from src.gbm_paths import simulate_gbm_paths


@dataclass
class MonteCarloResult:
    price: float
    std_error: float


def _standard_error(discounted_payoffs: np.ndarray, antithetic: bool) -> float:
    """Same antithetic pair-averaging fix as in asian_option.py and
    barrier_option.py -- see asian_option.py's _standard_error docstring
    for the full story of why this matters."""
    n = len(discounted_payoffs)
    if not antithetic:
        return float(discounted_payoffs.std(ddof=1) / math.sqrt(n))
    half = n // 2
    pair_averages = (discounted_payoffs[:half] + discounted_payoffs[half:]) / 2.0
    return float(pair_averages.std(ddof=1) / math.sqrt(half))


def monte_carlo_vanilla(S0: float, K: float, r: float, sigma: float, T: float,
                        n_paths: int, option_type: str = "call",
                        antithetic: bool = False, seed: int = None) -> MonteCarloResult:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    # single-step simulation is all a vanilla (non-path-dependent) payoff needs
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=1, n_paths=n_paths, antithetic=antithetic, seed=seed)
    terminal = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal, 0.0)

    discounted_payoffs = math.exp(-r * T) * payoffs
    price = float(discounted_payoffs.mean())
    std_error = _standard_error(discounted_payoffs, antithetic)

    return MonteCarloResult(price=price, std_error=std_error)
