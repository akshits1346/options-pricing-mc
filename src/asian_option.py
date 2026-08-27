"""
Monte Carlo pricer for Asian options (arithmetic-average payoff), with
an exact closed-form price for the GEOMETRIC-average case used purely
to validate the simulation machinery -- there is no simple closed form
for the arithmetic average (that's exactly why Monte Carlo is used for
it at all), but the geometric-average case IS solvable in closed form,
and getting that case right is strong evidence the path simulation,
discretization, and payoff/discounting logic are all correct before
trusting the arithmetic-average number that can't be checked directly.

CLOSED-FORM DERIVATION (discrete monitoring, n equally-spaced points):
Under risk-neutral GBM, ln(S_i) = ln(S0) + (r - 0.5*sigma^2)*t_i +
sigma*W(t_i). The geometric average G = (prod_{i=1}^n S_i)^(1/n), so
ln(G) is a linear combination of the (correlated) W(t_i), hence
Gaussian. Working out its mean and variance exactly:

    mean(ln G)     = ln(S0) + (r - 0.5*sigma^2) * T * (n+1)/(2n)
    variance(ln G) = sigma^2 * T * (n+1)*(2n+1) / (6*n^2)

(using Var(sum W(t_i)) = dt * sum_{i,j} min(i,j) = dt * n(n+1)(2n+1)/6,
a standard identity for partial sums of Brownian motion values at
equally-spaced times). Given ln(G) ~ Normal(mu, s^2), the discounted
expected call payoff has the standard lognormal-expectation closed
form used below.
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


def geometric_asian_call_closed_form(S0: float, K: float, r: float, sigma: float, T: float, n: int) -> float:
    mu = math.log(S0) + (r - 0.5 * sigma ** 2) * T * (n + 1) / (2 * n)
    variance = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n ** 2)
    s = math.sqrt(variance)

    d1 = (mu - math.log(K) + variance) / s
    d2 = d1 - s

    return math.exp(-r * T) * (math.exp(mu + variance / 2) * norm.cdf(d1) - K * norm.cdf(d2))


def monte_carlo_asian_call(S0: float, K: float, r: float, sigma: float, T: float,
                           n_steps: int, n_paths: int, average_type: str = "arithmetic",
                           antithetic: bool = False, seed: int = None,
                           paths: np.ndarray = None) -> MonteCarloResult:
    """
    average_type: "arithmetic" or "geometric" -- which average of the
    monitored prices (S_1 ... S_n, excluding S_0) defines the payoff.

    `paths` can be passed directly (pre-simulated) instead of simulating
    fresh ones here -- this exists specifically so a test can price both
    arithmetic and geometric payoffs against the EXACT SAME simulated
    paths, which is what makes the AM-GM pathwise-dominance test in
    test_asian_option.py an exact inequality rather than a statistical one.
    """
    if average_type not in ("arithmetic", "geometric"):
        raise ValueError("average_type must be 'arithmetic' or 'geometric'")

    if paths is None:
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=antithetic, seed=seed)

    monitored = paths[:, 1:]  # exclude column 0 (S0 itself) -- only S_1..S_n are monitored, matching the closed-form derivation

    if average_type == "arithmetic":
        averages = monitored.mean(axis=1)
    else:
        averages = np.exp(np.log(monitored).mean(axis=1))

    payoffs = np.maximum(averages - K, 0.0)
    discounted_payoffs = math.exp(-r * T) * payoffs

    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std(ddof=1) / math.sqrt(len(discounted_payoffs)))

    return MonteCarloResult(price=price, std_error=std_error)
