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
    If `paths` is supplied directly, the `antithetic` argument here must
    still correctly describe whether those paths were generated
    antithetically -- it controls how std_error is computed, not just
    how paths are simulated.
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
    std_error = _standard_error(discounted_payoffs, antithetic)

    return MonteCarloResult(price=price, std_error=std_error)


def monte_carlo_asian_call_control_variate(S0: float, K: float, r: float, sigma: float, T: float,
                                           n_steps: int, n_paths: int,
                                           antithetic: bool = False, seed: int = None,
                                           paths: np.ndarray = None) -> MonteCarloResult:
    """
    Control variate variance reduction for the arithmetic Asian call,
    using the geometric Asian's EXACT closed form
    (geometric_asian_call_closed_form) as the control. On the SAME
    simulated paths, the arithmetic and geometric averages are very
    highly correlated -- they're both averages of the identical price
    path, just averaged differently -- which is exactly the condition
    that makes a control variate effective.

    Standard control variate estimator, applied per-path:
        Y_cv_i = Y_i - beta * (X_i - E[X])
    where Y_i is the discounted arithmetic payoff on path i, X_i is the
    discounted GEOMETRIC payoff on that SAME path i, E[X] is the EXACT
    geometric closed-form price (known exactly, not estimated -- that's
    the entire point of a control variate), and
        beta_hat = sample_Cov(Y, X) / sample_Var(X)
    is chosen from the same sample to minimize the resulting variance.
    mean(Y_cv) is still an unbiased estimator of E[Y] for ANY value of
    beta (the (X_i - E[X]) term has mean zero by construction, since
    E[X] is the true mean, not the sample mean) -- estimating beta from
    the same sample only affects how MUCH variance is removed, not
    whether the estimator is still valid.
    """
    if paths is None:
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=antithetic, seed=seed)

    monitored = paths[:, 1:]
    n = monitored.shape[1]  # number of monitored points, matches the closed form's n

    arithmetic_avg = monitored.mean(axis=1)
    geometric_avg = np.exp(np.log(monitored).mean(axis=1))

    discount = math.exp(-r * T)
    Y = discount * np.maximum(arithmetic_avg - K, 0.0)
    X = discount * np.maximum(geometric_avg - K, 0.0)
    exact_X_mean = geometric_asian_call_closed_form(S0, K, r, sigma, T, n)

    beta_hat = np.cov(Y, X, ddof=1)[0, 1] / np.var(X, ddof=1)
    Y_cv = Y - beta_hat * (X - exact_X_mean)

    price = float(Y_cv.mean())
    std_error = _standard_error(Y_cv, antithetic)

    return MonteCarloResult(price=price, std_error=std_error)


def _standard_error(discounted_payoffs: np.ndarray, antithetic: bool) -> float:
    """
    THIS FUNCTION EXISTS BECAUSE OF A BUG I FOUND WHILE TESTING: computing
    std_error as discounted_payoffs.std(ddof=1)/sqrt(n) treats every path
    as an independent sample, which is correct for plain Monte Carlo but
    WRONG for antithetic sampling -- half the paths are deliberately
    correlated (negatively) with the other half, by construction. Treating
    them as independent doesn't just give a slightly-off number, it hides
    the entire benefit of antithetic variates: a direct empirical check
    (see the commit that introduced this fix) showed antithetic pairs have
    a real correlation of about -0.50, but the naive standard error showed
    ~0% variance reduction instead of the ~50% the correlation implies.

    The correct approach for antithetic sampling: form n/2 pair-averages
    ((Y_i + Y_i')/2 for each antithetic pair), then compute standard
    error across THOSE n/2 values, not across all n raw payoffs.
    """
    n = len(discounted_payoffs)
    if not antithetic:
        return float(discounted_payoffs.std(ddof=1) / math.sqrt(n))

    half = n // 2
    pair_averages = (discounted_payoffs[:half] + discounted_payoffs[half:]) / 2.0
    return float(pair_averages.std(ddof=1) / math.sqrt(half))
